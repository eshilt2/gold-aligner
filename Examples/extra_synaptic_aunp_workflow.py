import numpy as np
from cryoet_alignment import read, write 
from cryoet_alignment.io.cryoet_data_portal import Alignment
import mrcfile
# import matplotlib.pyplot as plt
import os
import re
import math
import subprocess
import glob
import pandas as pd
import csv
# from gold_aligner.align_gold import realign_gold
# from gold_aligner.fix_alpha_offset import fix_alpha_offset 
from gold_aligner._convert_imod_to_aretomo_aln import imod_to_aretomo, aretomo_to_imod, full_aretomo_to_imod
import imodmodel
from gold_aligner._align_to_tilt_series import *
from gold_aligner._select_aunps import *
from gold_aligner._convolution_and_cross_correlation import *
from gold_aligner._make_IMOD_model import *
from gold_aligner._fit_gaussian import *
from gold_aligner._find_clusters import*

from scipy import ndimage as ndi

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from gold_aligner.fix_alpha_offset import fix_alpha_offset

from torch_grid_utils.fftfreq_grid import dft_center
from torch_grid_utils.coordinate_grid import coordinate_grid
from torch.nn.functional import conv2d

def find_3d_gauss_peak(invert_tomo,     # array : initially aligned data from tomogram.mrc * -1 
                           peak_coords,     # list: list of coordinate list [x,y,z]
                           cutoff = 95      # int: disregaurd points with sigmas above certain percentile
                           ): 
    new_point_coords = []
    list_of_sigmas = []
    list_of_fitted = []
    for entry in peak_coords:

        # crops area around each point
        selected_area = invert_tomo[entry[2]-5:entry[2]+5,entry[1]-5:entry[1]+5, entry[0]-5:entry[0]+5]
        selected_area = selected_area - selected_area.min()
        area_shape = selected_area.shape

        x = np.arange(area_shape[2])
        y = np.arange(area_shape[1])
        z = np.arange(area_shape[0])

        z, y, x = np.meshgrid(z, y, x, indexing='ij')
        coords = (x, y, z)
        guess = [area_shape[0]/2, area_shape[1]/2, area_shape[2]/2, selected_area.max(), area_shape[0]/5, area_shape[1]/5, area_shape[2]/5, selected_area.min()]
        bounds = ([0, 0, 0, 0, 0.1, 0.1, 0.1, -np.inf], [7, 7, 7, np.inf, 2.25, 2.25, 5, np.inf])
        fit, _ = curve_fit(get_3d_gaussian, coords, selected_area.ravel(), p0=guess, bounds = bounds, method = 'dogbox')
        x0, y0, z0, amp, sigx, sigy, sigz, back = fit
        list_of_fitted.append(get_3d_gaussian((x,y,z),*fit).reshape(area_shape))
        x_c = x0-5 + entry[0]
        y_c = y0-5 + entry[1]
        z_c = z0-5 + entry[2]
        new_point_coords.append([x_c,y_c,z_c])
        list_of_sigmas.append([sigx,sigy,sigz])
        
    new_point_coords = np.array(new_point_coords)

    unzip_sig = list(zip(*list_of_sigmas))
    cutoff_value = [np.percentile(np.array(unzip_sig[dim]), cutoff) for dim in range(3)] 
    indx_discard = [np.where(unzip_sig[dim] >= cutoff_value[dim]) for dim in range(3)]
    discard = np.concatenate([indx_discard[0][0], indx_discard[1][0], indx_discard[2][0]])
    peak_coords_OI = np.delete(new_point_coords, discard, axis = 0)
    list_of_fitted = np.delete(list_of_fitted, discard, axis = 0)
    list_of_sigmas = np.delete(list_of_sigmas, discard, axis = 0)
    return peak_coords_OI, list_of_fitted, list_of_sigmas 


def realign_custom( tomo_path,              # str   : path and name of tomogram
                    tilt_path,              # str   : path and name of tilt series
                    aln_path,               # str   : path and name of aligned tomogram
                    output_aln_path,        # str   : path and name of outputed aln file
                    center_OI,              # (int, int, int)    : center of cylinder that will determine selected aunps in radius around center
                    radius_OI,              # int   : radius around center_OI where aunps will be selected
                    bin,                    # int   : binning of image
                    date,                   # str   : used by me to save cross corr images to folder called whatever the date in Pictures
                    rel_threshold = 0.3,
                    alpha_offset = 0,           # flt   : ensures proper alpha offset of imod file
                    border_cutoff = (70,50,50) # (int, int, int): creates mask around border of tomo to not select peaks
                    ):
    ### Load in all files ################################################
    with mrcfile.open(tomo_path) as mrctomo: # get tomo data
        invert_tomo = mrctomo.data * -1 #flip black and white so peak_local_max picks up dark points
        tomo_shape = invert_tomo.shape
    
    read_alignment = read(aln_path)
    if hasattr(read_alignment, 'xf') == True:
        aretomo3_alignment = Alignment.from_imod(read_alignment)
        with mrcfile.open(tilt_path) as mrctilt:
            tilt = mrctilt.data
            tilt_shape = tilt.shape
            invert_shape = (tilt_shape[2], tilt_shape[1], tilt_shape[0])
        aretomo3_alignment = aretomo3_alignment.to_aretomo(ts_size=invert_shape)
        if aretomo3_alignment.AlphaOffset != alpha_offset:
            aretomo3_alignment = fix_alpha_offset(aretomo3_alignment, alpha_offset)

    else:
        aretomo3_alignment = read_alignment
        if aretomo3_alignment.AlphaOffset != alpha_offset:
            aretomo3_alignment = fix_alpha_offset(aretomo3_alignment, alpha_offset)

    with mrcfile.open(tilt_path) as mrctilt:
        if aretomo3_alignment.DarkFrames == []:
            tilt = mrctilt.data
            tilt_shape = tilt.shape
        else:
            tilt = mrctilt.data
            dark_slices = [frame.section_idx for frame in aretomo3_alignment.DarkFrames]
            tilt = np.delete(tilt, dark_slices, axis = 0)
            tilt_shape = tilt.shape 
    print('loaded files')        

    tomo_num = re.findall(r"(._\d*_*).mrc",tilt_path)[0] # gets tomo number for im
    tomo_name = re.findall(r".*/(.*).mrc",tilt_path)[0]
    ######################################################################
    # _, peak_coords = select_aunps(invert_tomo, rel_threshold, border_cutoff, center_OI, radius_OI)
    peak_coords = select_aunps_rect_region(invert_tomo, rel_threshold, border_cutoff, (393,352), (742,793)) 
    
    while len(peak_coords) >= 75 or len(peak_coords) <= 9:
        rel_threshold += 0.25
        print(rel_threshold, len(peak_coords))
        peak_coords = select_aunps_rect_region(invert_tomo, rel_threshold, border_cutoff, (393,352), (742,793)) 
        while len(peak_coords) <= 9:
            rel_threshold -= 0.1
            peak_coords = select_aunps_rect_region(invert_tomo, rel_threshold, border_cutoff, (393,352), (742,793)) 
    # peaks = [ent for ent in peak_coords if center_OI[0]+radius_OI > ent[0] and ent[0] >center_OI[0]-radius_OI and center_OI[1]+radius_OI > ent[1] and ent[1] >center_OI[1]-radius_OI]
    peaks = peak_coords
    peak_coords_OI, _, list_of_sigmas = find_3d_gauss_peak(invert_tomo, peaks)
    make_IMOD_model_UPDATED(peak_coords_OI, f'/nrs/liza/cathy_tomos/ddw/imod_alignments/extra_synaptic_AuNP/EGmilled24-2_68_main_az_plus/{tomo_name}')
    plot_3d_sigmas(list_of_sigmas, tomo_name, f'/groups/liza/Pictures/{date}/{tomo_name}:_{len(list_of_sigmas)}')
    plt.close()
    print('made model')
    print('')

    final_coords = align_to_tilt_series(tomo_shape, tilt_shape, peak_coords_OI, bin, aretomo3_alignment)
    make_IMOD_model_UPDATED(final_coords, f'/nrs/liza/cathy_tomos/ddw/imod_alignments/extra_synaptic_AuNP/EGmilled24-2_68_main_az_plus/{tomo_name}_tilt')
    base_img, circle_img = make_model_layer_components(tilt, aretomo3_alignment, 7, final_coords, shape = 'circle')
    
    # Convolve cirlce with pixel placement
    conv_image, rev_conv_coords = fourier_convolution(circle_img, base_img, False)
    cropped_phase, shift, saved_cropped_phase = cross_corr(tilt*-1, conv_image)
    fig1 = plot_xcorr_peaks(saved_cropped_phase, shift, f'{tomo_name} cross corr')
    plt.savefig(f"/groups/liza/Pictures/{date}/{tomo_name}:_{len(list_of_sigmas)}_xcorr.png", dpi=300)
    plt.close()

    ideal_sigma, auto_shift, auto_cropped_phase = cross_corr(conv_image, conv_image)
    fig2 = plot_xcorr_peaks(auto_cropped_phase, auto_shift, f'Auto xcorr conv_img {tomo_name}')
    plt.savefig(f"/groups/liza/Pictures/{date}/{tomo_name}:_{len(list_of_sigmas)}_auto_xcorr_mask.png", dpi=300)

    _, autoxcorr_shift, autoxcorr_phase = cross_corr(tilt * -1, conv_image, ideal_sigma)
    fig3 = plot_xcorr_peaks(autoxcorr_phase, autoxcorr_shift, f'{tomo_name} with ideal sigma')
    plt.savefig(f"/groups/liza/Pictures/{date}/{tomo_name}:_{len(list_of_sigmas)}_autoxcorrxcorr_with_fixed_sigma.png", dpi=300)
    plt.close()

    if alpha_offset != aretomo3_alignment.AlphaOffset and alpha_offset != None: # checks if alpha offset is incorrect and adjusts it
        aretomo3_alignment = fix_alpha_offset(aretomo3_alignment, alpha_offset)
    
    for i, algnmt in enumerate(aretomo3_alignment.GlobalAlignments):
        algnmt.tx = algnmt.tx + shift[i][0]
        algnmt.ty = algnmt.ty + shift[i][1]

    aretomo3_alignment.LocalAlignments = [] # if this was run w local patch correction, this removes that 
    aretomo3_alignment.NumPatches = 0 # also to revert from initial local patch correction to global only
    imod_aln = aretomo_to_imod(aretomo3_alignment, tilt_shape, 2.5)
    write(imod_aln, f"/nrs/liza/cathy_tomos/ddw/imod_alignments/extra_synaptic_AuNP/EGmilled24-2_68_main_az_plus/20231017_EGmilled24-2_68")
    write(aretomo3_alignment, output_aln_path)

def just_fit_sigmas(og_tomo_path, new_tomo_path, center_OI, radius_OI, date):
    border_cutoff = (70,50,50)
    with mrcfile.open(og_tomo_path) as mrctomo: # get tomo data
        og_invert_tomo = mrctomo.data * -1 #flip black and white so peak_local_max picks up dark points
    
    with mrcfile.open(new_tomo_path) as mrctomo: # get tomo data
        new_invert_tomo = mrctomo.data * -1 #flip black and white so peak_local_max picks up dark points
    
    _, og_peaks = select_aunps(og_invert_tomo, 0.3, border_cutoff, center_OI, radius_OI)
    _, new_peaks = select_aunps(new_invert_tomo, 0.3, border_cutoff, center_OI, radius_OI)
    og_peak_OI, _, og_list_of_sigmas = find_3d_gaussian_peaks(og_invert_tomo, og_peaks)
    new_peak_OI, _, new_list_of_sigmas = find_3d_gaussian_peaks(new_invert_tomo, new_peaks)

    compare_3d_sigmas(og_list_of_sigmas, 'original', new_list_of_sigmas, 're-aligned', f'/groups/liza/Pictures/{date}/comparison_of_alignment_boxplot_EGmilled24-2_68.png', '20231017_EGmilled24-2_68 comparison')

if __name__ == '__main__':
    og_tomo_path = '/scratch/pompeii/elferich/gouaux_tomo/tomograms/15F1_tomograms/TOP_TOMOS/20231017_EGmilled24-2_68/best_alignment/20231017_EGmilled24-2_68_full_rec_BP_3DCTF_BIN4.mrc'
    new_tomo_path = '/nrs/liza/cathy_tomos/ddw/imod_alignments/extra_synaptic_AuNP/EGmilled24-2_68_x515_y485/20231017_EGmilled24-2_68_full_rec_BP_3DCTF_BIN4.mrc'
    
    just_fit_sigmas(og_tomo_path, new_tomo_path, (515, 485), 120, '11_07_2025')


    aln_path = '/scratch/pompeii/elferich/gouaux_tomo/tomograms/15F1_tomograms/TOP_TOMOS/20231017_EGmilled24-2_68/best_alignment/20231017_EGmilled24-2_68'
    tilt_path = '/scratch/pompeii/elferich/gouaux_tomo/tomograms/15F1_tomograms/TOP_TOMOS/20231017_EGmilled24-2_68/20231017_EGmilled24-2_68.mrc'
    tilt_com_path = '/scratch/pompeii/elferich/gouaux_tomo/tomograms/15F1_tomograms/TOP_TOMOS/20231017_EGmilled24-2_68/best_alignment/tilt.com'
    output_aln_path = '/nrs/liza/cathy_tomos/ddw/imod_alignments/extra_synaptic_AuNP/EGmilled24-2_68_main_az_plus/20231017_EGmilled24-2_68.aln'
    aln = read(aln_path)
    with mrcfile.open(tilt_path) as mrctilt:
        tilt = mrctilt.data
        tilt_shape = tilt.shape
        invert_shape = (tilt_shape[2], tilt_shape[1], tilt_shape[0])

    for line in open(tilt_com_path):
        if re.findall(r'OFFSET*', line) == ['OFFSET']:
            match = re.findall(r'[0-9]+.[0-9]+', line)
            alphaOffset = float(match[0])

    tomo_path = '/scratch/pompeii/elferich/gouaux_tomo/tomograms/15F1_tomograms/TOP_TOMOS/20231017_EGmilled24-2_68/best_alignment/20231017_EGmilled24-2_68_full_rec_BP_3DCTF_BIN4.mrc'

    realign_custom(tomo_path, tilt_path, aln_path, output_aln_path, (748,830), 120, 4, "10_23_2025", 0.3, alphaOffset)
    print('done')
