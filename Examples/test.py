# dual aunp workflow

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
# from scipy.spatial import KDTree

def realign_with_mono_selected(picks,               # array : preselected picks used as initial search points for aunps
                            tomo_path,              # str   : path and name of tomogram
                            tilt_path,              # str   : path and name of tilt series
                            aln_path,               # str   : path and name of aligned tomogram
                            output_aln_path,        # str   : path and name of outputed aln file
                            center_OI,              # (int, int, int)    : center of cylinder that will determine selected aunps in radius around center
                            radius_OI,              # int   : radius around center_OI where aunps will be selected
                            bin,                    # int   : binning of image
                            alpha_offset,           # flt   : ensures proper alpha offset of imod file
                            date,                   # str   : used by me to save cross corr images to folder called whatever the date in Pictures
                            number                  # int   : denotes az
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
    else:
        aretomo3_alignment = read_alignment

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

    peaks = [ent for ent in picks if center_OI[0]+radius_OI > ent[0] and ent[0] >center_OI[0]-radius_OI and center_OI[1]+radius_OI > ent[1] and ent[1] >center_OI[1]-radius_OI]
    peak_coords_OI, _, list_of_sigmas = find_3d_gaussian_peaks(invert_tomo, peaks)
    make_IMOD_model_UPDATED(peak_coords_OI, f'//nrs/liza/cathy_tomos/ddw/imod_alignments/15F1and5F11_TOPTOMOS_inprogress/{tomo_name}/{tomo_name}_azs_mpicks_{number}')
    plot_3d_sigmas(list_of_sigmas, tomo_name, f'/groups/liza/Pictures/{date}/{tomo_name}_az{number}:_{len(list_of_sigmas)}')
    plt.close()
    print('made model')
    print('')

    final_coords = align_to_tilt_series(tomo_shape, tilt_shape, peak_coords_OI, bin, aretomo3_alignment)

    base_img, circle_img = make_model_layer_components(tilt, aretomo3_alignment, 8, final_coords, shape = 'circle')
    
    # Convolve cirlce with pixel placement
    conv_image, rev_conv_coords = fourier_convolution(circle_img, base_img, False)
    f = final_coords[:-1] +1
    make_IMOD_model_UPDATED(final_coords, f'/nrs/liza/cathy_tomos/ddw/imod_alignments/15F1and5F11_TOPTOMOS_inprogress/{tomo_name}/{tomo_name}_ZEROalphaOffset')


    cropped_phase, shift, saved_cropped_phase = cross_corr(tilt*-1, conv_image)
    fig1 = plot_xcorr_peaks(saved_cropped_phase, shift, f'{tomo_name}_az{number} cross corr')
    plt.savefig(f"/groups/liza/Pictures/{date}/{tomo_name}_az{number}:_{len(list_of_sigmas)}_xcorr.png", dpi=300)
    plt.close()

    ideal_sigma, auto_shift, auto_cropped_phase = cross_corr(conv_image, conv_image)
    fig2 = plot_xcorr_peaks(auto_cropped_phase, auto_shift, f'Auto xcorr conv_img {tomo_name}_az{number}')
    plt.savefig(f"/groups/liza/Pictures/{date}/{tomo_name}_az{number}:_{len(list_of_sigmas)}_auto_xcorr_mask.png", dpi=300)

    _, autoxcorr_shift, autoxcorr_phase = cross_corr(tilt * -1, conv_image, ideal_sigma)
    fig3 = plot_xcorr_peaks(autoxcorr_phase, autoxcorr_shift, f'{tomo_name}_az{number} with ideal sigma')
    plt.savefig(f"/groups/liza/Pictures/{date}/{tomo_name}_az{number}:_{len(list_of_sigmas)}_autoxcorrxcorr_with_fixed_sigma.png", dpi=300)
    plt.close()

    if alpha_offset != aretomo3_alignment.AlphaOffset and alpha_offset != None: # checks if alpha offset is incorrect and adjusts it
        aretomo3_alignment = fix_alpha_offset(aretomo3_alignment, alpha_offset)
    
    for i, algnmt in enumerate(aretomo3_alignment.GlobalAlignments):
        algnmt.tx = algnmt.tx + shift[i][0] 
        algnmt.ty = algnmt.ty + shift[i][1]

    aretomo3_alignment.LocalAlignments = [] # if this was run w local patch correction, this removes that 
    aretomo3_alignment.NumPatches = 0 # also to revert from initial local patch correction to global only

    imod_aln = aretomo_to_imod(aretomo3_alignment, tilt_shape, 2.5)

    write(imod_aln, f"{output_aln_path}/{tomo_name}")
    write(aretomo3_alignment, f"{output_aln_path}/{tomo_name}.aln")


if __name__ == "__main__":
# grab tomo name --> read in mono coords --> set tomo to 3DCTF corrected tomo --> aln to init aln
    import starfile
    import ast
    
    aunps = starfile.read(f"/nrs/liza/cathy_tomos/15F1and5F11_TOPTOMOS/20251117_AMmilled40-2_Position_166/patch_tracking/manual_picking_aunp_HK_re.star")
    aunps = aunps[["faCoordinateZ", "faCoordinateY", "faCoordinateX", "type"]]

    picks = aunps[["faCoordinateZ", "faCoordinateY", "faCoordinateX"]].to_numpy()
    picks = np.array(picks)
    make_IMOD_model_UPDATED(aunps, f'/nrs/liza/cathy_tomos/15F1and5F11_TOPTOMOS/20251117_AMmilled40-2_Position_166/imod_model')
    tomo = f"/nrs/liza/cathy_tomos/15F1and5F11_TOPTOMOS/20251117_AMmilled40-2_Position_166/patch_tracking/20251117_AMmilled40-2_Position_166_full_rec_BP_3DCTF_BIN4.mrc"
    tilt = f"/nrs/liza/cathy_tomos/15F1and5F11_TOPTOMOS/20251117_AMmilled40-2_Position_166/20251117_AMmilled40-2_Position_166.mrc"
    aln = f"/nrs/liza/cathy_tomos/15F1and5F11_TOPTOMOS/20251117_AMmilled40-2_Position_166/patch_tracking/20251117_AMmilled40-2_Position_166"
    output_path = f"/nrs/liza/cathy_tomos/ddw/imod_alignments/15F1and5F11_TOPTOMOS_inprogress/20251117_AMmilled40-2_Position_166/test/"
    tilt_com_path = "/nrs/liza/cathy_tomos/15F1and5F11_TOPTOMOS/20251117_AMmilled40-2_Position_166/patch_tracking/tilt.com"

    for line in open(tilt_com_path):
        if re.findall(r'OFFSET*', line) == ['OFFSET']:
            match = re.findall(r'[0-9]+.[0-9]+', line)
            alphaOffset = float(match[0])

    realign_with_mono_selected(picks, tomo, tilt, aln, output_path, (205,347), 300, 4,  alphaOffset, "12_05_2025/test", 0)
    # realign_with_mono_selected(picks, tomo, tilt, aln, output_path, (235,526), 120, 4,  0, "12_05_2025", 0)

