# Looking into multiple tomograms, it seems that shifts go more extreme w tilts but ¯\_(ツ)_/¯
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

from datetime import datetime

from scipy import ndimage as ndi

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from gold_aligner.fix_alpha_offset import fix_alpha_offset

from torch_grid_utils.fftfreq_grid import dft_center
from torch_grid_utils.coordinate_grid import coordinate_grid
from torch.nn.functional import conv2d


def get_shifts( tomo_path,              # str   : path and name of tomogram
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
    _, peak_coords = select_aunps(invert_tomo, rel_threshold, border_cutoff, center_OI, radius_OI)
    # peak_coords = select_aunps_rect_region(invert_tomo, rel_threshold, border_cutoff, (393,352), (742,793)) 
    
    while len(peak_coords) >= 150 or len(peak_coords) <= 9:
        rel_threshold += 0.25
        print(rel_threshold, len(peak_coords))
        _, peak_coords = select_aunps(invert_tomo, rel_threshold, border_cutoff, center_OI, radius_OI)
        while len(peak_coords) <= 15:
            rel_threshold -= 0.1
            _, peak_coords = select_aunps(invert_tomo, rel_threshold, border_cutoff, center_OI, radius_OI)
    peaks = [ent for ent in peak_coords if center_OI[0]+radius_OI > ent[0] and ent[0] >center_OI[0]-radius_OI and center_OI[1]+radius_OI > ent[1] and ent[1] >center_OI[1]-radius_OI]
    peaks = peak_coords
    peak_coords_OI, _, list_of_sigmas = find_3d_gaussian_peaks(invert_tomo, peaks)
    selected_points = peak_coords_OI
    selected_sigmas = list_of_sigmas

    low_xy_sigma = 0.5
    low_z_sigma = 1
    while len(selected_points) >= 50:
        filtered_coords, filtered_sigmas = zip(*[
            (coord, sigma) for coord, sigma in zip(selected_points, selected_sigmas)
            if sigma[0] >= low_xy_sigma and sigma[1] >= low_xy_sigma and sigma[2] >= low_z_sigma and sigma[2] < 4
            ])
        filtered_coords = np.array(filtered_coords)
        filtered_sigmas = np.array(filtered_sigmas)

        pred = cluster_points(filtered_coords, f'/groups/liza/Pictures/10_22_2025/{tomo_name}_{len(peak_coords)}', True)
        selected_points = filtered_coords[pred != -1]
        selected_sigmas = filtered_sigmas[pred != -1]
        low_xy_sigma += 0.025
        low_z_sigma += 0.025
    plt.imshow(np.max(invert_tomo, axis = 0))
    plt.plot(peak_coords_OI[:,0], peak_coords_OI[:,1], 'rx')
    plt.savefig(f'/groups/liza/Pictures/{date}/{tomo_name}_picks.png', dpi = 300)
    plot_3d_sigmas(list_of_sigmas, tomo_name, f'/groups/liza/Pictures/{date}/{tomo_name}:_{len(list_of_sigmas)}')
    plt.close()
    print('made model')
    print('')

    final_coords = align_to_tilt_series(tomo_shape, tilt_shape, peak_coords_OI, bin, aretomo3_alignment)
    # make_IMOD_model_UPDATED(final_coords, f'/nrs/liza/cathy_tomos/ddw/imod_alignments/extra_synaptic_AuNP/EGmilled24-2_68_main_az_plus/{tomo_name}_tilt')
    base_img, circle_img = make_model_layer_components(tilt, aretomo3_alignment, 7, final_coords, shape = 'circle')
    
    # Convolve cirlce with pixel placement
    conv_image, rev_conv_coords = fourier_convolution(circle_img, base_img, False)
    cropped_phase, shift, saved_cropped_phase = cross_corr(tilt*-1, conv_image) # init cross corr
    ideal_sigma, auto_shift, auto_cropped_phase = cross_corr(conv_image, conv_image) # cross corr of image w image to get what ideal fit looks like
    _, autoxcorr_shift, autoxcorr_phase = cross_corr(tilt * -1, conv_image, ideal_sigma) # cross corr of ideal sigma w actual image
    fig3 = plot_xcorr_peaks(autoxcorr_phase, autoxcorr_shift, f'{tomo_name} with ideal sigma')
    plt.savefig(f"/groups/liza/Pictures/{date}/{tomo_name}:_{len(list_of_sigmas)}_autoxcorrxcorr_with_fixed_sigma.png", dpi=300)

    plt.figure()
    tilt_ang = [t.tilt for t in aretomo3_alignment.GlobalAlignments]
    plt.plot(tilt_ang, autoxcorr_shift[:,0], label = 'x shift')
    plt.plot(tilt_ang, autoxcorr_shift[:,1], label = 'y shift')
    plt.hlines(y=0, xmin = np.array(tilt_ang).min(), xmax= (np.array(tilt_ang).max()), colors = 'black', linestyles = '--')
    plt.title(f'{tomo_name} shift vs tilt')
    plt.legend()
    plt.savefig(f'/groups/liza/Pictures/{date}/{tomo_name}_shiftvtilt')
    plt.close('all')
    return autoxcorr_shift, tilt_ang




df = pd.DataFrame(columns = ['tomo_name', 'shift', 'tilt_ang'])

center = np.genfromtxt('/groups/liza/Downloads/Tomograms and centers - 15f1_tomo_centers.csv', delimiter=',', skip_header = 1)    
tuple_centers = [(c[2],c[1]) for c in center]


path_to_all_folders = '/nrs/elferich/top_exclude/top_exclude/'
folder_list = os.listdir(path_to_all_folders)
skip_list = ['20240523_HippWaffle_125']
for i, folder in enumerate(folder_list):
    if os.path.exists(f"{path_to_all_folders}{folder}/fiducial_tracking/active_zonograms"):
        folder_path = f"{path_to_all_folders}{folder}/fiducial_tracking"
        aln_path = f"{path_to_all_folders}{folder}/fiducial_tracking/{folder}"
    else:             
        aln_path = f"{path_to_all_folders}{folder}/patch_tracking/{folder}"
        folder_path = f"{path_to_all_folders}{folder}/patch_tracking"
            
    tilt_path = f'{path_to_all_folders}/{folder}/{folder}.mrc'
    tilt_com_path = f'{folder_path}/tilt.com'

    aln = read(aln_path)
    with mrcfile.open(tilt_path) as mrctilt:
        tilt = mrctilt.data
        tilt_shape = tilt.shape
        invert_shape = (tilt_shape[2], tilt_shape[1], tilt_shape[0])

    for line in open(tilt_com_path):
        if re.findall(r'OFFSET*', line) == ['OFFSET']:
            match = re.findall(r'[0-9]+.[0-9]+', line)
            alphaOffset = float(match[0])
    output_aln_path = ''
    tomo_path = f'/nrs/liza/cathy_tomos/top_exclude_init/{folder}_Vol.mrc'
    date = datetime.today().strftime('%m_%d_%Y')
    autoxcorr_shift, tilt_ang = get_shifts(tomo_path, tilt_path, aln_path, output_aln_path, (tuple_centers[i][0],tuple_centers[i][1]), 120, 4, date, 0.3, alphaOffset)

    for t, a in zip(tilt_ang, autoxcorr_shift):
        df.loc[len(df)] = {
            'tomo_name': folder,
            'shift': a,
            'tilt_ang': t
        }

print('done')