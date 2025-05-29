import mrcfile
import numpy as np
import gold_aligner as ga
from gold_aligner._align_to_tilt_series import *
from gold_aligner._select_aunps import *
from gold_aligner._convolution_and_cross_correlation import *
from gold_aligner._make_IMOD_model import *
from gold_aligner._fit_gaussian import *

import os
from scipy import ndimage as ndi
from cryoet_alignment import read, write 
from cryoet_alignment.io.cryoet_data_portal import Alignment

import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from gold_aligner.fix_alpha_offset import fix_alpha_offset

from torch_grid_utils.fftfreq_grid import dft_center
from torch_grid_utils.coordinate_grid import coordinate_grid
from torch.nn.functional import conv2d
from scipy.spatial import KDTree


def realign_gold(tomo_path,                     # str               : path to initially aligned tomogram.mrc file
                 tilt_path,                     # str               : path to initially aligned tiltseries.mrc file
                 aln_path,                      # str               : path to aretomo3 generated .aln file used for initially aligned tomogram
                 output_aln_path,               # str               : output aln file name
                 border_cutoff = (70,50,50),    # (int, int, int)   : will crop selection area during peak picking (z, y, x)
                 rel_threshold = 0.4,           # int               : sets threshold for selection local peaks. max(peak)*threshold.   
                 bin = 4.85,                    # flt               : binning used in original aretomo3 tomogram reconstruction, default 4.85
                 center_OI = None,              # (int,int)         : will select gold only within sphere centered at (x,y)
                 radius_OI = None,              # int               : will select gold only within sphere centered at (x,y) with radius r
                 tomo_au_model = False,         # T/F               : will return .mod of selected gold in tomogram
                 tilt_au_model = False,         # T/F               : will return .mod of selected gold in tilt series
                 conv_radius = 7,               # int               : sets size of gold particle used for convolution
                 tilt_conv_au_model = False,    # T/F               : will return .mod of convolved gold in tilt series
                 alpha_offset = None,           # int               : correct Alpha Offset if known
                 dimer = False,                 # bool              : if False uses single gaussian 3D fit; if True uses gaussian mixture model
                 debugger = False
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
    tomo_name = re.findall(r".*/(.*).mrc",tilt_path)
    ######################################################################
    ###  Select gold particles in tomo  ##################################
    _, peak_coords = select_aunps(invert_tomo, rel_threshold, border_cutoff, center_OI, radius_OI)

    print('picked_peaks')

    if dimer == False: # fit gaussian model to find subpixel center
        peak_coords_OI, _, list_of_sigmas = find_3d_gaussian_peaks(invert_tomo, peak_coords)
    else:
        peak_coords_OI = get_mixed_gaussian(invert_tomo, peak_coords)
    print('3d gaussian fit')


    if tomo_au_model == True: # returns .mod model of all points selected in tomogram
            make_imod_model(peak_coords_OI, tomo_num, custom_name = '/nrs/liza/cathy_tomos/test_reconstruction/test_thresholds/138_fid_test')
            plot_3d_sigmas(list_of_sigmas, tomo_name, '/groups/liza/Pictures/05_29_2025/138_fid_test')
    print('made model')
    ######################################################################
    ### Align gold particles to tilt series ##############################

    final_coords = align_to_tilt_series(tomo_shape, tilt_shape, peak_coords_OI, bin, aretomo3_alignment)


    # OPTIONAL
    if tilt_au_model == True:
        if aretomo3_alignment.DarkFrames != []:
            for indx in dark_slices:
                for i, coord in enumerate(final_coords):
                    if coord[2] >= indx:
                        coord[2] = coord[2] + 1
                    final_coords[i] = coord

        tilt_coords = np.array(final_coords)
            
        make_imod_model(tilt_coords, tomo_num, "tilt")


    ######################################################################
    ### Create model layer of AUNPs based on their position ##############
    # creating base image where each gold particle peak is just a point

    base_img, circle_img = make_model_layer_components(tilt, aretomo3_alignment, conv_radius, final_coords, shape = 'circle')


    # Convolve cirlce with pixel placement
    conv_image, rev_conv_coords = convolve_image(circle_img, base_img, tilt_conv_au_model)

    # Optional
    if tilt_conv_au_model == True:
        conv_coords = np.array(rev_conv_coords)
        make_imod_model(conv_coords, tomo_num, "tilt_conv")

    ######################################################################
    ###  Cross correlation to get shift
    cropped_phase, shift, saved_cropped_phase = cross_corr(tilt, conv_image)

    ######################################################################
    ### Create new .aln file

    if alpha_offset != aretomo3_alignment.AlphaOffset and alpha_offset != None: # checks if alpha offset is incorrect and adjusts it
        aretomo3_alignment = fix_alpha_offset(aretomo3_alignment, alpha_offset)
    
    for i, algnmt in enumerate(aretomo3_alignment.GlobalAlignments):
        algnmt.tx = algnmt.tx + shift[i][0] 
        algnmt.ty = algnmt.ty + shift[i][1]

    aretomo3_alignment.LocalAlignments = [] # if this was run w local patch correction, this removes that 
    aretomo3_alignment.NumPatches = 0 # also to revert from initial local patch correction to global only

    write(aretomo3_alignment, output_aln_path)

    return saved_cropped_phase, shift

if __name__ == '__main__':
    
    tomo = "/nrs/liza/cathy_tomos/tomos_init_rerun/20240111_WaffleHipp_138_Vol.mrc"
    tilt = "/nrs/liza/cathy_tomos/15f1_top_topop/20240111_WaffleHipp_138/20240111_WaffleHipp_138.mrc"
    aln = "/nrs/liza/cathy_tomos/15f1_top_topop/20240111_WaffleHipp_138/20240111_WaffleHipp_138_original.aln"
    aln_output = "/nrs/liza/cathy_tomos/test_reconstruction/20240111_WaffleHipp_138_test.aln"

    bin = 4
    # center = (335, 190)
    # radius = 120
    # center = (460,800) # tomo 64
    # center = (100, 380) # tomo 127
    # radius = 120
    border = (5,50,50)
    # center = None
    # radius = None
    threshold = 0.3
    realign_gold(tomo, tilt, aln, aln_output, border_cutoff= border, rel_threshold=threshold, bin = bin, tomo_au_model = True, tilt_au_model=False, alpha_offset = None, debugger=False)

    # tomo = "/nrs/liza/hoyoung_dimer_tomos/dimer_tomos/dimer_tomos_init/20250210_HippWaffle_64_Vol.mrc"
    # tilt = "/nrs/liza/hoyoung_dimer_tomos/dimer_tomos/dual/20250210_HippWaffle_64.mrc"
    # aln = "/nrs/liza/hoyoung_dimer_tomos/dimer_tomos/dual/20250210_HippWaffle_64.aln"
    # aln_output = "/nrs/liza/hoyoung_dimer_tomos/dimer_bin2_oneItr/20250210_HippWaffle_64_bin2_oneItr.aln"
    
    # tomo = "/nrs/liza/hoyoung_dimer_tomos/dimer_tomos/only_dimers_init/tiltseries3.mrc_Vol.mrc"
    # tilt = "/nrs/liza/hoyoung_dimer_tomos/dimer_tomos/only_dimers/tiltseries3.mrc"
    # aln = "/nrs/liza/hoyoung_dimer_tomos/dimer_tomos/only_dimers_init/tiltseries3.mrc_original.aln"
    # aln_output = "/nrs/liza/hoyoung_dimer_tomos/dimer_tomos/only_dimers_mixG_oneItr/tiltseries3.mrc_mixG_oneItr.aln"
    
    # bin = 2.1775
    # alpha_tilt = 20
    # border = (140, 200, 200)
    # center = (1013,1510)

    # radius = 240