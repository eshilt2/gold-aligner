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
def align_by_section(main_path, cmd_path):
    folder_list = [d for d in os.listdir(main_path) if os.path.isdir(os.path.join(main_path, d))]
    for i, folder in enumerate(folder_list):
    # go file by file, reconstructing tomos
        subprocess.run(f'ln -sfn /nrs/liza/cathy_tomos/test_reconstruction/test_patch/AMmilled12-2_53_sectioned_corner_rerun/{folder}/20241030_AMmilled12-2_53_{folder}.aln /nrs/liza/cathy_tomos/test_reconstruction/test_patch/aretomo_files/20241030_AMmilled12-2_53.aln', shell = True)
        cmd = (
        "ml cuda/cuda-11.3.1 &&"
        f"/nrs/liza/AreTomo3/AreTomo3 -InPrefix /nrs/liza/cathy_tomos/test_reconstruction/test_patch/aretomo_files/20241030_AMmilled12 " 
        f"-InSuffix .mrc -OutDir /nrs/liza/cathy_tomos/test_reconstruction/test_patch/AMmilled12-2_53_sectioned_corner_rerun/{folder} -Cmd 2 -Serial 1 -Wbp 1 -FlipVol 1 "
        "-VolZ 1600 -AtBin 4 -Gpu 0 -Cs 0.01"
    )
        with open(cmd_path, "w") as file:
            file.write(cmd) 

        subprocess.run([
            "gnome-terminal",
            "--wait",
            "--",
            "bash", "-i", "-c", f"{cmd_path}"
        ])

        print(folder)

def make_sectioned_area(n, tomo_shape):
    arr = np.zeros(tomo_shape, dtype=int)

    y_splits = np.array_split(np.arange(tomo_shape[1]), n)
    x_splits = np.array_split(np.arange(tomo_shape[2]), n)

    label = 1
    for i, y_idx in enumerate(y_splits):
        for j, x_idx in enumerate(x_splits):
            arr[:, y_idx[:, None], x_idx] = label
            label += 1

    return arr


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
    tomo_name = re.findall(r".*/(.*).mrc",tilt_path)[0]
    ######################################################################
    ###  Select gold particles in tomo  ##################################
    if debugger == False:
        _, peak_coords = select_aunps(invert_tomo, rel_threshold, border_cutoff, center_OI, radius_OI)
    else:
        sectioned_area = make_sectioned_area(n=4, tomo_shape=tomo_shape)
        all_peaks = select_aunps_sectioned(invert_tomo, rel_threshold, border_cutoff, sectioned_area)
        x, y, z = np.array(all_peaks).T
        region_labels = sectioned_area[z, y, x]
        print('picked_peaks')
        for section in range(1,17):# make sure to change w n
            peak_coords = np.array(all_peaks)[region_labels == section]
            if len(peak_coords) < 9:
                continue

            if dimer == False: # fit gaussian model to find subpixel center
                peak_coords_OI, _, list_of_sigmas = find_3d_gaussian_peaks(invert_tomo, peak_coords)
            else:
                peak_coords_OI = get_mixed_gaussian(invert_tomo, peak_coords)
            if len(peak_coords_OI) < 9:
                continue
            if os.path.exists(f"/nrs/liza/cathy_tomos/test_reconstruction/test_patch/AMmilled12-2_53_sectioned_corner_rerun/section_{section}"== False):
                subprocess.run(f"mkdir /nrs/liza/cathy_tomos/test_reconstruction/test_patch/AMmilled12-2_53_sectioned_corner_rerun/section_{section}", shell = True)        
            print('3d gaussian fit')


            if tomo_au_model == True: # returns .mod model of all points selected in tomogram
                    make_imod_model(peak_coords_OI, tomo_num, custom_name = f'/nrs/liza/cathy_tomos/test_reconstruction/test_patch/AMmilled12-2_53_sectioned_corner_rerun/section_{section}/{tomo_name}_{section}')
                    plot_3d_sigmas(list_of_sigmas, tomo_name, f'/groups/liza/Pictures/06_02_2025/{tomo_name}:_{len(list_of_sigmas)}_{section}_cornerFixed')
                    plt.close()
            print('made model')
            print('')
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
                    
                make_imod_model(tilt_coords, tomo_num, f'/nrs/liza/cathy_tomos/test_reconstruction/test_patch/{tomo_name}')


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
            
            fig = plot_xcorr_peaks(saved_cropped_phase, shift, f'20241030_AMmilled12-2_53_section_{section}')
            plt.savefig(f"/groups/liza/Pictures/06_02_2025/{tomo_name}:_{len(list_of_sigmas)}_{section}_corner_fixed_cross_corr.png", dpi=300)
            plt.close(fig)
            print('saved_xcorr_plot')
            ######################################################################
            ### Create new .aln file

            if alpha_offset != aretomo3_alignment.AlphaOffset and alpha_offset != None: # checks if alpha offset is incorrect and adjusts it
                aretomo3_alignment = fix_alpha_offset(aretomo3_alignment, alpha_offset)
            
            for i, algnmt in enumerate(aretomo3_alignment.GlobalAlignments):
                algnmt.tx = algnmt.tx + shift[i][0] 
                algnmt.ty = algnmt.ty + shift[i][1]

            aretomo3_alignment.LocalAlignments = [] # if this was run w local patch correction, this removes that 
            aretomo3_alignment.NumPatches = 0 # also to revert from initial local patch correction to global only
            section_aln_path = f"{output_aln_path}/section_{section}/20241030_AMmilled12-2_53_section_{section}.aln"
            write(aretomo3_alignment, section_aln_path)

        return saved_cropped_phase, shift

if __name__ == '__main__':
    main_path = "/nrs/liza/cathy_tomos/test_reconstruction/test_patch/AMmilled12-2_53_sectioned_corner_rerun/"
    cmd_path = "/nrs/liza/cathy_tomos/cmd_patch_align"
    align_by_section(main_path, cmd_path)

    tomo = f"/nrs/liza/cathy_tomos/test_reconstruction/test_patch/AMmilled12-2_53_sectioned_corner_rerun/20241030_AMmilled12-2_53_cornerFixed_rerun_Vol.mrc"
    tilt = f"/nrs/liza/cathy_tomos/15f1_top_topop/20241030_AMmilled12-2_53/20241030_AMmilled12-2_53.mrc"
    aln = f"/nrs/liza/cathy_tomos/tomos_init_rerun/20241030_AMmilled12-2_53_rerun.aln"
    aln_output = "/nrs/liza/cathy_tomos/test_reconstruction/test_patch/AMmilled12-2_53_sectioned_corner_rerun"

    # master_path = "/nrs/liza/cathy_tomos/15f1_top_topop/"
    # folder_list = os.listdir(master_path)
    # for i, folder in enumerate(folder_list):
    #     if folder == "20231026_HippAu_26":
    #         tomo = f"/nrs/liza/cathy_tomos/tomos_init_rerun/{folder}_Vol.mrc"
    #         tilt = f"/nrs/liza/cathy_tomos/15f1_top_topop/{folder}/{folder}.mrc"
    #         aln = f"/nrs/liza/cathy_tomos/15f1_top_topop/{folder}/{folder}_original.aln"
    #         aln_output = "/nrs/liza/cathy_tomos/test_reconstruction/20240111_WaffleHipp_138_test.aln.aln"

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
    realign_gold(tomo, tilt, aln, aln_output, border_cutoff= border, rel_threshold=threshold, bin = bin, tomo_au_model = True, tilt_au_model=False, alpha_offset = None, debugger=True)

