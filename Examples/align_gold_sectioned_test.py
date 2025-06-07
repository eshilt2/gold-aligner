import mrcfile
import numpy as np
import gold_aligner as ga
from gold_aligner._align_to_tilt_series import *
from gold_aligner._select_aunps import *
from gold_aligner._convolution_and_cross_correlation import *
from gold_aligner._make_IMOD_model import *
from gold_aligner._fit_gaussian import *
from gold_aligner._find_clusters import cluster_points

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
        subprocess.run(f'ln -sfn /nrs/liza/cathy_tomos/test_reconstruction/test_rm_fiducials/{folder}/20241030_AMmilled12-2_53_{folder}.aln /nrs/liza/cathy_tomos/test_reconstruction/test_patch/aretomo_files/20241030_AMmilled12-2_53.aln', shell = True)
        cmd = (
        "ml cuda/cuda-11.3.1 &&"
        f"/nrs/liza/AreTomo3/AreTomo3 -InPrefix /nrs/liza/cathy_tomos/test_reconstruction/test_patch/aretomo_files/20241030_AMmilled12 " 
        f"-InSuffix .mrc -OutDir /nrs/liza/cathy_tomos/test_reconstruction/test_rm_fiducials/{folder} -Cmd 2 -Serial 1 -Wbp 1 -FlipVol 1 "
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
                 border_cutoff = (5,50,50),     # (int, int, int)    : will crop selection area during peak picking (z, y, x)
                 rel_threshold = 0.4,           # int               : sets threshold for selection local peaks. max(peak)*threshold.   
                 bin = 4.85,                    # flt               : binning used in original aretomo3 tomogram reconstruction, default 4.85
                 tomo_au_model = False,         # T/F               : will return .mod of selected gold in tomogram
                 tilt_au_model = False,         # T/F               : will return .mod of selected gold in tilt series
                 conv_radius = 7,               # int               : sets size of gold particle used for convolution
                 tilt_conv_au_model = False,    # T/F               : will return .mod of convolved gold in tilt series
                 alpha_offset = None,           # int               : correct Alpha Offset if known    
                 n_sections = 4,                # int               : n x n section generation of tomogram
                 img_output_path = '/groups/liza/Pictures/06_07_2025', 
                 core_folder_path = '/nrs/liza/cathy_tomos/test_reconstruction/test_xcorr/20240111_WaffleHipp_183', 
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
    
    _, peak_coords = select_aunps(invert_tomo, rel_threshold, border_cutoff, None, None)
    sectioned_area = make_sectioned_area(n=n_sections, tomo_shape=tomo_shape)

    all_peaks = select_aunps_sectioned(invert_tomo, rel_threshold, border_cutoff, sectioned_area)
    peak_coords_OI, _, list_of_sigmas = find_3d_gaussian_peaks(invert_tomo, all_peaks)
    filtered_coords, filtered_sigmas = zip(*[
        (coord, sigma) for coord, sigma in zip(peak_coords_OI, list_of_sigmas)
        if sigma[0] >= 0.5 and sigma[1] >= 0.5 and sigma[2] >= 1 and sigma[2] < 4
        ])
    filtered_coords = np.array(filtered_coords)
    filtered_sigmas = np.array(filtered_sigmas)

    pred = cluster_points(filtered_coords, f'{img_output_path}/{tomo_name}_{rel_threshold}_cluster.png', True)
    selected_points = filtered_coords[pred != -1]
    selected_sigmas = filtered_sigmas[pred != -1]
    if tomo_au_model == True: # returns .mod model of all points selected in tomogram
        plot_3d_sigmas(selected_sigmas, tomo_name, f'{img_output_path}/{tomo_name}:_{len(selected_sigmas)}_{rel_threshold}_yes_fiducials_filtered.png')
        make_imod_model(np.array(selected_points), tomo_num, custom_name = f'{core_folder_path}/{tomo_name}_{rel_threshold}_post_cluster_yes_fiducial')

    x, y, z = np.array(selected_points).T
    region_labels = sectioned_area[z.astype(int), y.astype(int), x.astype(int)]
    print('picked_peaks')

    for patch in range(1,17):# make sure to change w n
        peak_coords = np.array(selected_points)[region_labels == patch]
        if len(peak_coords) < 9:
            continue

        if os.path.exists(f"{core_folder_path}/patch_{patch}"== False):
            subprocess.run(f"mkdir {core_folder_path}/patch_{patch}", shell = True)        
        print('3d gaussian fit')


        if tomo_au_model == True: # returns .mod model of all points selected in tomogram
                make_imod_model(peak_coords, tomo_num, custom_name = f'{core_folder_path}/patch_{patch}/{tomo_name}_{patch}')
                plot_3d_sigmas(list_of_sigmas, tomo_name, f'{img_output_path}/{tomo_name}:_{len(list_of_sigmas)}_{patch}_post_cluster_yes_fiducial')
                plt.close()
        print('made model')
        print('')

        ######################################################################
        ### Align gold particles to tilt series ##############################

        final_coords = align_to_tilt_series(tomo_shape, tilt_shape, peak_coords, bin, aretomo3_alignment)


        # OPTIONAL
        if tilt_au_model == True:
            if aretomo3_alignment.DarkFrames != []:
                for indx in dark_slices:
                    for i, coord in enumerate(final_coords):
                        if coord[2] >= indx:
                            coord[2] = coord[2] + 1
                        final_coords[i] = coord

            tilt_coords = np.array(final_coords)
                
            make_imod_model(tilt_coords, tomo_num, f'{core_folder_path}/{tomo_name}')


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

        cropped_phase, shift, saved_cropped_phase = cross_corr(tilt * -1, conv_image)
        fig1 = plot_xcorr_peaks(saved_cropped_phase, shift, f'NOT NORMALIZED {tomo_name}_{patch}')
        plt.savefig(f"{img_output_path}/{tomo_name}:_{len(list_of_sigmas)}_{patch}_xcorr_not_norm.png", dpi=300)
        plt.close()
        print('saved_xcorr_plot')

        ideal_sigma, auto_shift, auto_cropped_phase = cross_corr(conv_image, conv_image)
        fig2 = plot_xcorr_peaks(auto_cropped_phase, auto_shift, f'Auto xcorr conv_img {tomo_name}_{patch}')
        plt.savefig(f"{img_output_path}/{tomo_name}:_{len(list_of_sigmas)}_{patch}_auto_xcorr_mask.png", dpi=300)

        # cropped_phase, shift, saved_cropped_phase = cross_corr(tilt * -1, conv_image)
        # fig3 = plot_xcorr_peaks(saved_cropped_phase, shift, f'Final Shift {tomo_name}_{patch}')
        # plt.savefig(f"{img_output_path}/{tomo_name}:_{len(list_of_sigmas)}_{patch}_xcorr_final_shift.png", dpi=300)
        _, autoxcorr_shift, autoxcorr_phase = cross_corr(tilt * -1, conv_image, ideal_sigma)
        fig3 = plot_xcorr_peaks(autoxcorr_phase, autoxcorr_shift, f'autoxcorrxcorr_with_fixed_sigma {tomo_name}_{patch}')
        plt.savefig(f"{img_output_path}/{tomo_name}:_{len(list_of_sigmas)}_{patch}_autoxcorrxcorr_with_fixed_sigma.png", dpi=300)
        plt.close()
        ######################################################################
        ### Create new .aln file

        if alpha_offset != aretomo3_alignment.AlphaOffset and alpha_offset != None: # checks if alpha offset is incorrect and adjusts it
            aretomo3_alignment = fix_alpha_offset(aretomo3_alignment, alpha_offset)
        
        for i, algnmt in enumerate(aretomo3_alignment.GlobalAlignments):
            algnmt.tx = algnmt.tx + autoxcorr_shift[i][0] 
            algnmt.ty = algnmt.ty + autoxcorr_shift[i][1]

        aretomo3_alignment.LocalAlignments = [] # if this was run w local patch correction, this removes that 
        aretomo3_alignment.NumPatches = 0 # also to revert from initial local patch correction to global only
        patch_aln_path = f"{output_aln_path}{tomo_name}/patch_{patch}/{tomo_name}_patch_{patch}_autoxcorr.aln"
        write(aretomo3_alignment, patch_aln_path)

        subprocess.run(f'ln -sfn {patch_aln_path} /nrs/liza/cathy_tomos/15f1_top_topop/{tomo_name}/{tomo_name}.aln', shell = True)
        cmd_path = "/nrs/liza/cathy_tomos/cmd_patch_align"
        cmd = (
        "ml cuda/cuda-11.3.1 &&"
        f"/nrs/liza/AreTomo3/AreTomo3 -InPrefix /nrs/liza/cathy_tomos/15f1_top_topop/{tomo_name}/{tomo_name} " 
        f"-InSuffix .mrc -OutDir {core_folder_path}/patch_{patch} -Cmd 2 -Serial 1 -Wbp 1 -FlipVol 1 "
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
        print('end')
    return saved_cropped_phase, shift

if __name__ == '__main__':
    # main_path = "{core_folder_path}/"
    # cmd_path = "/nrs/liza/cathy_tomos/cmd_patch_align"
    # align_by_patch(main_path, cmd_path)

    tomo = f"/nrs/liza/cathy_tomos/tomos_init_rerun/20240111_WaffleHipp_183_Vol.mrc"
    tilt = f"/nrs/liza/cathy_tomos/15f1_top_topop/20240111_WaffleHipp_183/20240111_WaffleHipp_183.mrc"
    aln = f"/nrs/liza/cathy_tomos/tomos_init_rerun/20240111_WaffleHipp_183_rerun.aln"
    aln_output = "/nrs/liza/cathy_tomos/test_reconstruction/test_xcorr/"
    bin = 4
    border = (5,50,50)
    i = 0.3
    realign_gold(tomo, tilt, aln, aln_output, border_cutoff= border, rel_threshold=i, bin = bin, tomo_au_model = True, tilt_au_model=False, alpha_offset = None, debugger=False)

    path_to_all_folders = '/nrs/liza/cathy_tomos/15f1_top_topop/'


    folder_list = os.listdir(path_to_all_folders)
    for i, folder in enumerate(folder_list):
        if folder == '.stfolder' or i>=7: # allows for picking up after an error (use debugger to see which folder it got stuck on)
            continue
        if os.path.exists(f"/nrs/liza/cathy_tomos/test_reconstruction/test_xcorr/{folder}"== False):
            subprocess.run(f'mkdir /nrs/liza/cathy_tomos/test_reconstruction/test_xcorr/{folder}', shell = True)
        if os.path.exists(f"/groups/liza/Pictures/06_06_2025/{folder}"== False):
            subprocess.run(f'mkdir /groups/liza/Pictures/06_06_2025/{folder}', shell = True)
        tomo = f"/nrs/liza/cathy_tomos/tomos_init_rerun/{folder}_Vol.mrc"
        tilt = f"/nrs/liza/cathy_tomos/15f1_top_topop/{folder}/{folder}.mrc"
        aln = f"/nrs/liza/cathy_tomos/tomos_init_rerun/{folder}_rerun.aln"
        aln_output = f"/nrs/liza/cathy_tomos/test_reconstruction/test_xcorr/{folder}"
        img_output_path = f'/groups/liza/Pictures/06_06_2025/{folder}'
        core_folder_path = f'/nrs/liza/cathy_tomos/test_reconstruction/test_xcorr/{folder}'
        realign_gold(tomo, tilt, aln, aln_output, (5,50,50), 0.3, tomo_au_model= True, img_output_path=img_output_path, core_folder_path=core_folder_path)


    tomo = f"/nrs/liza/cathy_tomos/tomos_init_rerun/20231017_EGmilled24-2_68_Vol.mrc"
    tilt = f"/nrs/liza/cathy_tomos/15f1_top_topop/20231017_EGmilled24-2_68/20231017_EGmilled24-2_68.mrc"
    aln = f"/nrs/liza/cathy_tomos/tomos_init_rerun/20231017_EGmilled24-2_68_rerun.aln"
    aln_output = "/nrs/liza/cathy_tomos/test_reconstruction/test_tilt/"

    # master_path = "/nrs/liza/cathy_tomos/15f1_top_topop/"
    # folder_list = os.listdir(master_path)
    # for i, folder in enumerate(folder_list):
    #     if folder == "20231026_HippAu_26":
    #         tomo = f"/nrs/liza/cathy_tomos/tomos_init_rerun/{folder}_Vol.mrc"
    #         tilt = f"/nrs/liza/cathy_tomos/15f1_top_topop/{folder}/{folder}.mrc"
    #         aln = f"/nrs/liza/cathy_tomos/15f1_top_topop/{folder}/{folder}_original.aln"
    #         aln_output = "/nrs/liza/cathy_tomos/test_reconstruction/20231017_HippAu_150_test.aln.aln"

    bin = 4
    # center = (335, 190)
    # radius = 120
    # center = (460,800) # tomo 64
    # center = (100, 380) # tomo 127
    # radius = 120
    border = (5,50,50)
    # center = None
    # radius = None
    i = 0.3
    # threshold = [0.45, 0.5, 0.55, 0.6]
    # for i in threshold:
    realign_gold(tomo, tilt, aln, aln_output, border_cutoff= border, rel_threshold=i, bin = bin, tomo_au_model = True, tilt_au_model= False, alpha_offset = None, debugger=True)

