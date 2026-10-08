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
    make_IMOD_model_UPDATED(peak_coords_OI, f'/nrs/liza/cathy_tomos/mono_dimer_azs_oneItr/{folder}_azs_mpicks_{number}')
    plot_3d_sigmas(list_of_sigmas, tomo_name, f'/groups/liza/Pictures/{date}/{tomo_name}_az{number}:_{len(list_of_sigmas)}')
    plt.close()
    print('made model')
    print('')

    final_coords = align_to_tilt_series(tomo_shape, tilt_shape, peak_coords_OI, bin, aretomo3_alignment)

    base_img, circle_img = make_model_layer_components(tilt, aretomo3_alignment, 7, final_coords, shape = 'circle')
    
    # Convolve cirlce with pixel placement
    conv_image, rev_conv_coords = convolve_image(circle_img, base_img, False)
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

    write(aretomo3_alignment, output_aln_path)

def create_aretomo_alns(path_to_all_folders, core_path, cmd_path):
    folder_list = os.listdir(path_to_all_folders)
    skip_list = [
    "20250418_AMmilled29-2_Position_58_7",
    "20250418_AMmilled29-2_Position_88",
    "20250418_AMmilled29-2_Position_47",
    "20250418_AMmilled29-2_Position_86",
    "20250418_AMmilled29-2_Position_87"]
    for i, folder in enumerate(folder_list):
        if folder == '.stfolder': # allows for picking up after an error (use debugger to see which folder it got stuck on)
            continue
        # if folder in skip_list:
        #     folder = 
        tilt_path = f"{path_to_all_folders}{folder}/{folder}.mrc"
        aln_path = f"{path_to_all_folders}{folder}/best_alignment/{folder}"
        if os.path.exists(f"{aln_path}.xf"):
            print(f"Folder '{aln_path}' exists.")
            if os.path.exists(f"{path_to_all_folders}{folder}/best_alignment/active_zonograms"):
                folder_path = f"{path_to_all_folders}{folder}/best_alignment"
        #     elif os.path.exists(f"{path_to_all_folders}{folder}/fiducial_tracking/active_zonograms"):
        #         folder_path = f"{path_to_all_folders}{folder}/fiducial_tracking"
        #         aln_path = f"{path_to_all_folders}{folder}/fiducial_tracking/{folder}"
        #     else:             
        #         aln_path = f"{path_to_all_folders}{folder}/patch_tracking/{folder}"
        #         folder_path = f"{path_to_all_folders}{folder}/patch_tracking"
        # elif os.path.exists(f"{path_to_all_folders}{folder}/fiducial_tracking/{folder}.xf"):
        #         folder_path = f"{path_to_all_folders}{folder}/fiducial_tracking"
        #         aln_path = f"{path_to_all_folders}{folder}/fiducial_tracking/{folder}"
        # else:
        #     aln_path = f"{path_to_all_folders}{folder}/patch_tracking/{folder}"
        #     folder_path = f"{path_to_all_folders}{folder}/patch_tracking"
        if os.path.exists(f"/nrs/liza/cathy_tomos/mono_dimer/{folder}") == False:
            subprocess.run(f"mkdir /nrs/liza/cathy_tomos/mono_dimer/{folder}/", shell = True)
        output_path = f"/nrs/liza/cathy_tomos/mono_dimer_init/{folder}_init.aln"


        #convert IMOD to Aretomo3 .aln
        aln = read(aln_path)
        with mrcfile.open(tilt_path) as mrctilt:
            tilt = mrctilt.data
            tilt_shape = tilt.shape
            invert_shape = (tilt_shape[2], tilt_shape[1], tilt_shape[0])

        check = imod_to_aretomo(aln, invert_shape, f"{folder_path}/{folder}.tlt")
        
        write(check, output_path)

        # convert IMOD .tlt to Aretomo3 _TLT.txt
        subprocess.run(f"ln -sfn {folder_path}/{folder}.tlt {core_path}{folder}/{folder}_TLT.txt", shell = True)
        subprocess.run(f"ln -sfn {output_path} {core_path}{folder}/{folder}.aln", shell = True)
        subprocess.run(f"ln -sfn {path_to_all_folders}/{folder}/{folder}.mrc {core_path}{folder}/{folder}.mrc", shell = True)

        cmd = (
        "ml cuda/cuda-11.3.1 &&"
        f"/nrs/liza/AreTomo3/AreTomo3 -InPrefix {core_path}{folder}/20 "
        f"-InSuffix .mrc -OutDir /nrs/liza/cathy_tomos/mono_dimer_init/ -Cmd 2 -Serial 1 -Wbp 1 -FlipVol 1 "
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
    print('end')

def check_dimer_sigmas():

if __name__ == "__main__":
# grab tomo name --> read in mono coords --> set tomo to 3DCTF corrected tomo --> aln to init aln
    center = np.genfromtxt('/nrs/liza/cathy_tomos/mono_dimer/mono_dimers.csv', delimiter=',', skip_header = 1)    

    folder_list = os.listdir("/nrs/elferich/15F1and5F11dimer/TOP_TOMOS/")
    special_list = [
    "20250418_AMmilled29-2_Position_58_7",
    "20250418_AMmilled29-2_Position_88",
    "20250418_AMmilled29-2_Position_47",
    "20250418_AMmilled29-2_Position_86",
    "20250418_AMmilled29-2_Position_87"] #,
    #"20241206_AMmilled24-3_107"] # done
    for i, folder in enumerate(folder_list):
        folder2 = folder
        if folder in special_list: #or folder != "20241206_AMmilled24-3_107":
            if folder == "20250418_AMmilled29-2_Position_58_7":
                folder2 = folder[-13:]
            else:
                folder2 = folder[-11:]
        subprocess.run(f'mkdir /nrs/liza/cathy_tomos/ddw/imod_alignments/mono_dimer_azs/{folder}', shell = True)
        if i < 8:
            continue
        for file in glob.glob(f"/nrs/elferich/15F1and5F11dimer/TOP_TOMOS/{folder}/best_alignment/active_zonograms/active_zonogram_*.npy"):
            match = re.search(r"active_zonogram_(\d+)\.npy", file)
            if match:
                number = int(match.group(1))
            else:
                continue
            print(f"Found file: {file} with number: {number}")
            az_data = np.load(f"/nrs/elferich/15F1and5F11dimer/TOP_TOMOS/{folder}/best_alignment/active_zonograms/active_zonogram_{number}.npy" ,allow_pickle=True)
            center = tuple(az_data.tolist()["center"])

            # with mrcfile.open(f'/nrs/elferich/15F1and5F11dimer/TOP_TOMOS/{folder}/best_alignment/active_zonograms/active_zonogram_{number}.mrc') as az:
            #     radius = az.header['nx']        

            with open(f"/nrs/elferich/hoyung_coordinates/{folder2}/{folder2}_full_rec_SIRT_3DCTF_BIN2.coords") as f:
                mylist = f.read().splitlines()
                listing = [i.split() for i in mylist]
                int_list = [[int(y) for y in x] for x in listing]
                mono_list = [n for n in int_list if n[0] == 1]
                array = np.array(mono_list)
                m_picks = np.array(array[:,1:4]/2).astype(int)
                make_IMOD_model_UPDATED(m_picks, f'/nrs/liza/cathy_tomos/mono_dimer_azs_oneItr/{folder}_hoyoung_picks_direct_{number}')
            # tomo = f"/nrs/liza/cathy_tomos/ddw/imod_alignments/mono_dimer/{folder}/{folder2}_full_rec_BP_3DCTF_BIN4.mrc"
            tomo = f"/nrs/elferich/15F1and5F11dimer/TOP_TOMOS/{folder}/best_alignment/{folder2}_full_rec_BP_3DCTF_BIN4.mrc"
            tilt = f"/nrs/elferich/15F1and5F11dimer/TOP_TOMOS/{folder}/{folder2}.mrc"
            aln = f"/nrs/liza/cathy_tomos/mono_dimer_init/{folder}_init.aln"
            output_aln = f"/nrs/liza/cathy_tomos/mono_dimer_azs_oneItr/{folder2}_azs_oneItr{number}.aln"
            tilt_com_path = f"/nrs/elferich/15F1and5F11dimer/TOP_TOMOS/{folder}/best_alignment/tilt.com"
            for line in open(tilt_com_path):
                if re.findall(r'OFFSET*', line) == ['OFFSET']:
                    match = re.findall(r'[0-9]+.[0-9]+', line)
                    alphaOffset = float(match[0])

            realign_with_mono_selected(m_picks, tomo, tilt, aln, output_aln, (center[0], center[1]), 120, 4,  0, "09_10_2025", number)
            # realign_gold(tomo, tilt, aln, output_aln, bin = 4, rel_threshold= 0.6, center_OI = (245, 500), radius_OI=120, tomo_au_model= True)
            subprocess.run(f'mkdir /nrs/liza/cathy_tomos/ddw/imod_alignments/mono_dimer_azs/{folder}/aunp_tracking_{number}', shell = True)
            full_aretomo_to_imod(output_aln, tilt, f'/nrs/liza/cathy_tomos/ddw/imod_alignments/mono_dimer_azs/{folder}/aunp_tracking_{number}')

            print('done')
