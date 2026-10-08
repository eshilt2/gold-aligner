# rerunning 5f11 with deep et picker selections

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

def find_3d_gauss_peaks(invert_tomo,     # array : initially aligned data from tomogram.mrc * -1 
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
        try:
            fit, _ = curve_fit(get_3d_gaussian, coords, selected_area.ravel(), p0=guess, bounds = bounds, method = 'dogbox')
        except RuntimeError:
            try:
                fit, _ = curve_fit(get_3d_gaussian, coords, selected_area.ravel(), p0=guess, bounds = bounds, method = 'trf')
            except RuntimeError:
                continue
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

def realign_with_mono_selected(picks,               # array : preselected picks used as initial search points for aunps
                            tomo_path,              # str   : path and name of tomogram
                            tilt_path,              # str   : path and name of tilt series
                            aln_path,               # str   : path and name of aligned tomogram
                            output_aln_path,        # str   : path and name of outputed aln file
                            center_OI,              # (int, int, int)    : center of cylinder that will determine selected aunps in radius around center
                            radius_OI,              # int   : radius around center_OI where aunps will be selected
                            bin,                    # int   : binning of image
                            alpha_offset,           # flt   : ensures proper alpha offset of imod file
                            date                    # str   : used by me to save cross corr images to folder called whatever the date in Pictures
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

    peak_coords_OI, _, list_of_sigmas = find_3d_gauss_peaks(invert_tomo, peaks, 100)
    if len(peak_coords_OI) < 8:
        new_picks = [n for n in picks if 0 not in n]
        peaks = new_picks
        peak_coords_OI, _, list_of_sigmas = find_3d_gauss_peaks(invert_tomo, peaks, 100)

    # make_IMOD_model_UPDATED(peak_coords_OI, f'/nrs/liza/cathy_tomos/mono_dimer_azs_oneItr/{folder}')
    plot_3d_sigmas(list_of_sigmas, tomo_name, f'/groups/liza/Pictures/{date}/{tomo_name}:_{len(list_of_sigmas)}')
    plt.close()
    print('made model')
    print('')

    final_coords = align_to_tilt_series(tomo_shape, tilt_shape, peak_coords_OI, bin, aretomo3_alignment)

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
    write(imod_aln, output_aln)
    write(aretomo3_alignment, output_aln + f"{tomo_name}.aln")

    

def init_tomos(command_path, folder, aln_path, tilt_path, alpha_offset):
    subprocess.run(f'mkdir /nrs/liza/cathy_tomos/DetP_top_exclude_5f11/init_tomos/{folder}', shell = True)
    os.chdir(f"/nrs/liza/cathy_tomos/DetP_top_exclude_5f11/init_tomos/{folder}")
    subprocess.run(f"ln -sfn /nrs/elferich/5f11_top_exclude/{folder}/{folder}.mrc /nrs/elferich/5f11_top_exclude/{folder}/{folder}.rawtlt .", shell = True)
    
    read_alignment = read(aln_path)
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
   
    write(aretomo3_alignment, f'/nrs/liza/cathy_tomos/DetP_top_exclude_5f11/init_tomos/{folder}/{folder}.aln')

    cmd = (
    "ml cuda/cuda-11.3.1 &&"
    f"/nrs/liza/AreTomo3/AreTomo3 -InPrefix /nrs/liza/cathy_tomos/DetP_top_exclude_5f11/init_tomos/{folder}/ "
    f"-InSuffix .mrc -OutDir /nrs/liza/cathy_tomos/DetP_top_exclude_5f11/init_tomos/{folder} -Cmd 2 -Serial 1 -Wbp 1 -FlipVol 1 "
    "-VolZ 1500 -AtBin 4 -Gpu 0 -Cs 0.01"
    )
    with open(command_path, "w") as file:
        file.write(cmd) 

    subprocess.run([
        "gnome-terminal",
        "--wait",
        "--",
        "bash", "-i", "-c", f"{command_path}"
    ])
    
    print(folder)
    

if __name__ == "__main__":
# grab tomo name --> read in mono coords --> set tomo to 3DCTF corrected tomo --> aln to init aln
    centers = np.genfromtxt('/nrs/liza/cathy_tomos/DetP_top_exclude_5f11/5f11_tomo_centers_edited.csv', delimiter=',', skip_header = 1)    

    folder_list = os.listdir("/nrs/elferich/5f11_top_exclude/")
    command_path = '/nrs/liza/cathy_tomos/DetP_top_exclude_5f11/cmd_tomo_init'


    for i, folder in enumerate(folder_list):
        os.chdir(f"/nrs/elferich/5f11_top_exclude/{folder}/")
        if folder == '5f11_top_exclude':
            continue
        if folder == "20240830_AMmilled16_36" or folder == "20241203_AMmilled23-1_Position_14" or folder == "20241202_AMmilled23-2_Position_27":
            continue
        
        tilt_path = f"/nrs/elferich/5f11_top_exclude/{folder}/{folder}.mrc"
        aln_path = f"/nrs/elferich/5f11_top_exclude/{folder}/fiducial_tracking/{folder}"
        if os.path.exists(f"{aln_path}.xf"):
            print(f"Folder '{aln_path}' exists.")
            if os.path.exists(f"/nrs/elferich/5f11_top_exclude/{folder}/fiducial_tracking/active_zonograms"):
                folder_path = f"/nrs/elferich/5f11_top_exclude/{folder}/fiducial_tracking"
            else:             
                aln_path = f"/nrs/elferich/5f11_top_exclude/{folder}/patch_tracking/{folder}"
                folder_path = f"/nrs/elferich/5f11_top_exclude/{folder}/patch_tracking"
        else:
            aln_path = f"/nrs/elferich/5f11_top_exclude/{folder}/patch_tracking/{folder}"
            folder_path = f"/nrs/elferich/5f11_top_exclude/{folder}/patch_tracking"
        # tilt_com_path = f"{folder_path}/tilt.com"           
        # for line in open(tilt_com_path):
        #     if re.findall(r'OFFSET*', line) == ['OFFSET']:
        #         match = re.findall(r'[0-9]+.[0-9]+', line)
        #         alphaOffset = float(match[0])
        # center = centers[i]
        # init_tomos(command_path, folder, aln_path, tilt_path, alphaOffset)

        subprocess.run(f'mkdir /nrs/liza/cathy_tomos/ddw/imod_alignments/DetP_top_exclude_5f11/{folder}', shell = True)
        with open(f"/nrs/liza/cathy_tomos/DetP_top_exclude_5f11/5F11_coords_BIN2/{folder}_full_rec_BP_3DCTF_BIN2.coords") as f:
            mylist = f.read().splitlines()
            listing = [i.split() for i in mylist]
            int_list = [[int(y) for y in x] for x in listing]
            mono_list = [n for n in int_list if n[0] == 1]
            array = np.array(mono_list)
            m_picks = np.array(array[:,1:4]/2).astype(int)
            m_picks[:,2] = m_picks[:,2]*(1600/1800)
            make_IMOD_model_UPDATED(m_picks, f'/nrs/liza/cathy_tomos/DetP_top_exclude_5f11/{folder}_hoyoung_picks_direct_test')
            # tomo = f"/nrs/liza/cathy_tomos/ddw/imod_alignments/mono_dimer/{folder}/{folder2}_full_rec_BP_3DCTF_BIN4.mrc"
        tomo = f"/nrs/liza/cathy_tomos/tomos_5f11_init/{folder}_Vol.mrc"
        tilt = f"/nrs/elferich/5f11_top_exclude/{folder}/{folder}.mrc"
        aln = aln_path
        output_aln = f"/nrs/liza/cathy_tomos/ddw/imod_alignments/DetP_top_exclude_5f11/{folder}/"
        tilt_com_path = f"{folder_path}/tilt.com"           
        for line in open(tilt_com_path):
            if re.findall(r'OFFSET*', line) == ['OFFSET']:
                match = re.findall(r'[0-9]+.[0-9]+', line)
                alphaOffset = float(match[0])
        center = centers[i]
        if math.isnan(center[0]) and math.isnan(center[1]):
            continue
        realign_with_mono_selected(m_picks, tomo, tilt, aln, output_aln, (center[1], center[2]), 120, 4,  0, "11_25_2025")
        # realign_gold(tomo, tilt, aln, output_aln, bin = 4, rel_threshold= 0.6, center_OI = (245, 500), radius_OI=120, tomo_au_model= True)

        print('done')
