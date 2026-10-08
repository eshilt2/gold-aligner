# dual aunp workflow

### NOTES ###
# 1.) I did not load in the aretomo .aln correctly, I should have used:             
#       {...}
#           invert_shape = (tilt_shape[2], tilt_shape[1], tilt_shape[0])
#       aretomo3_alignment = aretomo3_alignment.to_aretomo(ts_size=invert_shape)
#    I had not so when I wrote the imod files the shape was wrong.
# 
# 2.) MAKE SURE TO CHECK THE ALPHA OFFSET
#    Good rule of thumb is to make sure the tilts splay out about equally around 0 in the aretomo aln
#    If not, make sure the AlphaOffset is being applied correctly

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

from pathlib import Path

def get_az_centers(tomo_names, best, az_candidates=range(5)):
    base = Path("/groups/liza/temp_for_transfer/15F1-H4K2Cys_TOPTOMOS/WARP")
    results = []

    for n, tomo_name in enumerate(tomo_names):
        _type = best[n]

        for az_num in az_candidates:  # try 0–4
            npy_path = base / tomo_name / f"{_type}_tracking" / "active_zonograms" / f"active_zonogram_{az_num}.npy"

            if not npy_path.exists():
                continue  # skip missing files cleanly

            try:
                az_data = np.load(npy_path, allow_pickle=True)
                center = az_data.tolist()["center"]

                results.append({
                    "tomo_name": tomo_name,
                    "az_num": az_num,
                    "center": center,
                    "type": _type
                })

            except Exception:
                continue

    return results

def find_3d_gaussian_peaks_fix(invert_tomo,     # array : initially aligned data from tomogram.mrc * -1 
                           peak_coords,     # list: list of coordinate list [x,y,z]
                           cutoff = 95      # int: disregaurd points with sigmas above certain percentile
                           ): 
    new_point_coords = []
    list_of_sigmas = []
    list_of_fitted = []
    for en, entry in enumerate(peak_coords):

        # crops area around each point
        selected_area = invert_tomo[entry[2]-5:entry[2]+5,entry[1]-5:entry[1]+5, entry[0]-5:entry[0]+5]
        selected_area = selected_area - selected_area.min()
        area_shape = selected_area.shape

        x = np.arange(area_shape[2])
        y = np.arange(area_shape[1])
        z = np.arange(area_shape[0])

        z, y, x = np.meshgrid(z, y, x, indexing='ij')
        coords = (x, y, z)
        guess = [area_shape[0]/2, area_shape[1]/2, area_shape[2]/2, selected_area.max(), 1, 1, 4, selected_area.min()]
        bounds = ([0, 0, 0, 0, 0.01, 0.01, 0.01, -np.inf], [10, 10, 10, np.inf, 2.25, 2.25, 5, np.inf])
        try:
            fit, _ = curve_fit(get_3d_gaussian, coords, selected_area.ravel(), p0=guess, bounds = bounds, method = 'trf', maxfev = 50000) # added maxfev = 50000 on 02/09/2026
        except RuntimeError:
            try:
                fit, _ = curve_fit(get_3d_gaussian, coords, selected_area.ravel(), p0=guess, bounds = bounds, method = 'dogbox', maxfev = 50000) # added maxfev = 50000 on 02/09/2026
            except RuntimeError:
                holder = peak_coords.tolist()
                holder.pop(en)
                peak_coords = np.array(holder)
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
            tilt_shape_init = tilt.shape
            invert_shape = (tilt_shape_init[2], tilt_shape_init[1], tilt_shape_init[0])
        aretomo3_alignment = aretomo3_alignment.to_aretomo(ts_size=invert_shape)
    else:
        aretomo3_alignment = read_alignment

    # if alpha_offset != aretomo3_alignment.AlphaOffset and alpha_offset != None: # checks if alpha offset is incorrect and adjusts it
    #     aretomo3_alignment = fix_alpha_offset(aretomo3_alignment, alpha_offset)

    with mrcfile.open(tilt_path) as mrctilt:
        if aretomo3_alignment.DarkFrames == []:
            tilt = mrctilt.data
            tilt_shape = tilt.shape
        else:
            tilt = mrctilt.data
            dark_slices = [frame.section_idx for frame in aretomo3_alignment.DarkFrames]
            tilt = np.delete(tilt, dark_slices, axis = 0)
            tilt_shape = tilt.shape 
            invert_shape = tilt_shape[::-1]
    print('loaded files')        
    # aretomo3_alignment = Alignment.from_imod(read_alignment).to_aretomo(ts_size=invert_shape)
    if alpha_offset != aretomo3_alignment.AlphaOffset and alpha_offset != None: # checks if alpha offset is incorrect and adjusts it
        aretomo3_alignment = fix_alpha_offset(aretomo3_alignment, alpha_offset)


    tomo_num = re.findall(r"(._\d*_*).st",tilt_path)[0] # gets tomo number for im
    tomo_name = re.findall(r".*/(.*).st",tilt_path)[0]
    ######################################################################    

    peaks = [ent for ent in picks if center_OI[0]+radius_OI > ent[0] and ent[0] >center_OI[0]-radius_OI and center_OI[1]+radius_OI > ent[1] and ent[1] >center_OI[1]-radius_OI]
    if len(peaks) < 12:
        print(f"{tomo_name} number:{number} picked failed attempting voxel")
        invert_tomo = invert_tomo.astype(np.float32)
        _,picks = select_aunps(invert_tomo, .6, (70,50,50), center_OI.tolist(), radius_OI) 

        peaks = [ent for ent in picks if center_OI[0]+radius_OI > ent[0] and ent[0] >center_OI[0]-radius_OI and center_OI[1]+radius_OI > ent[1] and ent[1] >center_OI[1]-radius_OI]
        if len(peaks) < 12:
            print(f"{tomo_name} number:{number} failed")
            return 'fail'
        return 'fail'
    
    peak_coords_rounded = np.round(peaks).astype(int)
    peak_coords_OI, _, list_of_sigmas = find_3d_gaussian_peaks_fix(invert_tomo, peak_coords_rounded)
    make_IMOD_model_UPDATED(peak_coords_OI, f'//nrs/liza/cathy_tomos/ddw/imod_alignments/15F1-H4K2Cys_TOPTOMOS/WARP/{tomo_name}/{tomo_name}_azs_mpicks_{number}')
    plot_3d_sigmas(list_of_sigmas, tomo_name, f'/groups/liza/Pictures/{date}/{tomo_name}_az{number}:_{len(list_of_sigmas)}')
    plt.close()
    print(f'made model of {len(peak_coords_OI)} picks')
    print('')

    #ACCOUNTING FOR Diff in alignment pix v new 2.5A tilt series
    for aln in aretomo3_alignment.GlobalAlignments:
        aln.tilt = aln.tilt * -1
    #    aln.tx = aln.tx * (9.6/2.5) 
    #    aln.ty = aln.ty * (9.6/2.5)


    final_coords = align_to_tilt_series(tomo_shape, tilt_shape, peak_coords_OI, bin, aretomo3_alignment)

    base_img, circle_img = make_model_layer_components(tilt, aretomo3_alignment, 7, final_coords, shape = 'circle')
    
    # Convolve cirlce with pixel placement
    conv_image, rev_conv_coords = fourier_convolution(circle_img, base_img, False)
    f = final_coords[:-1] +1
    make_IMOD_model_UPDATED(final_coords, f'/nrs/liza/cathy_tomos/ddw/imod_alignments/15F1-H4K2Cys_TOPTOMOS/WARP/{tomo_name}/{tomo_name}_tilt_series_{alpha_offset}')


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
    
    for i, algnmt in enumerate(aretomo3_alignment.GlobalAlignments):
        algnmt.tx = algnmt.tx + shift[i][0] 
        algnmt.ty = algnmt.ty + shift[i][1]

    aretomo3_alignment.LocalAlignments = [] # if this was run w local patch correction, this removes that 
    aretomo3_alignment.NumPatches = 0 # also to revert from initial local patch correction to global only
    
    # write 2.5A pixel size alns
    #imod_aln = aretomo_to_imod(aretomo3_alignment, tilt_shape_init[::-1], 2.5)
    #write(imod_aln, f"{output_aln_path}/{tomo_name}")
    #write(aretomo3_alignment, f"{output_aln_path}/{tomo_name}.aln")
    
    # write 9.6A pixel size alns
    # for aln in aretomo3_alignment.GlobalAlignments:
    #     aln.tilt = aln.tilt * -1
    #     aln.tx = aln.tx / (9.6/2.5) 
    #     aln.ty = aln.ty / (9.6/2.5)

    # imod_aln = aretomo_to_imod(aretomo3_alignment, tilt_shape, 9.6)
    # write(imod_aln, f"{output_aln_path}/{tomo_name}_9.6A")
    # write(aretomo3_alignment, f"{output_aln_path}/{tomo_name}_9.6A.aln")
    return 'complete'



if __name__ == "__main__":
# grab tomo name --> read in mono coords --> set tomo to 3DCTF corrected tomo --> aln to init aln
    import starfile
    import ast

    centers_df = pd.read_csv('/groups/liza/temp_for_transfer/15F1-H4K2Cys_TOPTOMOS/15F1-H4K2Cys_TOPTOMOS_warp.csv', sep=',')

    tomo_names= centers_df['tomo_name']
    best = centers_df['type']
    # az_selected = centers_df['az_num']
    az_centers = get_az_centers(tomo_names,best)

    az_centers_df = pd.DataFrame(az_centers)
    az_centers_df['outcome'] = None

    outcomes = []
    #align_manual_picks(az_centers)
    second_date = ['AMmilled49-1_Position_10_2','AMmilled49-1_Position_20', 'AMmilled49-1_Position_28']

    az_centers_df = pd.read_csv("/groups/liza/temp_for_transfer/15F1-H4K2Cys_TOPTOMOS/az_centers_with_outcomes_redone_in_warp.csv")
    for i, az in enumerate(az_centers):

        folder = az['tomo_name']
        az_num = az['az_num']
        center = az['center']
        best_type = az['type']
        if folder != "AMmilled49-2_Position_20":
            continue
        if folder in second_date:
            prefix = '20260131'
        else:
            prefix = '20260130'

        os.chdir(f"/groups/liza/temp_for_transfer/15F1-H4K2Cys_TOPTOMOS/WARP/{folder}/")
    
        aln_path = f"/groups/liza/temp_for_transfer/15F1-H4K2Cys_TOPTOMOS/WARP/{folder}/{best_type}_tracking/{folder}"
        folder_path = f"/groups/liza/temp_for_transfer/15F1-H4K2Cys_TOPTOMOS/WARP/{folder}/{best_type}_tracking"

        aunps = starfile.read(f"{folder_path}/aunps/aunp_tm_BP_active_zone_all.star")
        aunps = aunps[["faCoordinateX", "faCoordinateY", "faCoordinateZ", 'active_zone']]

        picks = aunps[["faCoordinateX", "faCoordinateY", "faCoordinateZ"]].to_numpy()
        picks = np.array(picks)
        make_IMOD_model_UPDATED(aunps, f'/groups/liza/temp_for_transfer/15F1-H4K2Cys_TOPTOMOS/WARP/{folder}/imod_model')

        tomo = f"{folder_path}/{folder}_full_rec_BP_3DCTF_BIN4.mrc"
        tilt = f"/groups/liza/temp_for_transfer/15F1-H4K2Cys_TOPTOMOS/WARP/{folder}/{best_type}_tracking/{folder}.st"
        aln = f"{aln_path}"
        subprocess.run(f'mkdir /nrs/liza/cathy_tomos/ddw/imod_alignments/15F1-H4K2Cys_TOPTOMOS/WARP/{folder}', shell = True)
        subprocess.run(f'mkdir /nrs/liza/cathy_tomos/ddw/imod_alignments/15F1-H4K2Cys_TOPTOMOS/WARP/{folder}/{az_num}', shell = True)
        output_aln = f"/nrs/liza/cathy_tomos/ddw/imod_alignments/15F1-H4K2Cys_TOPTOMOS/WARP/{folder}/{az_num}"
        tilt_com_path = f"/groups/liza/temp_for_transfer/15F1-H4K2Cys_TOPTOMOS/{prefix}_{folder}/{best_type}_tracking//tilt.com"
        subprocess.run(f'rm /nrs/liza/cathy_tomos/ddw/imod_alignments/15F1-H4K2Cys_TOPTOMOS/WARP/{folder}/{az_num}/{folder}.tlt', shell = True)
        subprocess.run(f'rm /nrs/liza/cathy_tomos/ddw/imod_alignments/15F1-H4K2Cys_TOPTOMOS/WARP/{folder}/{az_num}/{folder}.xtlt', shell = True)
        subprocess.run(f'rm /nrs/liza/cathy_tomos/ddw/imod_alignments/15F1-H4K2Cys_TOPTOMOS/WARP/{folder}/{az_num}/tilt.com', shell = True)
        subprocess.run(f'rm /nrs/liza/cathy_tomos/ddw/imod_alignments/15F1-H4K2Cys_TOPTOMOS/WARP/{folder}/{az_num}/newst.com', shell = True)




        for line in open(tilt_com_path):
            if 'OFFSET' in line:
                match = re.findall(r'-?\d+\.\d+', line)
                if match:
                    alphaOffset = float(match[0])
                #else:
                #    alphaOffset = 0

        alphaOffset = 0
        date = "04_28_2026"
        outcome = realign_with_mono_selected(picks, tomo, tilt, aln, output_aln, center, 120, 4,  alphaOffset, date, az_num)
        az_centers_df.loc[i, 'outcome'] = outcome

        az_centers_df.to_csv(
            "/groups/liza/temp_for_transfer/15F1-H4K2Cys_TOPTOMOS/az_centers_with_outcomes_redone_in_warp.csv",
            index=False
        )

        outcomes.append(outcome)

    print(outcomes)
        # realign_with_mono_selected(picks, tomo, tilt, aln, output_path, (235,526), 120, 4,  0, "12_05_2025", 0)

