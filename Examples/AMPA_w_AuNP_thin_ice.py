
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

tomo_path = '/nrs/liza/cathy_tomos/thin_ice_purifiedAMPA_AuNPconj/20251031_grid198_nAMPAR-Fab-H12Cys-AuNP_Position_4/patch_tracking/20251031_grid198_nAMPAR-Fab-H12Cys-AuNP_Position_4_full_rec_BP_3DCTF_BIN4.mrc' 
aln_path = '/nrs/liza/cathy_tomos/thin_ice_purifiedAMPA_AuNPconj/20251031_grid198_nAMPAR-Fab-H12Cys-AuNP_Position_4/patch_tracking/20251031_grid198_nAMPAR-Fab-H12Cys-AuNP_Position_4'
tilt_path = '/nrs/liza/cathy_tomos/thin_ice_purifiedAMPA_AuNPconj/20251031_grid198_nAMPAR-Fab-H12Cys-AuNP_Position_4/20251031_grid198_nAMPAR-Fab-H12Cys-AuNP_Position_4.mrc'
tilt_com_path = f"/nrs/liza/cathy_tomos/thin_ice_purifiedAMPA_AuNPconj/20251031_grid198_nAMPAR-Fab-H12Cys-AuNP_Position_4/patch_tracking/tilt.bakfa"
output_aln_path = "/nrs/liza/cathy_tomos/thin_ice_purifiedAMPA_AuNPconj/20251031_grid198_nAMPAR-Fab-H12Cys-AuNP_Position_4/aunp_aln/"
tomo_name = 'AuNP_Position_4'
date = '01_29_2026'

for line in open(tilt_com_path):
    if 'OFFSET' in line:
        match = re.findall(r'-?\d+\.\d+', line)
        if match:
            alphaOffset = float(match[0])

alpha_offset = alphaOffset

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

if alpha_offset != aretomo3_alignment.AlphaOffset and alpha_offset != None: # checks if alpha offset is incorrect and adjusts it
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

_, peak_coords = select_aunps(invert_tomo, 0.5, (70,50,50), (tomo_shape[2]/2, tomo_shape[1]/2), 800)

peak_coords_OI, _, list_of_sigmas = find_3d_gaussian_peaks(invert_tomo, peak_coords)
make_IMOD_model_UPDATED(np.array(peak_coords_OI), f'/nrs/liza/cathy_tomos/thin_ice_purifiedAMPA_AuNPconj/20251031_grid198_nAMPAR-Fab-H12Cys-AuNP_Position_4/aunp_aln/AuNP_Position_4')
print('picked ' + str(len(peak_coords_OI)) + ' aunps')


plot_3d_sigmas(list_of_sigmas, tomo_name, f'/groups/liza/Pictures/{date}/{tomo_name}:_{len(list_of_sigmas)}')
plt.close()


final_coords = align_to_tilt_series(tomo_shape, tilt_shape, peak_coords_OI, 4, aretomo3_alignment)

base_img, circle_img = make_model_layer_components(tilt, aretomo3_alignment, 7, final_coords, shape = 'circle')

# Convolve cirlce with pixel placement
conv_image, rev_conv_coords = fourier_convolution(circle_img, base_img, False)
f = final_coords[:-1] +1
# make_IMOD_model_UPDATED(final_coords, f'/nrs/liza/cathy_tomos/ddw/imod_alignments/15F1and5F11_TOPTOMOS_inprogress2/{tomo_name}/{tomo_name}_ZEROalphaOffset')


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

for i, algnmt in enumerate(aretomo3_alignment.GlobalAlignments):
    algnmt.tx = algnmt.tx + shift[i][0] 
    algnmt.ty = algnmt.ty + shift[i][1]

aretomo3_alignment.LocalAlignments = [] # if this was run w local patch correction, this removes that 
aretomo3_alignment.NumPatches = 0 # also to revert from initial local patch correction to global only

imod_aln = aretomo_to_imod(aretomo3_alignment, invert_shape, 2.5)

write(imod_aln, f"{output_aln_path}/{tomo_name}")
write(aretomo3_alignment, f"{output_aln_path}/{tomo_name}.aln")

