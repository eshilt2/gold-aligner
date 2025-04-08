import mrcfile
import numpy as np
from numpy.fft import rfft2, irfft2, fftshift
import os
from scipy import ndimage as ndi
from cryoet_alignment import read
from cryoet_alignment import write
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torch_affine_utils.transforms_3d import Rx, Ry, Rz, T, S
from torch_affine_utils import homogenise_coordinates
from skimage.feature import peak_local_max
from imodmodel import ImodModel
from imodmodel.models import (
    Contour,
    ContourHeader,
    Mesh,
    IMAT,
    SLAN,
    MeshHeader,
    Object,
    ObjectHeader,
)

# tomo_path             str : path to initially aligned tomogram.mrc file
# tilt_path             str : path to initially aligned tiltseries.mrc file
# aln_path              str : path to aretomo3 generated .aln file used for initially aligned tomogram
# output_aln_path       str : output aln file name
#
#  OPTIONAL:
# center_OI             (int,int) : will select gold only within sphere centered at (x,y)
# radius_OI             int : will select gold only within sphere centered at (x,y) with radius r
# tomo_au_model         T/F : will return .mod of selected gold in tomogram
# tilt_au_model         T/F : will return .mod of selected gold in tilt series
# tilt_conv_au_model    T/F : will return .mod of convolved gold in tilt series
# binning               flt : binning used in original aretomo3 tomogram reconstruction, default 4.85
# conv_radius           int : sets size of gold particle used for convolution



def realign_gold(tomo_path, tilt_path, aln_path, output_aln_path, binning = 4.85, center_OI = None, radius_OI = None, tomo_au_model = False, tilt_au_model = False, conv_radius = 7, tilt_conv_au_model = False):
    
    ### Load in all files ################################################
    with mrcfile.open(tomo_path) as mrctomo: # get tomo data
        invert_tomo = mrctomo.data * -1 #flip black and white so peak_local_max picks up dark points
        tomo_shape = invert_tomo.shape
    
    aretomo3_alignment = read(aln_path)

    with mrcfile.open(tilt_path) as mrctilt:
         tilt = mrctilt.data
         tilt_shape = tilt.shape 

    ######################################################################

    ###  Select gold particles in tomo  ##################################
    peak_coords = peak_local_max(invert_tomo, min_distance=3, threshold_rel = .4, exclude_border = (70,50,50)).tolist()

    for entry in peak_coords: # convert zyx -> xyz
            entry[0], entry[2] = entry[2], entry[0]
            peak_coords[peak_coords.index(entry)] = entry
    peak_coords_OI = np.array(peak_coords)
    
   # OPTIONAL
    if center_OI != None:  # get only the points within a certain area of the tomogram
        peak_coords_OI = np.array([ent for ent in peak_coords if center_OI[0]+radius_OI > ent[0] and ent[0] >center_OI[0]-radius_OI and center_OI[1]+radius_OI > ent[1] and ent[1] >center_OI[1]-radius_OI])
    
    if tomo_au_model == True: # returns .mod model of all points selected in tomogram
        modelPeak = ImodModel(objects=[
            Object(
                contours=[
                    Contour(
                        header=ContourHeader(
                            psize= peak_coords_OI.shape[0],
                            flags=16,
                            time=0,
                            surf=0,
                        ),
                        points = peak_coords_OI,
                    )
                ]
            )
        ])

        modelPeak.to_file('tomo_au.mod')
    ######################################################################

    ### Align gold particles to tilt series ##############################
    # move tomogram AU points' origin to (0,0,0)
    translation = T(torch.tensor([[-tomo_shape[2]/2, -tomo_shape[1]/2, -tomo_shape[0]/2]]))
    homogenise_coords = homogenise_coordinates(peak_coords_OI.tolist()).float()
    translated_points = translation @ homogenise_coords.T

    # transform from 3D -> 2D
    t_coords = []
    final_coords = []
    list_z = []
    t_coords = []
    

    for z, algnmt in enumerate(aretomo3_alignment.GlobalAlignments):
        rot_z = Rz(torch.tensor([algnmt.rot]))
        rot_y = Ry(torch.tensor([algnmt.tilt]))
        translation = T(torch.tensor([algnmt.tx, algnmt.ty, 0]))
        scaling = S(torch.tensor([[bin, bin, bin]]))
        transform =  translation @ rot_z @ rot_y @ scaling # note: matrix multplx in python is from left to right
        t_points = np.array((transform @ translated_points)[0])
        for i in range(len(t_points[0])):
            t_coords.append([t_points[0][i],t_points[1][i], z])

    
    # translate back to 0,0 corner of tilt series
    translation = T(torch.tensor([tilt_shape[2]/2, tilt_shape[1]/2, 0]))
    coords_h = homogenise_coordinates(t_coords).float()
    t_coords_h = np.array(translation @ coords_h.T)
    final_coords = np.array(list(zip(t_coords_h[0], t_coords_h[1], t_coords_h[2])))

    # OPTIONAL
    if tilt_au_model == True:
            modelPeak = ImodModel(objects=[
            Object(
                # header = ObjectHeader(
                #     contsize = len(countours)
                # )
                contours=[
                    Contour(
                        header=ContourHeader(
                            psize= final_coords.shape[0],
                            flags=16,
                            time=0,
                            surf=0,
                        ),
                        points = final_coords,
                    )
                ]
            )
        ])

            modelPeak.to_file('tilt_au.mod')

    ######################################################################

    ### Create model layer of AUNPs based on their position ##############
    # creating base image where each gold particle peak is just a point
    base_img = np.zeros_like(tilt)
    for coord in final_coords:
        base_img[coord[2].astype('int')][coord[1].astype('int')][coord[0].astype('int')] = 1
    
    # creating circle array
    circle_img = np.zeros([conv_radius*2, conv_radius*2])
    center_point = (conv_radius, conv_radius)
    for x in range(conv_radius*2):
         for y in range(conv_radius*2):
              r = (x-center_point[0])**2 + (y-center_point[1])**2
              if round(np.sqrt(r)) < conv_radius:
                   circle_img[x][y] = 1
    
    # Convolve cirlce with pixel placement
    kernal = torch.tensor(circle_img.astype('float'))
    image = torch.tensor(base_img.astype('float'))
    rev_conv_coords = []
    conv_image = torch.zeros_like(image)

    # convolving base and circle img
    for i, layer in enumerate(image):
         conv_image[i] = torch.nn.functional.conv2d(layer.unsqueeze(0).unsqueeze(0), kernal.unsqueeze(0).unsqueeze(0), padding = 'same')
         conv_coords_zipped = np.where(torch.asarray(conv_image[i]) >= 1)
         conv_coords_zipped = (np.array([i] * len(conv_coords_zipped[0])),) + conv_coords_zipped
         conv_coords = list(zip(*conv_coords_zipped))
         for entry in conv_coords:
              holder = list(entry)
              holder.reverse()
              rev_conv_coords.append(holder)

    # Optional
    if tilt_conv_au_model == True:
        conv_coords = np.array(rev_conv_coords)

        modelPeak = ImodModel(objects=[
            Object(
                # header = ObjectHeader(
                #     contsize = len(countours)
                # )
                contours=[
                    Contour(
                        header=ContourHeader(
                            psize= conv_coords.shape[0],
                            flags=16,
                            time=0,
                            surf=0,
                        ),
                        points = conv_coords,
                    )
                ]
            )
        ])

        modelPeak.to_file('tilt_conv_au.mod')


    ######################################################################
    
    ### Phase cross correlation to get shift
    tilt_inv = tilt*-1
    raw_shift = np.zeros([tilt_inv.shape[0],2])
    shift = np.zeros([tilt_inv.shape[0],2])
    saved_cropped_phase = np.zeros([conv_image.shape[0],100,100])

    y_shape = int(conv_image.shape[2]/2)
    x_shape = int(conv_image.shape[1]/2)
    for i, slice in enumerate(conv_image):
         tilt_fft = rfft2(tilt_inv[i])
         conv_img_fft = rfft2(conv_image[i])
         cross = tilt_fft*conv_img_fft.conj()
         phase = fftshift(irfft2(cross))
         cropped_phase = phase[x_shape-50:x_shape+50, y_shape-50:y_shape+50]
         raw_shift[i] = np.unravel_index(np.argmax(cropped_phase, axis=None), cropped_phase.shape)
         shift[i] = raw_shift[i][0] - 50, raw_shift[i][1] - 50
         # OPTIONAL
         saved_cropped_phase[i] = cropped_phase

    ######################################################################

    ### Create new .aln file
    for i, algnmt in enumerate(aretomo3_alignment.GlobalAlignments):
        algnmt.tx = algnmt.tx + shift[i][1] 
        algnmt.ty = algnmt.ty + shift[i][0]
    
    write(aretomo3_alignment, output_aln_path)

    return saved_cropped_phase





tomo = "/nrs/liza/aretomoe3_rm_patch_recon_xy2/20231017_EGmilled24-2_68_Vol.mrc"
tilt = "/nrs/liza/aretomoe3_rm_patch/20231017_EGmilled24-2_68.mrc"
aln = "/nrs/liza/aretomoe3_rm_patch/20231017_EGmilled24-2_68.aln"
aln_output = "/nrs/liza/aretomoe3_rm_patch_recon_xy2/20231017_EGmilled24-2_68_xy2_new.aln"
bin = 4.85
center = (543, 462)
radius = 120
cropped_phase = realign_gold(tomo, tilt, aln, aln_output, bin, center, radius, tomo_au_model = True)


## Sanity check of phase cross correlation
fig, axs = plt.subplots(6, 6)
axs = axs.ravel()
for i in range(cropped_phase.shape[0]):
     axs[i].imshow(cropped_phase[i])
     axs[i].plot(50,50, "o", markersize=3)
print('end')


