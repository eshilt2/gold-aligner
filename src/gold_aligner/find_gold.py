import mrcfile
import numpy as np
import os
from scipy import ndimage as ndi
from cryoet_alignment import read
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torch_affine_utils.transforms_3d import Rx, Ry, Rz, T, S
from torch_affine_utils import homogenise_coordinates
from skimage.feature import peak_local_max
from skimage.registration import phase_cross_correlation
from skimage.registration._phase_cross_correlation import _upsampled_dft
from scipy.ndimage import fourier_shift
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

### Get AUNPs in tomograph
with mrcfile.open("/nrs/liza/aretomoe3_remove_patch_reconstruct/20231017_EGmilled24-2_68_Vol.mrc") as mrc:
    invert_img = mrc.data * -1 # flip black and white
    size_of_tomo = mrc.data.shape
    peak_coords = peak_local_max(invert_img, min_distance = 3,threshold_rel = .4, exclude_border = (70, 50,50)).tolist() 

    for entry in peak_coords:
        entry[0], entry[2] = entry[2], entry[0]
        peak_coords[peak_coords.index(entry)] = entry
    peak_coords = np.array(peak_coords)
    img = mrc.data[1,:,:]

modelPeak = ImodModel(objects=[
    Object(
        contours=[
            Contour(
                header=ContourHeader(
                    psize= peak_coords.shape[0],
                    flags=16,
                    time=0,
                    surf=0,
                ),
                points = peak_coords,
            )
        ]
    )
])

modelPeak.to_file('peak_model.mod')


### Align gold particles to tilt series
## move tomogram to have center at 0,0,0
translation = T(torch.tensor([[-size_of_tomo[2]/2, -size_of_tomo[1]/2, -size_of_tomo[0]/2]]))
homogenise_coords = homogenise_coordinates(peak_coords.tolist())
homogenise_coords = homogenise_coords.float()
translated_points = translation @ homogenise_coords.T

## transform from 3D -> 2D
t_coords = []
final_coords = []
list_of_z = []
translated_pointslist_of_z = translated_points[0] # fix size for use
aretomo3_alignment = read("/nrs/liza/aretomoe3_remove_patch/20231017_EGmilled24-2_68.aln")

for algnmt in aretomo3_alignment.GlobalAlignments:
    rot_z = Rz(torch.tensor([algnmt.rot]))
    rot_y = Ry(torch.tensor([algnmt.tilt])) # testing Rz and Ry as positive
    translation = T(torch.tensor([algnmt.tx, algnmt.ty, 0]))
    scaling = S(torch.tensor([[4.85, 4.85, 4.85]]))
    transform =  translation @ rot_z @ rot_y @ scaling
    t_points = transform @ translated_points
    for i in range(0, len(t_points[0][0])-1):
        t_coords.append([t_points[0][0][i].tolist(),t_points[0][1][i].tolist(), aretomo3_alignment.GlobalAlignments.index(algnmt)])
        list_of_z.append(aretomo3_alignment.GlobalAlignments.index(algnmt))


## translate back to 0,0 corner of tilt series
with mrcfile.open("/nrs/liza/aretomoe3_remove_patch/20231017_EGmilled24-2_68.mrc") as mrc_tilt:
     tilt_img_size = mrc_tilt.data.shape
translation = T(torch.tensor([tilt_img_size[2]/2, tilt_img_size[1]/2, 0]))
coords_h = homogenise_coordinates(t_coords)
coords_h = coords_h.float()
t_coords_h = translation @ coords_h.T
for i in range(0, len(t_coords_h[0])-1):
        final_coords.append([t_coords_h[0][i].tolist(),t_coords_h[1][i].tolist(), list_of_z[i]])
final_coords = np.array(final_coords)

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

modelPeak.to_file('2d_model_r_z_y.mod')


### Old Code

# all_coords = []
# peak_all_coords = []
#     for z in range(70,mrc.data.shape[0]-1):
#         current_slice = mrc.data[z, :,:]
#         #print(mrc.data[228, 484, 485])
#         minima = current_slice.min()
#         minimaLoc = np.where(current_slice <=minima*.85) # can play around with the cutoff

#         #using local max
#         invrt_img = current_slice * -1
#         image_max = ndi.maximum_filter(invrt_img, size= 10, mode='constant')
#         peak_coords = peak_local_max(invrt_img, min_distance = 25, exclude_border
#  = 50, threshold_rel = .85).tolist()
        
#         for coord in peak_coords:
#             #print(coord)
#             coord[1], coord[0] = coord[0], coord[1]
#             peak_coords[peak_coords.index(coord)] = coord + [z]
#             #print(coord)

#         peak_all_coords = peak_all_coords + peak_coords
#         for mini in range(len(minimaLoc[0])):
#             one_coord = [x[mini-1] for x in minimaLoc] + [z]
#             one_coord[1], one_coord[0] = one_coord[0], one_coord[1] # <- this sets coord[0] to x, coord[1] to y
#             if one_coord[0] >= 50 & one_coord[0] <= (mrc.data.shape[2]-50) & one_coord[1] >= 50 & one_coord[1] <= (mrc.data.shape[1]-50):
#                 all_coords.append(one_coord)
#             #else:
#                 #print(one_coord)

#         array_coords = np.array(all_coords)
#         flt_min_coords = array_coords.astype(np.float32)

#     peak_array_coords = np.array(peak_all_coords)

# model = ImodModel(objects=[
#     Object(
#         contours=[
#             Contour(
#                 header=ContourHeader(
#                     psize= flt_min_coords.shape[0],
#                     flags=16,
#                     time=0,
#                     surf=0,
#                 ),
#                 points = flt_min_coords,
#             )
#         ]
#     )
# ])

# model.to_file('test_model.mod')

