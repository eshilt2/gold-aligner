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

### Get AUNPs in tomogram
center = (543, 462)
radius = 120
with mrcfile.open("/nrs/liza/aretomoe3_remove_patch_reconstruct/20231017_EGmilled24-2_68_Vol.mrc") as mrc:
    invert_img = mrc.data * -1 # flip black and white
    size_of_tomo = mrc.data.shape
    peak_coords = peak_local_max(invert_img, min_distance = 3,threshold_rel = .4, exclude_border = (70, 50,50)).tolist() 

    for entry in peak_coords:
        entry[0], entry[2] = entry[2], entry[0]
        peak_coords[peak_coords.index(entry)] = entry
    peak_coords = np.array(peak_coords)
    ls_p_coords = list(zip(*peak_coords.tolist()))
    peak_coords_OI = [ent for ent in peak_coords if center[0]+radius > ent[0] and ent[0] >center[0]-radius and center[1]+radius > ent[1] and ent[1] >center[1]-radius]
    img = mrc.data[1,:,:]
    peak_coords_OI = np.array(peak_coords_OI)
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

modelPeak.to_file('peak_cropped_model.mod')


### Align gold particles to tilt series
## move tomogram to have center at 0,0,0
translation = T(torch.tensor([[-size_of_tomo[2]/2, -size_of_tomo[1]/2, -size_of_tomo[0]/2]]))
homogenise_coords = homogenise_coordinates(peak_coords_OI.tolist())
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
     mrc_img = mrc_tilt.data
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

modelPeak.to_file('2d_model_r_z_y_cropped.mod')


# Create model layer of AUNPs based on their position
base_img = np.zeros_like(mrc_img)
for gold in final_coords:
    gold = gold.astype('int')
    base_img[gold[2]][gold[1]][gold[0]] = 1
# create perfect circle
radii = 7
side = radii
# if radii % 2 == 0:
#      side = radii
# else:
#      side = radii + 1

circle_array = np.zeros([side*2, side*2])
center_point = (side, side)
for x in range(0,side*2):
    for y in range(0,side*2):
        r = (x-center_point[0])**2 + (y-center_point[1])**2
        if round(np.sqrt(r)) < radii:
             circle_array[x][y] = 1

kernal = torch.tensor(circle_array.astype('float'))
image = torch.tensor(base_img.astype('float'))
conv_image = torch.zeros_like(image)
rev_conv_coords = []
for i, layer in enumerate(image):
    conv_image[i] = torch.nn.functional.conv2d(layer.unsqueeze(0).unsqueeze(0), kernal.unsqueeze(0).unsqueeze(0), padding = 'same')
    conv_coords_zip = np.where(torch.asarray(conv_image[i]) >= 1)
    conv_coords_zip = (np.array([i] * len(conv_coords_zip[0])),) + conv_coords_zip
    conv_coords = list(zip(*conv_coords_zip))
    for entry in conv_coords:
        list_holder = list(entry)
        list_holder.reverse()
        rev_conv_coords.append(list_holder)

     
conv_coords = np.array(rev_conv_coords)
np.save('conv_coords.npy', conv_coords)
np.save('conv_image.npy', conv_image)
print('conv done')
print(conv_coords.shape)
#plt.imshow(conv_image[i])
#plt.show()

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

modelPeak.to_file('2d_model_conv.mod')
