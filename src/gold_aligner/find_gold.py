import mrcfile
import numpy as np
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
# sets area of interest and its radius
center = (543, 462)
radius = 120
def get_AUNP_positions(center, radius):
    with mrcfile.open("/nrs/liza/aretomoe3_remove_patch_reconstruct/20231017_EGmilled24-2_68_Vol.mrc") as mrc:
        invert_img = mrc.data * -1 # flip black and white
        size_of_tomo = mrc.data.shape
        peak_coords = peak_local_max(invert_img, min_distance = 3,threshold_rel = .4, exclude_border = (70, 50,50)).tolist() # gets darkest peak

        for entry in peak_coords: # switch x and z to invert zyx --> xyz
            entry[0], entry[2] = entry[2], entry[0]
            peak_coords[peak_coords.index(entry)] = entry
        peak_coords = np.array(peak_coords)
        ls_p_coords = list(zip(*peak_coords.tolist()))
        peak_coords_OI = [ent for ent in peak_coords if center[0]+radius > ent[0] and ent[0] >center[0]-radius and center[1]+radius > ent[1] and ent[1] >center[1]-radius] # filters points selected by if they are within radius
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
    return size_of_tomo, peak_coords_OI


### Align gold particles to tilt series
def align_to_tilt(size_of_tomo, peak_coords_OI):
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
    return mrc_img, final_coords


# Create model layer of AUNPs based on their position
def create_model_layer(mrc_img, final_coords):
    # creates img where each selected pixel = 1 while background = 0
    base_img = np.zeros_like(mrc_img)
    for gold in final_coords:
        gold = gold.astype('int')
        base_img[gold[2]][gold[1]][gold[0]] = 1
    # create perfect circle
    radii = 7
    side = radii

    circle_array = np.zeros([side*2, side*2])
    center_point = (side, side)
    for x in range(0,side*2):
        for y in range(0,side*2):
            r = (x-center_point[0])**2 + (y-center_point[1])**2
            if round(np.sqrt(r)) < radii:
                circle_array[x][y] = 1

    # convolve the circle with the pixel placement
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

    # saves as both npy array and mrc file
    conv_coords = np.array(rev_conv_coords)
    np.save('conv_coords_post_alignment.npy', conv_coords)
    np.save('conv_image_post_alignment.npy', conv_image)
    conv_image = conv_image.numpy()
    mrcfile.new("/nrs/liza/gold-aligner/gold_conv_images/gold_convolution_layer_post_alignment.mrc", data=conv_image.astype('float32'), overwrite = True) 
    print('conv done')
    print(conv_coords.shape)


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

    modelPeak.to_file('2d_model_conv_post_alignment.mod')
    return conv_coords, conv_image



def phase_cross_correlation(mrc_img, conv_gold_im_path):
# take cross corr and get proper shifts for each tilt  
    with mrcfile.open(conv_gold_im_path) as mrc:
        model_conv = mrc.data
        mrc_img_inverted = mrc_img * -1
        raw_shift = np.zeros([mrc.data.shape[0],2])
        shift = np.zeros([mrc.data.shape[0],2])
        y_shape = int(model_conv.shape[2]/2)
        x_shape = int(model_conv.shape[1]/2)
        for i, slice in enumerate(model_conv):
            model_fft = np.fft.rfft2(model_conv[i])
            img_fft = np.fft.rfft2(mrc_img_inverted[i])
            cross = img_fft * model_fft.conj() 
            phase = np.fft.fftshift(np.fft.irfft2(cross))
            cropped_phase = phase[x_shape-50:x_shape+50, y_shape-50:y_shape+50]
            raw_shift[i] = np.unravel_index(np.argmax(cropped_phase, axis=None), cropped_phase.shape)
            shift[i] = raw_shift[i][0] - 50, raw_shift[i][1] - 50
            #shift[i], error[i], phasediff[i] = phase_cross_correlation(mrc_img_flipped[i], model_conv[i], disambiguate = True, space = "real")
    return shift


## INITIAL RUN
conv_gold_im_path = "/nrs/liza/gold-aligner/gold_conv_images/gold_convolution_layer.mrc"
mrc_img, final_coords = align_to_tilt(get_AUNP_positions(center, radius)[0], get_AUNP_positions(center, radius)[1])
shifts = phase_cross_correlation(mrc_img, conv_gold_im_path)

# use shifts to create .aln file
aretomo3_alignment = read("/nrs/liza/aretomoe3_remove_patch/20231017_EGmilled24-2_68.aln")
with mrcfile.open(conv_gold_im_path) as mrc:
    model_conv = mrc.data
    for i, algnmt in enumerate(aretomo3_alignment.GlobalAlignments):
        algnmt.tx = algnmt.tx + shifts[i][1] 
        algnmt.ty = algnmt.ty + shifts[i][0]

write(aretomo3_alignment, "/nrs/liza/aretomoe3_remove_patch/20231017_EGmilled24-2_68_fixed_xy.aln")
if os.path.isfile("/nrs/liza/aretomoe3_remove_patch/20231017_EGmilled24-2_68_old_fixedtilt.aln") == False:
    os.rename('/nrs/liza/aretomoe3_remove_patch/20231017_EGmilled24-2_68.aln', '/nrs/liza/aretomoe3_remove_patch/20231017_EGmilled24-2_68_old_fixedtilt.aln')
    os.rename('/nrs/liza/aretomoe3_remove_patch/20231017_EGmilled24-2_68_fixed_xy.aln', '/nrs/liza/aretomoe3_remove_patch/20231017_EGmilled24-2_68.aln')
else:
    print("File has already been fixed, please review if you'd like to continue")






## adjust position of conv_gold to see if shift will be 0

# aretomo3_alignment = read("/nrs/liza/aretomoe3_remove_patch/20231017_EGmilled24-2_68.aln")
# x_coord, y_coord, z_coord = zip(*final_coords)
# final_subset_coords = []
# for slice in range(0,mrc_img.shape[0]):
#     index_range = np.where(np.array(z_coord) == slice)
#     subset_coords = final_coords[index_range[0][0]:index_range[0][-1]]
#     for coord in subset_coords:
#         final_subset_coords.append([coord[0] + shifts[slice][1], coord[1] + shifts[slice][0], slice])
# final_subset_coords = np.array(final_subset_coords)

# conv_gold_im_path = "/nrs/liza/gold-aligner/gold_conv_images/gold_convolution_layer.mrc"
# mrc_img, final_coords = align_to_tilt(get_AUNP_positions(center, radius)[0], get_AUNP_positions(center, radius)[1])
# shifts = phase_cross_correlation(mrc_img, conv_gold_im_path)
# conv_gold_im_path = "/nrs/liza/gold-aligner/gold_conv_images/gold_convolution_layer_post_alignment.mrc"
# shifts_post_alignment = phase_cross_correlation(mrc_img, conv_gold_im_path)

# with mrcfile.open(conv_gold_im_path) as mrc:
#     gold_mask = mrc.data
#     for z, shift in enumerate(shifts_post_alignment):
#         np.roll(gold_mask[z], int(shift[0]/2), axis = 1)
#         np.roll(gold_mask[z], int(shift[1]/2), axis = 0)
#     conv_gold_im_path = "/nrs/liza/gold-aligner/gold_conv_images/gold_convolution_layer.mrc"
#     mrc_path = "/nrs/liza/aretomoe3_remove_patch/20231017_EGmilled24-2_68.mrc"
#     test = phase_cross_correlation(gold_mask,mrc_path)

#create_model_layer(mrc_img, final_subset_coords)

#list1, list2, list3 = zip(*final_coords)
# chosen_slice = 16
# chosen_indx = np.where(np.array(list3) == chosen_slice)
# test_coords = final_coords[chosen_indx[0][0]:chosen_indx[0][-1]]
# final_test = np.zeros_like(test_coords)
# for i, entry in enumerate(test_coords):
#     final_test[i] = entry[0] + shifts[chosen_slice][1], entry[1] + shifts[chosen_slice][0], chosen_slice


# modelPeak = ImodModel(objects=[
#     Object(
#         # header = ObjectHeader(
#         #     contsize = len(countours)
#         # )
#         contours=[
#             Contour(
#                 header=ContourHeader(
#                     psize= final_test.shape[0],
#                     flags=16,
#                     time=0,
#                     surf=0,
#                 ),
#                 points = final_test,
#             )
#         ]
#     )
# ])
#modelPeak.to_file('test.mod')
