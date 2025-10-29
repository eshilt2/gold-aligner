"""Mask artifacts from rotation"""
import numpy as np
from scipy.ndimage import rotate
import mrcfile
import matplotlib.pyplot as plt
import math
from scipy.ndimage import zoom

from fidder.predict import predict_fiducial_mask
from fidder.erase import erase_masked_region
import torch

def exclude_corner_artifacts(tomo_shape, # tuple : shape of tomogram (z, y, x)
                             tomo,       # array : mrc data array
                             rot,        # flt   : angle of tilt axis in degree relative to the y (vertical) axis taken from alignment file
                             scale=0.95  # flt   : artifacts are often found further in from just rotation, scale by this factor to cut off all artifacts
                             ):
    # Create full-size binary mask
    blank_mask = np.ones(tomo_shape)

    # Rotate the mask same as the tomogram
    rotated_mask = rotate(blank_mask.astype(float), rot + 90, axes=(1, 2), reshape=False, order=0) > 0.5

    # Shrink only y and x dimensions
    scale_factors = (1.0, scale, scale)  # don't shrink z
    shrunk_mask = zoom(rotated_mask.astype(float), zoom=scale_factors, order=0) > 0.5

    # Center shrunk mask inside original rotated mask shape
    padded_mask = np.zeros_like(rotated_mask, dtype=bool)
    pad_y = (rotated_mask.shape[1] - shrunk_mask.shape[1]) // 2
    pad_x = (rotated_mask.shape[2] - shrunk_mask.shape[2]) // 2
    padded_mask[:, pad_y:pad_y+shrunk_mask.shape[1], pad_x:pad_x+shrunk_mask.shape[2]] = shrunk_mask

    # Crop the padded mask to match `tomo` shape
    crop_y = (padded_mask.shape[1] - tomo.shape[1]) // 2
    crop_x = (padded_mask.shape[2] - tomo.shape[2]) // 2

    cropped_mask = padded_mask[:,
    crop_y : crop_y + tomo.shape[1],
    crop_x : crop_x + tomo.shape[2]]

    # Apply mask to the rotated tomo
    masked_tomo = tomo.copy()
    background_noise = np.random.normal(np.mean(tomo), np.std(tomo), size=tomo.shape)
    masked_tomo[~cropped_mask] = background_noise[~cropped_mask]


    return masked_tomo

def mask_fiducials(tomo):
    # load your image
    image = torch.tensor(tomo)
    rm_fids_tomo = image
    # use a pretrained model to predict a mask
    for z, slice in enumerate(image):
        mask, probabilities = predict_fiducial_mask(
            slice, pixel_spacing=1.25, probability_threshold=0.5
        )
        rm_fids_tomo[z, :, :] = erase_masked_region(image=slice, mask=mask)
        print(z)
    return rm_fids_tomo

if __name__ == '__main__':
    tomo_path = f"/nrs/liza/cathy_tomos/tomos_init_rerun/20241030_AMmilled12-2_53_Vol.mrc"
    
    with mrcfile.open(tomo_path) as mrctomo: # get tomo data
        invert_tomo = mrctomo.data * -1 #flip black and white so peak_local_max picks up dark points
        tomo_shape = invert_tomo.shape
    # masked_tomo = exclude_corner_artifacts(tomo_shape, invert_tomo, -85.2237, .95)
    rm_fids_tomo = mask_fiducials(invert_tomo)
    array_tomo = np.array(rm_fids_tomo)
    with mrcfile.new('/nrs/liza/cathy_tomos/test_reconstruction/test_rm_fiducials/20241030_AMmilled12-2_53_rm_fiducials_1.25_Vol.mrc', overwrite = True) as mrc:
        mrc.set_data(array_tomo*-1)
    print('end')