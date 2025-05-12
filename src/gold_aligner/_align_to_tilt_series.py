""" Takes points in 3D and aligns down to tilt series"""

import torch
import numpy as np
from torch_affine_utils.transforms_3d import Rx, Ry, Rz, T, S
from torch_affine_utils import homogenise_coordinates

def align_to_tilt_series(tomo_shape,        # tuple : shape of tomogram array
                         tilt_shape,        # tuple : shape of tilt series array
                         peak_coords_OI,    # array : coordinates of AuNPs (x,y,z)
                         bin,               # flt   : binning used in original aretomo3 tomogram reconstruction, default 4.85
                         aretomo3_alignment # class : alignment file created by aretomo3 when aligning tomogram
                         ):
    # move tomogram AU points' origin to (0,0,0)
    translation = T(torch.tensor([[-tomo_shape[2]/2, -tomo_shape[1]/2, -tomo_shape[0]/2]]))
    homogenise_coords = homogenise_coordinates(peak_coords_OI.tolist()).float()
    translated_points = translation @ homogenise_coords.T

    # transform from 3D -> 2D
    t_coords = []
    final_coords = []
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
    return final_coords