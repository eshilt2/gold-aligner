""" Everything to do with creating mask, convolving, and cross correlating for alignment"""


import numpy as np
import torch
import math
import matplotlib.pyplot as plt
from torch_image_interpolation.image_interpolation_2d import insert_into_image_2d
from numpy.fft import rfft2, irfft2, fftshift
from gold_aligner._fit_gaussian import find_2d_gaussian_peak
from scipy.ndimage import rotate




def make_model_layer_components(tilt,               # array : tilt series array
                                aln,                # 
                                conv_radius,        # int   : radius of sphere/circle
                                final_coords,       # array : coordinates of points
                                shape = 'circle'    # str   : specify circle or sphere projection used for cross correlation  
                                ):
    
    base_img = np.zeros_like(tilt)
    
    if shape == 'circle':
        shape_img = make_circle(conv_radius)
    elif shape == 'elipse':
        shape_img = make_sphere(conv_radius, aln)

    for i, slice in enumerate(base_img):
        adj_coords = [np.array([coords[1]+1, coords[0]+1]) for coords in final_coords[np.where(final_coords[:,2] == i)]]
        layer, _ = insert_into_image_2d(
        values = torch.ones(len(adj_coords), dtype=torch.float32),
        image=torch.tensor(slice, dtype=torch.float32),
        coordinates=torch.tensor(adj_coords.copy(), dtype=torch.float32))
        base_img[i] = layer
    return base_img, shape_img 

def make_circle(conv_radius     # int : radius in pixels of circle
                ):
    # creating circle array
    circle_img = np.zeros([conv_radius*2, conv_radius*2])
    center_point = (conv_radius, conv_radius)
    for x in range(conv_radius*2):
        for y in range(conv_radius*2):
            r = (x-center_point[0])**2 + (y-center_point[1])**2
            if round(np.sqrt(r)) < conv_radius:
                circle_img[x][y] = 1
    return circle_img

def make_sphere(radius, aln): # so far not used but here for implementation
    sphere_coords = []
    sphere = np.zeros([radius*2, radius*2, radius*2])
    center_point = (radius, radius, radius)
    for x in range(radius*2):
        for y in range(radius*2):
            for z in range(radius*2):
                r = (x-center_point[0])**2 + (y-center_point[1])**2 + (z-center_point[2])**2
                if round(np.sqrt(r)) < radius:  # can I make this more accurate by giving a gradient around the edges??
                    sphere[x][y][z] = 1 
                    sphere_coords.append([x,y,z])
                    #sphere[x,y,z] = 1-(r/radius)

    list_of_elipses = []

    for i, algnmt in enumerate(aln.GlobalAlignments):
        sphere_vol = rotate(sphere, angle=algnmt.tilt, axes=(0,2), reshape=False, order=1)
        sphere_vol = rotate(sphere_vol, angle=algnmt.rot, axes=(0,2), reshape=False, order=1)
        
        proj_elipse = np.sum(sphere_vol, axis = 2)
        list_of_elipses.append(proj_elipse)
    
    return list_of_elipses

def convolve_image(circle_img,      # array     : circle with wich to convolve
                   base_img,        # array     : the image with peaks at which to convolve circle 
                   model = False    # bool      : if T returns coords for creation of imod model 
                   ): # Execution time: 35.0942 seconds on a 7 radius circle and 4746 3370 33 tomo
    if type(circle_img) is not list:
        kernal = torch.tensor(circle_img.astype('float'))
    image = torch.tensor(base_img.astype('float'))
    rev_conv_coords = []
    conv_image = torch.zeros_like(image)

    # convolving base and circle img
    for i, layer in enumerate(image):
        if type(circle_img) is list:
            kernal = torch.tensor(circle_img[i].astype('float'))
        conv_image[i] = torch.nn.functional.conv2d(layer.unsqueeze(0).unsqueeze(0), kernal.unsqueeze(0).unsqueeze(0), padding = 'same')
        
        if model == True:
            conv_coords_zipped = np.where(torch.asarray(conv_image[i]) >= 1)
            conv_coords_zipped = (np.array([i] * len(conv_coords_zipped[0])),) + conv_coords_zipped
            conv_coords = list(zip(*conv_coords_zipped))
            for entry in conv_coords:
                holder = list(entry)
                holder.reverse()
                rev_conv_coords.append(holder)
    return conv_image, rev_conv_coords

def cross_corr(tilt,            # array : numpy a 
               conv_image,
               sigma = None 
               ):
        raw_shift = np.zeros([tilt.shape[0],2])
        shift = np.zeros([tilt.shape[0],2])
        saved_cropped_phase = np.zeros([conv_image.shape[0],100,100])


        y_shape = int(conv_image.shape[2]/2)
        x_shape = int(conv_image.shape[1]/2)
        for i, slice in enumerate(conv_image):
            tilt_fft = rfft2(tilt[i])
            conv_img_fft = rfft2(conv_image[i])
            cross = tilt_fft*conv_img_fft.conj()
            phase = fftshift(irfft2(cross))
            cropped_phase = phase[x_shape-50:x_shape+50, y_shape-50:y_shape+50]
            raw_shift[i] = np.unravel_index(np.argmax(cropped_phase, axis=None), cropped_phase.shape)
            raw_shift[i], _, sigmas = find_2d_gaussian_peak(cropped_phase, raw_shift[i].astype('int'), sigma)
            shift[i] = raw_shift[i][0] - 50, raw_shift[i][1] - 50
            saved_cropped_phase[i] = cropped_phase
        return sigmas, shift, saved_cropped_phase


def plot_xcorr_peaks(saved_cropped_phase, shift, plot_name):
    fig, axs = plt.subplots(math.ceil(np.sqrt(saved_cropped_phase.shape[0])), math.ceil(np.sqrt(saved_cropped_phase.shape[0])), figsize=(10, 10))
    axs = axs.ravel()
    for i, cross in enumerate(saved_cropped_phase):
        axs[i].imshow(saved_cropped_phase[i])
        axs[i].plot(shift[i][0]+50, shift[i][1]+50, 'rx', markersize=1)
        axs[i].vlines(50,0,99, color = 'white', linestyles='dashed', linewidth=0.25)
        axs[i].hlines(50,0,99, color = 'white', linestyles='dashed', linewidth=0.25)
    
    fig.suptitle(f'{plot_name}')
    return fig


def xcorr_xcorr(): # try xcorr an expected peak with xcorr plot to get true peak rather than fitting noise
    return

def fourier_convolution(circle_img,      # array     : circle with wich to convolve
                base_img,          # array     : the image with peaks at which to convolve circle
                model = False       # bool      : return IMOD model
                ):# Execution time: 25.7764 seconds on a 7 radius circle and 4746 3370 33 tomo
    if circle_img.shape != base_img.shape[0:2]:
        circle_img = np.pad(circle_img, [(int((base_img.shape[1]-circle_img.shape[0])/2), int((base_img.shape[1]-circle_img.shape[0])/2)), (int((base_img.shape[2]-circle_img.shape[1])/2), int((base_img.shape[2]-circle_img.shape[1])/2))]) #((base_img.shape[1] - circle_img.shape[1])/2))
    
    if type(circle_img) is not list:
        kernal = torch.tensor(circle_img.astype('float'))
    image = torch.tensor(base_img.astype('float'))
    conv_image = np.zeros_like(image)
    kernal_fft = rfft2(kernal)
    rev_conv_coords = []

    for i, slice in enumerate(conv_image):
            image_fft = rfft2(image[i])
            slice = fftshift(irfft2(kernal_fft* image_fft))
            slice = np.roll(slice, shift=(-1, -1), axis=(0, 1))
            plt.imshow(slice)
            conv_image[i] = slice


    if model == True:
        conv_coords_zipped = np.where(torch.asarray(conv_image[i]) >= 1)
        conv_coords_zipped = (np.array([i] * len(conv_coords_zipped[0])),) + conv_coords_zipped
        conv_coords = list(zip(*conv_coords_zipped))
        for entry in conv_coords:
            holder = list(entry)
            holder.reverse()
            rev_conv_coords.append(holder)
    return conv_image, rev_conv_coords