
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from imodmodel import ImodModel
from imodmodel.models import (
    Contour,
    ContourHeader,
    Object,
)

# invert_tomo             array : initially aligned data from tomogram.mrc * -1 
# peak_coords             list: list of coordinate list [x,y,z]

def find_3d_gaussian_peaks(invert_tomo, peak_coords): 
    new_point_coords = []
    list_of_sigmas = []
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
        bounds = ([2, 2, 2, 0, 0.1, 0.1, 0.1, -np.inf], [7, 7, 7, np.inf, 2.25, 2.25, 3, np.inf])
        fit, _ = curve_fit(get_3d_gaussian, coords, selected_area.ravel(), p0=guess, bounds = bounds, method = 'trf')
        x0, y0, z0, amp, sigx, sigy, sigz, back = fit
        x_c = x0-5 + entry[0]
        y_c = y0-5 + entry[1]
        z_c = z0-5 + entry[2]
        new_point_coords.append([x_c,y_c,z_c])
        list_of_sigmas.append([sigx,sigy,sigz])
        
    new_point_coords = np.array(new_point_coords)

    unzip_sig = list(zip(*list_of_sigmas))
    cutoff = [np.percentile(np.array(unzip_sig[dim]), 95) for dim in range(3)] 
    indx_discard = [np.where(unzip_sig[dim] >= cutoff[dim]) for dim in range(3)]
    discard = np.concatenate([indx_discard[0][0], indx_discard[1][0], indx_discard[2][0]])
    peak_coords_OI = np.delete(new_point_coords, discard, axis = 0)



    return peak_coords_OI

def get_3d_gaussian(xyz, x0, y0, z0, amp, sigma_x, sigma_y, sigma_z, background):
    x, y, z = xyz
    g_3d = amp * np.exp(
        -(((x - x0)**2) / (2 * sigma_x**2) + 
          ((y - y0)**2) / (2 * sigma_y**2) + 
          ((z - z0)**2) / (2 * sigma_z**2))
    ) + background
    return g_3d.ravel()




# image                   array : initially aligned data from tomogram.mrc * -1 
# peak_coords             list: list of coordinate list [x,y,z]
def find_2d_gaussian_peak(image, peak_coords): 
    if image[tuple(peak_coords)] < 0:
        image = image - image.min()
    shape_img = image.shape
    x = np.arange(shape_img[1]) # x = 0 would be the first column 
    y = np.arange(shape_img[0]) # y = 0 would be the first row

    x,y = np.meshgrid(x,y)
    coords = (x, y)
    guess = [peak_coords[1], peak_coords[0], image.max(), shape_img[0]/10, shape_img[1]/10, image.min()]
    bounds = ([0, 0, 0, 0.1, 0.1, -np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
    fit, _ = curve_fit(get_2d_gaussian, coords, image.ravel(), p0=guess, bounds = bounds, method = 'trf')
    x0, y0, amp, sigx, sigy, back = fit
    fitted_gauss = get_2d_gaussian((x,y),*fit).reshape(shape_img)
    raw_shift = (x0, y0) 
    sigmas = (sigx, sigy)

    return raw_shift, fitted_gauss, sigmas


def get_2d_gaussian(xy, x0, y0, amp, sigma_x, sigma_y, background):
    x, y = xy
    g_2d = amp * np.exp(
        -(((x - x0)**2) / (2 * sigma_x**2) + 
          ((y - y0)**2) / (2 * sigma_y**2)) 
    ) + background
    return g_2d.ravel()


# TESTING ##

# fig, axs = plt.subplots(1,3)
# for i in range(0,30,10):
#     cropped_phase = np.load("/nrs/liza/gold-aligner/cropped_phases_for_gaussian_testing.npy")
#     guess_shift = np.unravel_index(np.argmax(cropped_phase[i], axis=None), cropped_phase[i].shape)
#     output, gauss = find_2d_gaussian_peak(cropped_phase[i], guess_shift)
    
#     axs[i//10].imshow(cropped_phase[i])
#     axs[i//10].imshow(gauss, cmap='hot', alpha = 0.5)
#     axs[i//10].plot(output[0], output[1], 'rx')

# print('end')