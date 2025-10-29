import mrcfile
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit
from skimage.feature import peak_local_max

from gold_aligner._make_IMOD_model import make_imod_model

def find_3d_gaussian_peaks(invert_tomo,     # array : initially aligned data from tomogram.mrc * -1 
                           peak_coords,     # list: list of coordinate list [x,y,z]
                           cutoff = 95      # int: disregaurd points with sigmas above certain percentile
                           ): 
    new_point_coords = []
    list_of_sigmas = []
    list_of_fitted = []
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
        bounds = ([0, 0, 0, 0, 0.1, 0.1, 0.1, -np.inf], [7, 7, 7, np.inf, 2.25, 2.25, 5, np.inf])
        fit, _ = curve_fit(get_3d_gaussian, coords, selected_area.ravel(), p0=guess, bounds = bounds, method = 'trf')
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

def get_3d_gaussian(xyz, x0, y0, z0, amp, sigma_x, sigma_y, sigma_z, background): 
    x, y, z = xyz
    g_3d = amp * np.exp(
        -(((x - x0)**2) / (2 * sigma_x**2) + 
          ((y - y0)**2) / (2 * sigma_y**2) + 
          ((z - z0)**2) / (2 * sigma_z**2))
    ) + background
    return g_3d.ravel()

def get_3d_sigmas(tomo_path,            # str : path to tomogram
                  center_oi = None,     # (int,int) : will select gold only within sphere centered at (x,y)
                  radius = None,        # int : will select gold only within sphere centered at (x,y) with radius r
                  plot = False,         # bool : if T will generate a subplot of x, y, z sigma distribution violin plot
                  cutoff = 95,           # int : percentile at which points are dropped
                  debugger = True
                  ):
    with mrcfile.open(tomo_path) as mrctomo: # get tomo data
        invert_tomo = mrctomo.data * -1
    
    peaks = peak_local_max(invert_tomo, min_distance=3, threshold_rel = .4, exclude_border = (70,50,50)).tolist()
    for entry in peaks: # convert zyx -> xyz
            entry[0], entry[2] = entry[2], entry[0]
            peaks[peaks.index(entry)] = entry
    if center_oi != None and radius != None:
        peak_coords = [ent for ent in peaks if center_oi[0]+radius > ent[0] and ent[0] >center_oi[0]-radius and center_oi[1]+radius > ent[1] and ent[1] >center_oi[1]-radius]
    else: 
        peak_coords = peaks
    sigmas = np.array(find_3d_gaussian_peaks(invert_tomo, peak_coords, cutoff)[2])
    fits = np.array(find_3d_gaussian_peaks(invert_tomo, peak_coords, cutoff)[1])
    
    
    if debugger == True:
        peak_coords_array = np.array(peak_coords)
        make_imod_model(peak_coords_array, tomo_num= 64, custom_name="dubug1")
    
    
    if plot == True:
        sig_title = ['x','y','z']
        fig, axs = plt.subplots(1,3, sharex=True, sharey=True) 
        sigmas = np.array(sigmas)
        axs.ravel()
        for i in range(sigmas.shape[1]):
            axs[i].set_title(sig_title[i])
            axs[i].violinplot(sigmas[:,i])
    return sigmas, fits

def make_3d_sigma_plot(tomos, tomo_name, center = None, radius = None, save_path=None):
    sigma_tomos = []
    for n in range(len(tomos)):
        sig, fitt = get_3d_sigmas(tomos[n], center, radius, plot = False, cutoff = 95)
        sigma_tomos.append(sig)
        # fitts.append(fitt)
        sigmas = np.array(sigma_tomos[n])
    sig_title = ['x','y','z']
    lengths = []
    
    fig, axs = plt.subplots(1,3, sharex=True, sharey=True, figsize=(10, 12))
    axs.ravel()
    for n in range(len(sigma_tomos)):
        sigmas = np.array(sigma_tomos[n])
        print(len(sigmas))
        lengths.append(len(sigmas))
        for i in range(sigmas.shape[1]):
            axs[i].set_title(sig_title[i])
            parts = axs[i].violinplot(sigmas[:,i], showmedians=False)
            violin_color = parts['cbars'].get_edgecolor().flatten()
            axs[i].plot([],[], color = violin_color, label = tomo_name[n])
            if i == sigmas.shape[1]-1:
                axs[i].legend()
    fig.suptitle("Distribution of Gaussian Standard Deviation by Axis")
    # plt.show()
    if save_path != None:
        plt.savefig(f"{save_path}", dpi = 300)

def compare_3d_sigmas_plot(tomos, tomo_names, mask= None, save_path = None):
    sigma_tomos = []
    for n in range(len(tomos)):
        sig, fitt = get_3d_sigmas(tomos[n], center, radius, plot = False, cutoff = 95)
        sigma_tomos.append(sig)
        # fitts.append(fitt)
        sigmas = np.array(sigma_tomos[n])
    sig_title = ['x','y','z']
    lengths = []
    
    fig, axs = plt.subplots(1,3, sharex=True, sharey=True, figsize=(10, 12))
    axs.ravel()
    for n in range(len(sigma_tomos)):
        sigmas = np.array(sigma_tomos[n])
        print(len(sigmas))
        lengths.append(len(sigmas))
        for i in range(sigmas.shape[1]):
            axs[i].set_title(sig_title[i])
            parts = axs[i].violinplot(sigmas[:,i], showmedians=False)
            violin_color = parts['cbars'].get_edgecolor().flatten()
            axs[i].plot([],[], color = violin_color, label = tomo_names[n])
            if i == sigmas.shape[1]-1:
                axs[i].legend()
    fig.suptitle("Distribution of Gaussian Standard Deviation by Axis")
    # plt.show()
    if save_path != None:
        plt.savefig(f"{save_path}", dpi = 300)
if __name__ == '__main__':
    tomo = "/nrs/liza/hoyoung_dimer_tomos/dimer_tomos/dimer_tomo64_mixgauss_fixed_distance_v3_2ndITR/20250210_HippWaffle_64_Vol.mrc" 
    tomo_1 = "/nrs/liza/hoyoung_dimer_tomos/dimer_tomos/dimer_tomo64_mixgauss_fixed_distance_v3/20250210_HippWaffle_64_Vol.mrc"
    center = (460,800)
    radius = 120

    tomo = "/nrs/liza/aretomoe3_rm_patch_gausses_alpha_circle/20231017_EGmilled24-2_68_Vol.mrc"
    tomo_1 = "/nrs/liza/aretomoe3_rm_patch_gausses_alpha_circle_2/20231017_EGmilled24-2_68_Vol.mrc"


    tomos_path = [tomo, tomo_1]
    tomo_names = ["One Itr", "Two Itr"]
    tomo_names = ["elipse", "circle"]
    tomo_names = ["one", "two"]


    center = (543, 462)
    radius = 120

    make_3d_sigma_plot(tomos_path, tomo_names, center, radius)
    print('end')