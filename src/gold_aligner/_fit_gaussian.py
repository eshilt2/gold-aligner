import mrcfile
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit
from skimage.feature import peak_local_max
from sklearn import mixture
from imodmodel import ImodModel
from imodmodel.models import (
    Contour,
    ContourHeader,
    Object,
)           

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

def find_2d_gaussian_peak(image,        # array : initially aligned data from tomogram.mrc * -1 
                          peak_coords   # list: list of coordinate list [x,y,z]
                          ): 
    if image[tuple(peak_coords)] < 0:
        image = image - image.min()
    shape_img = image.shape
    x = np.arange(shape_img[1]) # x = 0 would be the first column 
    y = np.arange(shape_img[0]) # y = 0 would be the first row

    x,y = np.meshgrid(x,y)
    coords = (x, y)
    guess = [peak_coords[1], peak_coords[0], image.max(), shape_img[0]/20, shape_img[1]/20, image.min()] # [peak_coords[1], peak_coords[0], image.max(), shape_img[0]/10, shape_img[1]/10, image.min()]
    bounds = ([0, 0, 0, 0.1, 0.1, -np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
    fit, _ = curve_fit(get_2d_gaussian, coords, image.ravel(), p0=guess, bounds = bounds, method = 'trf', maxfev = 5000)
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

def get_3d_sigmas(tomo_path,            # str : path to tomogram
                  center_oi = None,     # (int,int) : will select gold only within sphere centered at (x,y)
                  radius = None,        # int : will select gold only within sphere centered at (x,y) with radius r
                  plot = False,         # bool : if T will generate a subplot of x, y, z sigma distribution violin plot
                  cutoff = 95           # int : percentile at which points are dropped
                  ):
    with mrcfile.open(tomo_path) as mrctomo: # get tomo data
        invert_tomo = mrctomo.data * -1
    
    peaks = peak_local_max(invert_tomo, min_distance=3, threshold_rel = .4, exclude_border = (70,50,50)).tolist()
    for entry in peaks: # convert zyx -> xyz
            entry[0], entry[2] = entry[2], entry[0]
            peaks[peaks.index(entry)] = entry
    if center_oi != None and radius != None:
        peak_coords = [ent for ent in peaks if center_oi[0]+radius > ent[0] and ent[0] >center_oi[0]-radius and center_oi[1]+radius > ent[1] and ent[1] >center_oi[1]-radius]
    sigmas = np.array(find_3d_gaussian_peaks(invert_tomo, peak_coords, cutoff)[2])
    fits = np.array(find_3d_gaussian_peaks(invert_tomo, peak_coords, cutoff)[1])

    if plot == True:
        sig_title = ['x','y','z']
        fig, axs = plt.subplots(1,3, sharex=True, sharey=True) 
        sigmas = np.array(sigmas)
        axs.ravel()
        for i in range(sigmas.shape[1]):
            axs[i].title(sig_title[i])
            axs[i].violinplot(sigmas[:,i])

    
    return sigmas, fits

def get_mixed_gaussian(invert_tomo, peak_coords, plot = False):
    i = 0
    list_of_fitted = []
    list_of_selected_area = []
    xs= np.zeros([len(peak_coords), 2])
    ys= np.zeros([len(peak_coords), 2])
    zs= np.zeros([len(peak_coords), 2])
    coords1 = np.zeros([len(peak_coords), 3])
    coords2 = np.zeros([len(peak_coords), 3])
    for n, entry in enumerate(peak_coords):

        # crops area around each point
        selected_area = invert_tomo[entry[2]-10:entry[2]+10,entry[1]-10:entry[1]+10, entry[0]-10:entry[0]+10]

        # selected_area = invert_tomo[entry[2]-10:entry[2]+10,entry[1]-10:entry[1]+10, entry[0]-10:entry[0]+10]
        # selected_area = selected_area - selected_area.min()
        selected_area = np.maximum(0, selected_area)
        list_of_selected_area.append(selected_area)
        area_shape = selected_area.shape
        
        x = np.arange(area_shape[2])
        y = np.arange(area_shape[1])
        z = np.arange(area_shape[0])
        sigma_xs = np.arange(area_shape[0])
        sigma_ys = np.arange(area_shape[0])
        sigma_zs = np.arange(area_shape[0])

        z, y, x = np.meshgrid(z, y, x, indexing='ij')
        coords = (x, y, z)
        # guess = [area_shape[0]/2, area_shape[1]/2, area_shape[2]/2, area_shape[0]/2, area_shape[1]/2, area_shape[2]/2, 0.05, selected_area.max(), selected_area.max(), area_shape[0]/5, area_shape[1]/5, area_shape[2]/5, area_shape[0]/5, area_shape[1]/5, area_shape[2]/5, selected_area.min()]
        # bounds = ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, -np.inf], [9, 9, 9, 9, 9, 9, 1, selected_area.max(), selected_area.max(), area_shape[0]/4, area_shape[1]/4, area_shape[2]/4, area_shape[0]/4, area_shape[1]/4, area_shape[2]/4, np.inf])
        # fit, _ = curve_fit(get_3dgauss_mixture, coords, selected_area.ravel(), p0=guess, bounds = bounds, method = 'trf', maxfev=1000)
        guess = [area_shape[0]/2, area_shape[1]/2, area_shape[2]/2, area_shape[0]/2, area_shape[1]/2, area_shape[2]/2, 0.5, selected_area.max(), area_shape[0]/5, area_shape[1]/5, area_shape[2]/5, selected_area.min()]
        bounds = ([0, 0, 0, 0, 0, 0, .49, 0, 0.1, 0.1, 0.1, -np.inf], [20, 20, 20, 20, 20, 20, .51, selected_area.max(), area_shape[0]/4, area_shape[1]/4, area_shape[2]/4, np.inf])
        fit, _ = curve_fit(get_3dgauss_mixture, coords, selected_area.ravel(), p0=guess, bounds = bounds, method = 'trf', maxfev=1000)
        # x0_1, y0_1, z0_1, x0_2, y0_2, z0_2, pi, amp_1, amp_2, sigx1, sigy1, sigz1, sigx2, sigy2, sigz2, back = fit
        x0_1, y0_1, z0_1, x0_2, y0_2, z0_2, pi, amp_1, sigx1, sigy1, sigz1, back = fit

        x_c1 = x0_1-10 + entry[0]
        y_c1 = y0_1-10 + entry[1]
        z_c1 = z0_1-10 + entry[2]
        x_c2 = x0_2-10 + entry[0]
        y_c2 = y0_2-10 + entry[1]
        z_c2 = z0_2-10 + entry[2]
        delta_x = (x_c1 - x_c2)**2
        delta_y = (y_c1 - y_c2)**2
        delta_z = (z_c1 - z_c2)**2
        length = np.sqrt(delta_x+delta_y+delta_z)
        if length > 3:
            coords1[n] = x_c1, y_c1, z_c1
            coords2[n] = x_c2, y_c2, z_c2
            # xs[n] = x0_1, x0_2
            # ys[n] = y0_1, y0_2
            # zs[n] = z0_1, z0_2
            # sigma_xs = sigx1, sigx2
            # sigma_ys = sigy1, sigy2
            # sigma_zs = sigz1, sigz2
            gauss_fit = get_3dgauss_mixture((x,y,z),*fit).reshape(area_shape)
            list_of_fitted.append(get_3dgauss_mixture((x,y,z),*fit).reshape(area_shape))
            if plot == True:
                if n % 2 == 0:
                    fig, axs = plt.subplots(2, 3)
                    axs = axs.ravel()

                    # Z-Y slice at fixed X=10
                    axs[0].imshow(selected_area[:, :, 10], origin='lower')
                    axs[0].plot(y0_1, z0_1, 'rx')  # flipped from (z0_1, y0_1)
                    axs[0].plot(y0_2, z0_2, 'bx')
                    axs[0].set_title('ZY slice (X=10)')

                    # Z-X slice at fixed Y=10
                    axs[1].imshow(selected_area[:, 10, :], origin='lower')
                    axs[1].plot(x0_1, z0_1, 'rx')  # flipped from (z0_1, x0_1)
                    axs[1].plot(x0_2, z0_2, 'bx')
                    axs[1].set_title('ZX slice (Y=10)')

                    # Y-X slice at fixed Z=10
                    axs[2].imshow(selected_area[10, :, :], origin='lower')
                    axs[2].plot(x0_1, y0_1, 'rx')  # flipped from (y0_1, x0_1)
                    axs[2].plot(x0_2, y0_2, 'bx')
                    axs[2].set_title('YX slice (Z=10)')

                    # Same for Gaussian fit
                    axs[3].imshow(gauss_fit[:, :, 10], cmap='hot', origin='lower')
                    axs[3].plot(y0_1, z0_1, 'rx')
                    axs[3].plot(y0_2, z0_2, 'bx')
                    axs[3].set_title('Fit ZY (X=10)')

                    axs[4].imshow(gauss_fit[:, 10, :], cmap='hot', origin='lower')
                    axs[4].plot(x0_1, z0_1, 'rx')
                    axs[4].plot(x0_2, z0_2, 'bx')
                    axs[4].set_title('Fit ZX (Y=10)')

                    axs[5].imshow(gauss_fit[10, :, :], cmap='hot', origin='lower')
                    axs[5].plot(x0_1, y0_1, 'rx')
                    axs[5].plot(x0_2, y0_2, 'bx')
                    axs[5].set_title('Fit YX (Z=10)')

                    plt.tight_layout()
                    # plt.savefig(f"/groups/liza/Pictures/05_13_2025/gauss_mix_fixed_20x20_section{n}")
            
        full_coords = np.concatenate((coords1, coords2), axis=0)
        holder = np.where((full_coords == [0, 0, 0]).all(axis=1))[0]
        full_coords = np.delete(full_coords, holder, axis = 0)
    return full_coords

def get_3dgauss_mixture(xyz, x0_1, y0_1, z0_1, x0_2, y0_2, z0_2, pi, amp, sigma_x, sigma_y, sigma_z, background): 
    x, y, z = xyz
    x, y, z = xyz
    ## going ahead and fixing the amplitude and sigmas so they can't move
    g1 = np.exp(-(((x - x0_1)**2) / (2 * sigma_x**2) +
                  ((y - y0_1)**2) / (2 * sigma_y**2) +
                  ((z - z0_1)**2) / (2 * sigma_z**2)))
    
    g2 = np.exp(-(((x - x0_2)**2) / (2 * sigma_x**2) +
                  ((y - y0_2)**2) / (2 * sigma_y**2) +
                  ((z - z0_2)**2) / (2 * sigma_z**2)))

    g_3d = (amp * pi * g1) + (amp * (1 - pi) * g2) + background 
    return g_3d.ravel()

## saving original for reference
def get_3dgauss_mixture_take_1(xyz, x0_1, y0_1, z0_1, x0_2, y0_2, z0_2, pi, amp_1, amp_2, sigma_x1, sigma_y1, sigma_z1, sigma_x2, sigma_y2, sigma_z2, background): 
    x, y, z = xyz
    x, y, z = xyz

    g1 = np.exp(-(((x - x0_1)**2) / (2 * sigma_x1**2) +
                  ((y - y0_1)**2) / (2 * sigma_y1**2) +
                  ((z - z0_1)**2) / (2 * sigma_z1**2)))
    
    g2 = np.exp(-(((x - x0_2)**2) / (2 * sigma_x2**2) +
                  ((y - y0_2)**2) / (2 * sigma_y2**2) +
                  ((z - z0_2)**2) / (2 * sigma_z2**2)))

    g_3d = (amp_1 * pi * g1) + (amp_2 * (1 - pi) * g2) + background ## Add another amp! amp1 * pi * G1 + amp2 * (1 - pi) * G2
    return g_3d.ravel()




if __name__ == '__main__':
    # # TESTING ##
    # # tomo_3 = "/nrs/liza/aretomoe3_rm_patch_gauss_circle_alpha_two_itr/20231017_EGmilled24-2_68_Vol.mrc"
    # # tomo_4 = "/nrs/liza/aretomoe3_tes/20231017_EGmilled24-2_68_Vol.mrc"
    # tomo = "/nrs/liza/hoyoung_dimer_tomos/dimer_tomos/dimer_tomos_alphaoffset_only/20250210_HippWaffle_64_Vol.mrc"
    # tomo_2 = "/nrs/liza/hoyoung_dimer_tomos/dimer_tomos/dimer_tomo64_mixgauss_v1/20250210_HippWaffle_64_Vol.mrc"
    # # center = (515,615)
    # # center = (543, 462)
    # center = (460,800) # tomo 64
    # radius = 120

    # tomos = [tomo, tomo_2] #tomo_3, tomo_4]
    # sigma_tomos = []
    # fitts = []

    # for n in range(len(tomos)):
    #     sig, fitt = get_3d_sigmas(tomos[n], center_oi = center, radius=radius, plot = False, cutoff = 95)
    #     sigma_tomos.append(sig)
    #     fitts.append(fitt)
    #     sigmas = np.array(sigma_tomos[n])


    # sig_title = ['x','y','z']
    # tomo_name = ["original", "first itr", "second itr", "patch corr."]#, "third itr"]
    # tomo_name = ["CTF", "No CTF"]
    # tomo_name = ["Original", "One itr"]
    # lengths = []
    
    # fig, axs = plt.subplots(1,3, sharex=True, sharey=True, figsize=(10, 12))
    # axs.ravel()
    # for n in range(len(sigma_tomos)):
    #     sigmas = np.array(sigma_tomos[n])
    #     print(len(sigmas))
    #     lengths.append(len(sigmas))
    #     for i in range(sigmas.shape[1]):
    #         axs[i].set_title(sig_title[i])
    #         parts = axs[i].violinplot(sigmas[:,i], showmedians=False)
    #         violin_color = parts['cbars'].get_edgecolor().flatten()
    #         axs[i].plot([],[], color = violin_color, label = tomo_name[n])
    #         if i == sigmas.shape[1]-1:
    #             axs[i].legend()
    # fig.suptitle("Distribution of Gaussian Standard Deviation by Axis")
    
    # print('end')

    # testing get_mixed_gaussian()
    # tomo_path = "/nrs/liza/Aret3_rm_patch_tomo54_one_itr/20231017_EGmilled24-2_54_Vol.mrc"
    tomo_path = "/nrs/liza/hoyoung_dimer_tomos/dimer_tomos/only_dimers_init/tiltseries4.mrc_Vol.mrc"
    with mrcfile.open(tomo_path) as mrctomo: # get tomo data
        invert_tomo = mrctomo.data * -1 #flip black and white so peak_local_max picks up dark points
        tomo_shape = invert_tomo.shape

    rel_threshold = .6
    border_cutoff = (70,50,50)
    # center_OI = (460,800) # tomo 64
    #center_OI = (515,615)
    center_OI = (400, 600)
    radius_OI = 120

    peaks = peak_local_max(invert_tomo, min_distance=3, threshold_rel = rel_threshold, exclude_border = border_cutoff).tolist() #threshold_rel = .4,

    for entry in peaks: # convert zyx -> xyz
            entry[0], entry[2] = entry[2], entry[0]
            peaks[peaks.index(entry)] = entry

    # OPTIONAL
    if center_OI != None:  # get only the points within a certain area of the tomogram
        peak_coords = [ent for ent in peaks if center_OI[0]+radius_OI > ent[0] and ent[0] >center_OI[0]-radius_OI and center_OI[1]+radius_OI > ent[1] and ent[1] >center_OI[1]-radius_OI]
    else:
        peak_coords = peaks

    test = get_mixed_gaussian(invert_tomo, peak_coords, plot = True)
    print('end')




## testing 3d sigma plots

# tomo = "/nrs/liza/Aret3_rm_patch_tomo54/20231017_EGmilled24-2_54_Vol.mrc" 
# tomo_2 = "/nrs/liza/Aret3_rm_patch_tomo54_one_itr/20231017_EGmilled24-2_54_Vol.mrc"
# tomo_2 = "/nrs/liza/Aret3_rm_patch_tomo54/20231017_EGmilled24-2_54_2ND_Vol.mrc" 
#tomo_3 = "/nrs/liza/Aret3_rm_patch_tomo54_two_itr/20231017_EGmilled24-2_54_Vol.mrc"
#

# tomo = "/nrs/liza/aretomoe3_rm_patch/20231017_EGmilled24-2_68_Vol.mrc"
# tomo = "/scratch/pompeii/elferich/gouaux_tomo/tomograms/15F1_tomograms/TOP_TOMOS/20231017_EGmilled24-2_68/best_alignment/20231017_EGmilled24-2_68_full_rec_BP_3DCTF_BIN4.mrc"
# tomo_2 = "/nrs/liza/aretomoe3_rm_patch_gausses_alpha_circle/20231017_EGmilled24-2_68_Vol.mrc"
# # tomo_3 = "/nrs/liza/aretomoe3_rm_patch_gauss_circle_alpha_two_itr/20231017_EGmilled24-2_68_Vol.mrc"
# # tomo_4 = "/nrs/liza/aretomoe3_tes/20231017_EGmilled24-2_68_Vol.mrc"

# # center = (515,615)
# center = (543, 462)
# radius = 120

# tomos = [tomo, tomo_2] #tomo_3, tomo_4]
# sigma_tomos = []
# fitts = []

# for n in range(len(tomos)):
#     sig, fitt = get_3d_sigmas(tomos[n], center_oi = center, radius=radius, plot = False, cutoff = 95)
#     sigma_tomos.append(sig)
#     fitts.append(fitt)
#     sigmas = np.array(sigma_tomos[n])


# sig_title = ['x','y','z']
# tomo_name = ["original", "first itr", "second itr", "patch corr."]#, "third itr"]
# tomo_name = ["CTF", "No CTF"]
# tomo_name = ["IMOD", "Mine"]
# lengths = []
 
# fig, axs = plt.subplots(1,3, sharex=True, sharey=True, figsize=(10, 12))
# axs.ravel()
# for n in range(len(sigma_tomos)):
#     sigmas = np.array(sigma_tomos[n])
#     print(len(sigmas))
#     lengths.append(len(sigmas))
#     for i in range(sigmas.shape[1]):
#         axs[i].set_title(sig_title[i])
#         parts = axs[i].violinplot(sigmas[:,i], showmedians=False)
#         violin_color = parts['cbars'].get_edgecolor().flatten()
#         axs[i].plot([],[], color = violin_color, label = tomo_name[n])
#         if i == sigmas.shape[1]-1:
#             axs[i].legend()
# fig.suptitle("Distribution of Gaussian Standard Deviation by Axis")

# #plt.figure()
# #plt.plot(lengths, 'o')

# print('end')
# ### testing how each iteration changes
# # tomo = "/nrs/liza/aretomoe3_rm_patch/20231017_EGmilled24-2_68_Vol.mrc"
# # tomo_2 = "/nrs/liza/aretomoe3_rm_patch_gausses_alpha_circle_2/20231017_EGmilled24-2_68_Vol.mrc"
# # tomo_3 = "/nrs/liza/aretomoe3_rm_patch_gauss_circle_alpha_two_itr/20231017_EGmilled24-2_68_Vol.mrc"
# # #tomo_3 = "/nrs/liza/aretomoe3_rm_patch_gauss_alpha_offset/20231017_EGmilled24-2_68_Vol.mrc"
# # tomo_4 = "/nrs/liza/aretomoe3_rm_patch_gauss_circle_alpha_three_itr/20231017_EGmilled24-2_68_Vol.mrc"

# # center = (543, 462)
# # radius = 120

# # tomos = [tomo, tomo_2, tomo_3, tomo_4]
# # cutoffs = [100, 95, 90, 80]
# # sigma_tomos = []
# # fitts = []

# # sig_title = ['x','y','z']
# # fig, axs = plt.subplots(1,3, sharex = True, sharey = True)

# # for n in range(len(cutoffs)):
# #     sig, fitt = get_3d_sigmas(tomo, center_oi = center, radius=radius, plot = False, cutoff = cutoffs[n])
# #     sigma_tomos.append(sig)
# #     fitts.append(fitt)
# #     sigmas = np.array(sigma_tomos[n])
# #     for i in range(sigmas.shape[1]):
# #         axs[i].set_title(sig_title[i])
# #         parts = axs[i].violinplot(sigmas[:,i], showmedians=False)
# #         violin_color = parts['bodies'][0].get_facecolor().flatten()
# #         axs[i].plot([],[], color = violin_color, label = n)
# #         axs[i].legend()

    
# for n in range(len(tomos)):
#     sig, fitt = get_3d_sigmas(tomos[n], center_oi = center, radius=radius, plot = False, cutoff = 95)
#     sigma_tomos.append(sig)
#     fitts.append(fitt)
#     sigmas = np.array(sigma_tomos[n])



# sig_title = ['x','y','z']
# tomo_name = ["original", "first itr", "second itr", "third itr"]
# fig, axs = plt.subplots(1,3, sharex=True, sharey=True) 
# axs.ravel()
# for n in range(len(sigma_tomos)):
#     sigmas = np.array(sigma_tomos[n])
#     for i in range(sigmas.shape[1]):
#         axs[i].set_title(sig_title[i])
#         parts = axs[i].violinplot(sigmas[:,i], showmedians=False)
#         violin_color = parts['cbars'].get_edgecolor().flatten()
#         axs[i].plot([],[], color = violin_color, label = tomo_name[n])
#         if i == sigmas.shape[1]-1:
#             axs[i].legend()



# fig, axs = plt.subplots(1,3)
# for i in range(0,30,10):
#     cropped_phase = np.load("/nrs/liza/gold-aligner/cropped_phases_for_gaussian_testing.npy")
#     guess_shift = np.unravel_index(np.argmax(cropped_phase[i], axis=None), cropped_phase[i].shape)
#     output, gauss = find_2d_gaussian_peak(cropped_phase[i], guess_shift)
    
#     axs[i//10].imshow(cropped_phase[i])
#     axs[i//10].imshow(gauss, cmap='hot', alpha = 0.5)
#     axs[i//10].plot(output[0], output[1], 'rx')


#plt.show()
print('end')

