""" Selects gold nanoparticles in tomogram and returns coordinates"""


from skimage.feature import peak_local_max
from gold_aligner._fit_gaussian import find_3d_gaussian_peaks, get_mixed_gaussian
from gold_aligner._make_IMOD_model import make_imod_model
import os
import subprocess
import mrcfile
import numpy as np
# from gold_aligner._find_clusters import *

def select_aunps(inverted_tomo,         # array          : tomogram data *-1 to make gold nps positive peaks
                 rel_threshold,         # int            : sets threshold for selection local peaks. max(peak)*threshold.   
                 border_cutoff,         # (int, int, int): will crop selection area during peak picking (z, y, x)
                 center_OI,             # (int,int)      : will select gold only within sphere centered at (x,y)
                 radius_OI,             # int            : will select gold only within sphere centered at (x,y) with radius r         
                 debugger = False       # bool              
                 ):
    if debugger == False:
        peaks = peak_local_max(inverted_tomo, min_distance=3, threshold_rel = rel_threshold, exclude_border = border_cutoff).tolist() #threshold_rel = .4,
        pred = None
        for entry in peaks: # convert zyx -> xyz
                entry[0], entry[2] = entry[2], entry[0]
                peaks[peaks.index(entry)] = entry

        # OPTIONAL
        if center_OI != None:  # get only the points within a certain area of the tomogram
            peaks = [ent for ent in peaks if center_OI[0]+radius_OI > ent[0] and ent[0] >center_OI[0]-radius_OI and center_OI[1]+radius_OI > ent[1] and ent[1] >center_OI[1]-radius_OI]
        else:
            peak_coords = peaks
        #
    else:
        peaks = peak_local_max(inverted_tomo, min_distance=3, num_peaks=30, threshold_rel = rel_threshold, exclude_border = border_cutoff).tolist() #threshold_rel = .4,
        pred = None
        if center_OI != None:  # get only the points within a certain area of the tomogram
            peaks = [ent for ent in peaks if center_OI[0]+radius_OI > ent[0] and ent[0] >center_OI[0]-radius_OI and center_OI[1]+radius_OI > ent[1] and ent[1] >center_OI[1]-radius_OI]
        else:
            peak_coords = peaks
        for entry in peaks: # convert zyx -> xyz
                entry[0], entry[2] = entry[2], entry[0]
                peaks[peaks.index(entry)] = entry

        # OPTIONAL
        if center_OI != None:  # get only the points within a certain area of the tomogram
            peaks = [ent for ent in peaks if center_OI[0]+radius_OI > ent[0] and ent[0] >center_OI[0]-radius_OI and center_OI[1]+radius_OI > ent[1] and ent[1] >center_OI[1]-radius_OI]
        else:
            peak_coords = peaks
        # pred, peaks = find_clusters(inverted_tomo, rel_threshold, border_cutoff=(0,50,50))

        # for entry in peaks: # convert zyx -> xyz
        #     entry[0], entry[2] = entry[2], entry[0]
        #     peaks[peaks.index(entry)] = entry

    return pred, peaks


def select_aunps_rect_region(inverted_tomo,         # array          : tomogram data *-1 to make gold nps positive peaks
                 rel_threshold,         # int            : sets threshold for selection local peaks. max(peak)*threshold.   
                 border_cutoff,         # (int, int, int): will crop selection area during peak picking (z, y, x)
                 top_xy,                # (int,int)           : Top corner(x,y) 
                 bottom_xy             # (int,int)            : Bottom corner(x,y)         
        
                 ):

    peaks = peak_local_max(inverted_tomo, min_distance=3, threshold_rel = rel_threshold, exclude_border = border_cutoff).tolist() #threshold_rel = .4,
    pred = None
    for entry in peaks: # convert zyx -> xyz
            entry[0], entry[2] = entry[2], entry[0]
            peaks[peaks.index(entry)] = entry

    # OPTIONAL
    if top_xy != None:  # get only the points within a certain area of the tomogram
        peak_coords = [ent for ent in peaks if top_xy[0] < ent[0] and bottom_xy[0] > ent[0] and top_xy[1] < ent[1] and bottom_xy[1] > ent[1]]
    else:
        peak_coords = peaks


    return peak_coords


### IN TESTING
def select_aunps_sectioned(invert_tomo, rel_threshold, border_cutoff, sectioned_area):
        peaks = peak_local_max(invert_tomo, min_distance=3, threshold_rel = rel_threshold, exclude_border = border_cutoff, labels=sectioned_area).tolist() #threshold_rel = .4,
        for entry in peaks: # convert zyx -> xyz
                entry[0], entry[2] = entry[2], entry[0]
                peaks[peaks.index(entry)] = entry

        peak_coords = peaks
        return peak_coords

if __name__ == "__main__":
    master_path = "/nrs/liza/cathy_tomos/15f1_top_topop/"
    folder_list = os.listdir(master_path)
    for i, folder in enumerate(folder_list):
        tomo_path = f"/nrs/liza/cathy_tomos/tomos_init_rerun/{folder}_Vol.mrc"
        valid_folders = {
        "20240209_waffleHip_71"#,
        # "20231017_HippAu_132"#,
        # "20240111_WaffleHipp_150"
        }
        if i > 12 or folder == ".stfolder":
             continue
        print('found folder')
        with mrcfile.open(tomo_path) as mrctomo: # get tomo data
            invert_tomo = mrctomo.data * -1 #flip black and white so peak_local_max picks up dark points
            tomo_min = np.min(invert_tomo)
            tomo_max = np.max(invert_tomo)

        print('loaded tomo')

        for threshold in [.2, .25, .35]:
            _, peaks = select_aunps(invert_tomo, threshold, (70,50,50), None, None)
            make_imod_model(np.array(peaks), "", custom_name=f"/nrs/liza/cathy_tomos/test_reconstruction/test_thresholds/{folder}_{threshold}")
            print('end')
        
