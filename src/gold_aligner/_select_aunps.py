""" Selects gold nanoparticles in tomogram and returns coordinates"""


from skimage.feature import peak_local_max
from gold_aligner.fit_gaussian import find_3d_gaussian_peaks

def select_aunps(inverted_tomo,         # array          : tomogram data *-1 to make gold nps positive peaks
                 rel_threshold,         # int            : sets threshold for selection local peaks. max(peak)*threshold.   
                 border_cutoff,         # (int, int, int): will crop selection area during peak picking (z, y, x)
                 center_OI,             # (int,int)      : will select gold only within sphere centered at (x,y)
                 radius_OI              # int            : will select gold only within sphere centered at (x,y) with radius r
                 ):
    peaks = peak_local_max(inverted_tomo, min_distance=3, threshold_rel = rel_threshold, exclude_border = border_cutoff).tolist() #threshold_rel = .4,

    for entry in peaks: # convert zyx -> xyz
            entry[0], entry[2] = entry[2], entry[0]
            peaks[peaks.index(entry)] = entry

    # OPTIONAL
    if center_OI != None:  # get only the points within a certain area of the tomogram
        peak_coords = [ent for ent in peaks if center_OI[0]+radius_OI > ent[0] and ent[0] >center_OI[0]-radius_OI and center_OI[1]+radius_OI > ent[1] and ent[1] >center_OI[1]-radius_OI]
    else:
        peak_coords = peaks
    #

    peak_coords_OI, _, _ = find_3d_gaussian_peaks(inverted_tomo, peak_coords)

    return peak_coords_OI
