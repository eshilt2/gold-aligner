"""Everything to do with Cluster Selection"""

import mrcfile
import numpy as np
#import gold_aligner.align_gold as ag
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import OPTICS
from sklearn.cluster import DBSCAN
from sklearn.cluster import HDBSCAN
from skimage.feature import peak_local_max
from cryoet_alignment import read
from cryoet_alignment import write
import os
import subprocess




## working on clusters
def find_clusters(invert_tomo, rel_threshold, border_cutoff, plot = False):

    
    peaks = peak_local_max(invert_tomo, min_distance=3, threshold_rel = rel_threshold, exclude_border = border_cutoff).tolist() #threshold_rel = .4,

    for entry in peaks: # convert zyx -> xyz
            entry[0], entry[2] = entry[2], entry[0]
            peaks[peaks.index(entry)] = entry

    peaks = np.array(peaks)
    model = HDBSCAN(min_cluster_size=10)
     
    try:
        pred = model.fit_predict(peaks)
    except ValueError as e:
        print(f"HDBSCAN error: {e} — skipping {rel_threshold}.")
        return None, None
    # pred = model.fit_predict(peaks)
    if plot == True:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d') 
        ax.scatter(peaks[:,0], peaks[:,1], peaks[:,2], c=model.labels_)
        ax.view_init(azim=200)
        print('end')
        return pred, peaks

def cluster_points(peak_coords, img_save_path, plot = False):
    if type(peak_coords) != 'numpy.ndarray':
        peak_coords = np.array(peak_coords)

    model = HDBSCAN(min_cluster_size=10, max_cluster_size = 75)
    try:
        pred = model.fit_predict(peak_coords)
        if plot == True:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d') 
            ax.scatter(peak_coords[:,0], peak_coords[:,1], peak_coords[:,2], c=model.labels_)
            ax.view_init(azim=200)
            plt.savefig(img_save_path, dpi = 300)
            print('end')
    except ValueError:
        print(len(peak_coords))
        if plot == True:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d') 
            ax.scatter(peak_coords[:,0], peak_coords[:,1], peak_coords[:,2])
            ax.view_init(azim=200)
            plt.savefig(img_save_path, dpi = 300)
            print('end')
        pred = None
    
    return pred

if __name__ == "__main__":
    # tomo_path = "/nrs/liza/hoyoung_dimer_tomos/dimer_tomos/dimer_tomo64_mixgauss_fixed_distance_v3/20250210_HippWaffle_64_Vol.mrc"
    # tomo_path = "/nrs/liza/Aret3_rm_patch_tomo54_one_itr/20231017_EGmilled24-2_54_Vol.mrc"

    
    master_path = "/nrs/liza/cathy_tomos/15f1_top_topop/"
    folder_list = os.listdir(master_path)
    for i, folder in enumerate(folder_list):
        tomo_path = f"/nrs/liza/cathy_tomos/tomos_init_rerun/{folder}_Vol.mrc"
        valid_folders = {
        "20240209_waffleHip_71"#,
        # "20231017_HippAu_132"#,
        # "20240111_WaffleHipp_150"
        }
        if folder not in valid_folders or folder == ".stfolder":
             continue
        print('found folder')
        with mrcfile.open(tomo_path) as mrctomo: # get tomo data
            invert_tomo = mrctomo.data * -1 #flip black and white so peak_local_max picks up dark points
            tomo_min = np.min(invert_tomo)
            tomo_max = np.max(invert_tomo)

        print('loaded tomo')
        if os.path.exists(f'/groups/liza/Pictures/05_28_2025/{folder}_{tomo_min:.6f}_{tomo_max:.6f}') == False:
            subprocess.run(f"mkdir /groups/liza/Pictures/05_28_2025/{folder}_{tomo_min:.6f}-{tomo_max:.6f}", shell = True)

        for threshold in [.3,.4,.5,.6,.7]:

            pred, peaks = find_clusters(invert_tomo, threshold, (70,50,50))
            if type(pred) == np.ndarray:

                fig = plt.figure(figsize=(10,10))
                ax = fig.add_subplot(projection='3d') 
                for label in np.unique(pred):
                    # if label == -1:
                        #  continue
                    mask = pred == label
                    label_name = f"Cluster {label}" if label != -1 else "Noise"
                    ax.scatter(
                        peaks[mask, 0], peaks[mask, 1], peaks[mask, 2],
                        label=label_name
                    )
                # ax.scatter(peaks[:,0], peaks[:,1], peaks[:,2], c=pred)
                plt.legend(loc='best')
                ax.view_init(azim=200)
                plt.savefig(f'/groups/liza/Pictures/05_28_2025/{folder}/{folder}_{threshold}.png', dpi=300)
                plt.close(fig)
    print('end')


