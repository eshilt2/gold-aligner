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


## working on clusters
def find_clusters(invert_tomo, rel_threshold, border_cutoff):

    
    peaks = peak_local_max(invert_tomo, min_distance=3, threshold_rel = rel_threshold, exclude_border = border_cutoff).tolist() #threshold_rel = .4,

    for entry in peaks: # convert zyx -> xyz
            entry[0], entry[2] = entry[2], entry[0]
            peaks[peaks.index(entry)] = entry

    peaks = np.array(peaks)
    model = HDBSCAN(min_cluster_size=10) 
    model.fit_predict(peaks)
    pred = model.fit_predict(peaks)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d') 
    ax.scatter(peaks[:,0], peaks[:,1], peaks[:,2], c=model.labels_)
    ax.view_init(azim=200)
    print('end')
    return pred, peaks

if __name__ == "__main__":
    # tomo_path = "/nrs/liza/hoyoung_dimer_tomos/dimer_tomos/dimer_tomo64_mixgauss_fixed_distance_v3/20250210_HippWaffle_64_Vol.mrc"
    # tomo_path = "/nrs/liza/Aret3_rm_patch_tomo54_one_itr/20231017_EGmilled24-2_54_Vol.mrc"
    master_path = "/nrs/liza/cathy_tomos/15f1_top_topop/"
    folder_list = os.listdir(master_path)
    for i, folder in enumerate(folder_list):
        tomo_path = f"/nrs/liza/cathy_tomos/tomos_init_rerun/{folder}_Vol.mrc"
        if i <= 49 or folder == ".stfolder":
             continue
        with mrcfile.open(tomo_path) as mrctomo: # get tomo data
            invert_tomo = mrctomo.data * -1 #flip black and white so peak_local_max picks up dark points
            tomo_shape = invert_tomo.shape

        pred, peaks = find_clusters(invert_tomo, .3, (70,50,50))
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
        plt.savefig(f'/groups/liza/Pictures/05_22_2025/{folder}', dpi=300)
        plt.close(fig)
print('end')


