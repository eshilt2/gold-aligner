import os
import pandas as pd
from gold_aligner._make_IMOD_model import make_IMOD_model_UPDATED
import numpy as np

for num in range(4, 9):
    if num == 5:
        continue
    with open(f"/nrs/liza/cathy_tomos/thin_ice_mono_dimer_only/20240830_5F11Fab-AuNPdimer_HYgrid3_coords/20240830_5F11Fab-AuNPdimer_HYgrid3_{num}_full_rec_BP_flat.coords") as f:
        mylist = f.read().splitlines()
        listing = [i.split() for i in mylist]
        int_list = [[int(y) for y in x] for x in listing]
        mono_list = [n for n in int_list if n[0] == 1]
        dimer_list = [n for n in int_list if n[0] == 2]
        array = np.array(mono_list)
        darray = np.array(dimer_list)
        m_picks = np.array(array[:,1:4]).astype(int)
        d_picks = np.array(darray[:,1:4]).astype(int)
        make_IMOD_model_UPDATED(m_picks, f'/nrs/liza/cathy_tomos/thin_ice_mono_dimer_only/20240830_5F11Fab-AuNPdimer_HYgrid3_{num}_m')
        make_IMOD_model_UPDATED(d_picks, f'/nrs/liza/cathy_tomos/thin_ice_mono_dimer_only/20240830_5F11Fab-AuNPdimer_HYgrid3_{num}_d')