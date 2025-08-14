import numpy as np
from cryoet_alignment import read, write 
from cryoet_alignment.io.cryoet_data_portal import Alignment
import mrcfile
import matplotlib.pyplot as plt
import os
import re
import math
import subprocess
import glob
from gold_aligner.align_gold import realign_gold
from gold_aligner.fix_alpha_offset import fix_alpha_offset 
from gold_aligner._convert_imod_to_aretomo_aln import imod_to_aretomo, aretomo_to_imod

def create_aretomo_alns(path_to_all_folders, core_path, cmd_path):
    folder_list = os.listdir(path_to_all_folders)
    for i, folder in enumerate(folder_list):
        if folder == '.stfolder' or i != 23: # allows for picking up after an error (use debugger to see which folder it got stuck on)
            continue
        tilt_path = f"{core_path}{folder}/{folder}.mrc"
        aln_path = f"{core_path}{folder}/fiducial_tracking/{folder}"
        if os.path.exists(f"{aln_path}.xf"):
            print(f"Folder '{aln_path}' exists.")
            if os.path.exists(f"{core_path}{folder}/fiducial_tracking/active_zonograms"):
                folder_path = f"{core_path}{folder}/fiducial_tracking"
            else:             
                aln_path = f"{core_path}{folder}/patch_tracking/{folder}"
                folder_path = f"{core_path}{folder}/patch_tracking"
        else:
            aln_path = f"{core_path}{folder}/patch_tracking/{folder}"
            folder_path = f"{core_path}{folder}/patch_tracking"

        output_path = f"/nrs/liza/cathy_tomos/tomos_5f11_init/{folder}_init.aln"


        #convert IMOD to Aretomo3 .aln
        aln = read(aln_path)
        with mrcfile.open(tilt_path) as mrctilt:
            tilt = mrctilt.data
            tilt_shape = tilt.shape
            invert_shape = (tilt_shape[2], tilt_shape[1], tilt_shape[0])

        check = imod_to_aretomo(aln, invert_shape, f"{folder_path}/{folder}.tlt")
        
        write(check, output_path)

        # convert IMOD .tlt to Aretomo3 _TLT.txt
        subprocess.run(f"ln -sfn {folder_path}/{folder}.tlt {core_path}{folder}/{folder}_TLT.txt", shell = True)
        subprocess.run(f"ln -sfn {output_path} {core_path}{folder}/{folder}.aln", shell = True)

        cmd = (
        "ml cuda/cuda-11.3.1 &&"
        f"/nrs/liza/AreTomo3/AreTomo3 -InPrefix {path_to_all_folders}{folder}/20 "
        f"-InSuffix .mrc -OutDir /nrs/liza/cathy_tomos/tomos_5f11_init/ -Cmd 2 -Serial 1 -Wbp 1 -FlipVol 1 "
        "-VolZ 1600 -AtBin 4 -Gpu 0 -Cs 0.01"
        )
        with open(cmd_path, "w") as file:
            file.write(cmd) 

        subprocess.run([
            "gnome-terminal",
            "--wait",
            "--",
            "bash", "-i", "-c", f"{cmd_path}"
        ])
        
        print(folder)
    print('end')

def construct_tomos(command_path, path_to_all_folders):
    folder_list = os.listdir(path_to_all_folders)
    for i, folder in enumerate(folder_list):
    # go file by file, reconstructing tomos
        if folder == '.stfolder' or folder == '20240901_AMmilled13-1_43': # allows for picking up after an error (use debugger to see which folder it got stuck on)
            continue
        
        cmd = (
        "ml cuda/cuda-11.3.1 &&"
        f"/nrs/liza/AreTomo3/AreTomo3 -InPrefix {path_to_all_folders}{folder}/20 "
        "-InSuffix .mrc -OutDir /nrs/liza/cathy_tomos/top_exclude_init -Cmd 2 -Serial 1 -Wbp 1 -FlipVol 1 "
        "-VolZ 1600 -AtBin 4 -Gpu 0 -Cs 0.01"
        )
        with open(command_path, "w") as file:
            file.write(cmd) 

        subprocess.run([
            "gnome-terminal",
            "--wait",
            "--",
            "bash", "-i", "-c", f"{command_path}"
        ])
        
        print(folder)

def check_active_zonograms(path_to_folders):
    for folder in os.listdir(path_to_folders):
        aln_path = f"{path_to_folders}{folder}/fiducial_tracking/{folder}"

        if os.path.exists(f"{aln_path}.xf"):
            print(f"Folder '{aln_path}' exists.")
            if os.path.exists(f"{path_to_folders}{folder}/fiducial_tracking/active_zonograms"):
                zono_path = f"{path_to_folders}{folder}/fiducial_tracking/active_zonograms"
            else:             
                zono_path = f"{path_to_folders}{folder}/patch_tracking/active_zonograms"
        else:
                zono_path = f"{path_to_folders}{folder}/patch_tracking/active_zonograms"
        
        file = glob.glob(f"{zono_path}/active_zonogram_*.png")
        print(folder)
        subprocess.run(['eog'] + file)

    return

def open_napari(path_to_all_folders, tomo_path):
    folder_list = os.listdir(path_to_all_folders)
    for i, folder in enumerate(folder_list):
        if folder == "20240830_AMmilled16_36" or folder == "20241203_AMmilled23-1_Position_14":
            continue
        subprocess.run(f'napari {tomo_path}/{folder}_Vol.mrc', shell = True)
        print(folder)

def run_align_gold(path_to_all_folders, center_path):
    folder_list = os.listdir(path_to_all_folders)
    folder_exclude = ['20240523_HippWaffle_125', '20240830_AMmilled16_36', '20240822_HippWaffle_35', '20240822_HippWaffle_142', '20240822_HippWaffle_38', '20241203_AMmilled23-1_Position_14']
    centers = np.genfromtxt(center_path, delimiter=',', skip_header = 1)
    # second_itr_list = np.genfromtxt(list_path, delimiter=',', skip_header = 1, dtype=str)
    tuple_centers = [(entry[2],entry[1]) for entry in centers]
    for i, folder in enumerate(folder_list):
        if folder == ".stfolder" or folder in folder_exclude:
            continue
        print(folder)
        tomo = f"/nrs/liza/cathy_tomos/tomos_5f11_oneItr/{folder}_Vol.mrc"
        tilt = f"/nrs/liza/cathy_tomos/5f11_top_exclude/{folder}/{folder}.mrc"
        aln = f"/nrs/liza/cathy_tomos/5f11_top_exclude/{folder}/{folder}.aln"
        output_aln = f"/nrs/liza/cathy_tomos/tomos_5f11_secItr/{folder}_secItr.aln"

        realign_gold(tomo, tilt, aln, output_aln, bin = 4, rel_threshold= 0.5, center_OI = tuple_centers[i], radius_OI=120, tomo_au_model= True)

        subprocess.run(f"ln -sfn {output_aln} {path_to_all_folders}{folder}/{folder}.aln", shell = True)

        cmd = (
        "ml cuda/cuda-11.3.1 &&"
        f"/nrs/liza/AreTomo3/AreTomo3 -InPrefix {path_to_all_folders}{folder}/20 "
        f"-InSuffix .mrc -OutDir /nrs/liza/cathy_tomos/tomos_5f11_secItr/ -Cmd 2 -Serial 1 -Wbp 1 -FlipVol 1 "
        "-VolZ 1600 -AtBin 4 -Gpu 0 -Cs 0.01"
        )
        with open(cmd_path, "w") as file:
            file.write(cmd) 

        subprocess.run([
            "gnome-terminal",
            "--wait",
            "--",
            "bash", "-i", "-c", f"{cmd_path}"
        ])
        
        print(folder)

if __name__ == "__main__":
    # rerunning 20241030_AMmilled12-1_15 to select other patch
    tomo_AMmilled = f"/nrs/liza/cathy_tomos/20241030_AMmilled12-1_15_otherPatch/20241030_AMmilled12-1_15_Vol.mrc"
    tilt_AMmilled = f"/nrs/elferich/20241030_AMmilled12-1_15/20241030_AMmilled12-1_15.mrc"
    aln_AMmilled = f"/nrs/liza/cathy_tomos/20241030_AMmilled12-1_15_otherPatch/20241030_AMmilled12-1_15_otherPatch.aln"
    output_aln_AMmilled = f"/nrs/liza/cathy_tomos/20241030_AMmilled12-1_15_otherPatch/20241030_AMmilled12-1_15_otherPatch_secItr.aln"

    realign_gold(tomo_AMmilled, tilt_AMmilled, aln_AMmilled, output_aln_AMmilled, bin = 4, rel_threshold= 0.6, center_OI = (646, 680), radius_OI=120, tomo_au_model= True)


    ############
    path_5f11 = '/nrs/liza/cathy_tomos/5f11_top_exclude/'
    cmd_path = "/nrs/liza/cathy_tomos/cmd_exclude_tomo_init"


    path_to_all_folders = "/nrs/liza/cathy_tomos/top_exclude/"
    center_path = '/nrs/liza/cathy_tomos/tomos_5f11_oneItr/5f11_tomo_centers.csv'
    # center_path = '/nrs/liza/cathy_tomos/top_exclude_oneItr/15f1_tomo_centers.csv'
    
    # run_align_gold(path_to_all_folders, center_path)
    run_align_gold(path_5f11, center_path)
    open_napari(path_5f11, output_path)
    create_aretomo_alns(path_5f11, path_5f11, cmd_path)
    
    
    ############
    path_to_all_folders = "/nrs/liza/cathy_tomos/top_exclude/"
    core_path = "/nrs/liza/cathy_tomos/top_exclude/"
    
    cmd_path = "/nrs/liza/cathy_tomos/cmd_exclude_tomo_init"

    open_napari(path_to_all_folders)

    check_active_zonograms('/nrs/liza/cathy_tomos/top_exclude/')
    construct_tomos(cmd_path, path_to_all_folders)
    create_aretomo_alns(path_to_all_folders, core_path)