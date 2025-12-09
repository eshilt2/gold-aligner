import numpy as np
from cryoet_alignment import read, write 
from cryoet_alignment.io.cryoet_data_portal import Alignment
import mrcfile
import matplotlib.pyplot as plt
import os
import re
import math
import subprocess
from gold_aligner.align_gold import realign_gold
from gold_aligner.fix_alpha_offset import fix_alpha_offset 
from gold_aligner._convert_imod_to_aretomo_aln import imod_to_aretomo, aretomo_to_imod


def aretomo_alns_init(path_to_all_folders):
    folder_list = os.listdir(path_to_all_folders)
    for i, folder in enumerate(folder_list):
        if folder == '.stfolder' or folder == '20240901_AMmilled13-1_43' or i <= 49: # allows for picking up after an error (use debugger to see which folder it got stuck on)
            continue
        tilt_path = f"/nrs/liza/cathy_tomos/15f1_top_topop/{folder}/{folder}.mrc"
        aln_path = f"/nrs/liza/cathy_tomos/15f1_top_topop/{folder}/fiducial_tracking/{folder}"
        if os.path.exists(f"{aln_path}.xf"):
            print(f"Folder '{aln_path}' exists.")
            folder_path = f"/nrs/liza/cathy_tomos/15f1_top_topop/{folder}/fiducial_tracking"
            tilt_com_path = f"/nrs/liza/cathy_tomos/15f1_top_topop/{folder}/fiducial_tracking/tilt.com"
        else:
            aln_path = f"/nrs/liza/cathy_tomos/15f1_top_topop/{folder}/patch_tracking/{folder}"
            folder_path = f"/nrs/liza/cathy_tomos/15f1_top_topop/{folder}/patch_tracking"
            tilt_com_path = f"/nrs/liza/cathy_tomos/15f1_top_topop/{folder}/patch_tracking/tilt.com"

        output_path = f"/nrs/liza/cathy_tomos/tomos_init_rerun/{folder}_rerun.aln"

        #convert IMOD to Aretomo3 .aln
        aln = read(aln_path)
        with mrcfile.open(tilt_path) as mrctilt:
            tilt = mrctilt.data
            tilt_shape = tilt.shape
            invert_shape = (tilt_shape[2], tilt_shape[1], tilt_shape[0])

        check = imod_to_aretomo(aln, invert_shape, tilt_com_path)

        write(check, output_path)

        subprocess.run(f"ln -sfn {output_path} /nrs/liza/cathy_tomos/15f1_top_topop/{folder}/{folder}.aln", shell = True)


def construct_tomos(command_path, path_to_all_folders):
    folder_list = os.listdir(path_to_all_folders)
    for i, folder in enumerate(folder_list):
    # go file by file, reconstructing tomos
        if folder == '.stfolder' or folder == '20240901_AMmilled13-1_43': # allows for picking up after an error (use debugger to see which folder it got stuck on)
            continue
        
        cmd = f'ml cuda/cuda-11.3.1\n/nrs/liza/AreTomo3/AreTomo3 -InPrefix /nrs/liza/cathy_tomos/15f1_top_topop/{folder}/20 -InSuffix .mrc -OutDir ./tomos_init -Cmd 2 -Serial 1 -Wbp 1 -FlipVol 1 -VolZ 1600 -AtBin 4.85 4.85 4.85 -Gpu 0 -Cs 0.01'
        cmd = (
        "ml cuda/cuda-11.3.1 &&"
        f"/nrs/liza/AreTomo3/AreTomo3 -InPrefix /nrs/liza/cathy_tomos/15f1_top_topop/{folder}/20 "
        "-InSuffix .mrc -OutDir /nrs/liza/cathy_tomos/tomos_init_rerun -Cmd 2 -Serial 1 -Wbp 1 -FlipVol 1 "
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

def view_napari(path_to_all_folders):
    folder_list = os.listdir(path_to_all_folders)
    core = "/nrs/liza/cathy_tomos/tomos_init_rerun"
    for i, folder in enumerate(folder_list):
        if i % 3 != 0:
            continue        
        subprocess.run(f'napari {core}/{folder_list[i]}_Vol.mrc {core}/{folder_list[i+1]}_Vol.mrc {core}/{folder_list[i+2]}_Vol.mrc', shell = True)
        user_input = input("Press Enter to continue")

def realign_tomos(path_to_all_folders, center_path):
    folder_list = os.listdir(path_to_all_folders)
    centers = np.genfromtxt(center_path, delimiter=',', skip_header = 1)
    tuple_centers = [(entry[2],entry[1]) for entry in centers]
    for i, folder in enumerate(folder_list):
        if folder == ".stfolder" or folder == "20240901_AMmilled13-1_43" or i <= 31:
            continue
        # if i < 41: # used when things break to get back to certain tomogram
        #     continue
        print(folder)
        tomo = f"/nrs/liza/cathy_tomos/tomos_init_rerun/{folder}_Vol.mrc"
        tilt = f"/nrs/liza/cathy_tomos/15f1_top_topop/{folder}/{folder}.mrc"
        aln = f"/nrs/liza/cathy_tomos/15f1_top_topop/{folder}/{folder}.aln"
        output_aln = f"/nrs/liza/cathy_tomos/tomos_oneItr_rerun/{folder}_oneItr_rerun.aln"

        saved_cropped_phase, shift = realign_gold(tomo, tilt, aln, output_aln, bin = 4, rel_threshold= 0.3, center_OI = tuple_centers[i], radius_OI=120, tomo_au_model= True, alpha_offset=None)
        fig, axs = plt.subplots(math.ceil(np.sqrt(saved_cropped_phase.shape[0])), math.ceil(np.sqrt(saved_cropped_phase.shape[0])), figsize=(10, 10))
        axs = axs.ravel()
        for i, cross in enumerate(saved_cropped_phase):
            axs[i].imshow(saved_cropped_phase[i])
            axs[i].plot(shift[i][0]+50, shift[i][1]+50, 'rx', markersize=1)
            axs[i].vlines(50,0,99, color = 'white', linestyles='dashed', linewidth=0.25)
            axs[i].hlines(50,0,99, color = 'white', linestyles='dashed', linewidth=0.25)
        
        fig.suptitle(f'{folder}')
        plt.savefig(f"/groups/liza/Pictures/05_21_2025/{folder}_cross_corr.png", dpi=300)
        plt.close(fig)
        print(folder)

def symlink_rerun_aln(path_to_all_folders):
    folder_list = os.listdir(path_to_all_folders)
    for i, folder in enumerate(folder_list):
        if folder == ".stfolder" or folder == "20240901_AMmilled13-1_43":
            continue
        subprocess.run(f"ln -sfn /nrs/liza/cathy_tomos/tomos_oneItr_rerun/{folder}_oneItr_rerun.aln /nrs/liza/cathy_tomos/15f1_top_topop/{folder}/{folder}.aln", shell = True)


def construct_rerun_tomos(path_to_all_folders, command_path):
    folder_list = os.listdir(path_to_all_folders)
    for i, folder in enumerate(folder_list):
    # go file by file, reconstructing tomos
        if folder == ".stfolder" or folder == "20240901_AMmilled13-1_43" or folder == "20240111_WaffleHipp_150": # allows for picking up after an error (use debugger to see which folder it got stuck on)
            continue
        # cmd = f'ml cuda/cuda-11.3.1\n/nrs/liza/AreTomo3/AreTomo3 -InPrefix /nrs/liza/cathy_tomos/15f1_top_topop/{folder}/20 -InSuffix .mrc -OutDir ./tomos_init -Cmd 2 -Serial 1 -Wbp 1 -FlipVol 1 -VolZ 1600 -AtBin 4.85 4.85 4.85 -Gpu 0 -Cs 0.01'
        cmd = (
        "ml cuda/cuda-11.3.1 &&"
        f"/nrs/liza/AreTomo3/AreTomo3 -InPrefix /nrs/liza/cathy_tomos/15f1_top_topop/{folder}/20 "
        f"-InSuffix .mrc -OutDir /nrs/liza/cathy_tomos/tomos_oneItr_rerun -Cmd 2 -Serial 1 -Wbp 1 -FlipVol 1 "
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

def compare_napari_rerun(path_to_all_folders):
    folder_list = os.listdir(path_to_all_folders)
    core = "/nrs/liza/cathy_tomos/"
    for i, folder in enumerate(folder_list):
        if i <= 4:
            continue    
        subprocess.run(f'napari {core}tomos_oneItr_rerun/{folder}_Vol.mrc {core}tomos_init_rerun/{folder}_Vol.mrc', shell = True)
        user_input = input("Press Enter to continue")        

def convert_to_imod(path_to_all_folders):
    folder_list = os.listdir(path_to_all_folders)
    for i, folder in enumerate(folder_list):
    # go file by file, reconstructing tomos
        if folder == ".stfolder" or folder == "20240901_AMmilled13-1_43" or folder == "20240111_WaffleHipp_150": # allows for picking up after an error (use debugger to see which folder it got stuck on)
            continue
        aln = read(f"/nrs/liza/cathy_tomos/tomos_oneItr_rerun/{folder}_oneItr_rerun.aln")
        tilt_path = f"/nrs/liza/cathy_tomos/15f1_top_topop/{folder}/{folder}.mrc"
        with mrcfile.open(tilt_path) as mrctilt:
            tilt = mrctilt.data
            tilt_shape = tilt.shape
            invert_shape = (tilt_shape[2], tilt_shape[1], tilt_shape[0])
        
        if os.path.exists(f"/nrs/liza/cathy_tomos/tomos_imod_rerun/{folder}"):
            print('exists')
        else:
            subprocess.run(f'mkdir /nrs/liza/cathy_tomos/tomos_imod_rerun/{folder}', shell = True)

        output_path = f"/nrs/liza/cathy_tomos/tomos_imod_rerun/{folder}/{folder}"
        imod_aln = aretomo_to_imod(aln, invert_shape, 2.5)

        write(imod_aln, output_path)


############## original run
def create_aretomo_alns(path_to_all_folders):
    folder_list = os.listdir(path_to_all_folders)
    for i, folder in enumerate(folder_list):
        if folder == '.stfolder': # allows for picking up after an error (use debugger to see which folder it got stuck on)
            continue
        tilt_path = f"/nrs/liza/cathy_tomos/15f1_top_topop/{folder}/{folder}.mrc"
        aln_path = f"/nrs/liza/cathy_tomos/15f1_top_topop/{folder}/fiducial_tracking/{folder}"
        if os.path.exists(f"{aln_path}.xf"):
            print(f"Folder '{aln_path}' exists.")
            folder_path = f"/nrs/liza/cathy_tomos/15f1_top_topop/{folder}/fiducial_tracking"
            tilt_com_path = f"/nrs/liza/cathy_tomos/15f1_top_topop/{folder}/fiducial_tracking/tilt.com"
        else:
            aln_path = f"/nrs/liza/cathy_tomos/15f1_top_topop/{folder}/patch_tracking/{folder}"
            folder_path = f"/nrs/liza/cathy_tomos/15f1_top_topop/{folder}/patch_tracking"
            tilt_com_path = f"/nrs/liza/cathy_tomos/15f1_top_topop/{folder}/patch_tracking/tilt.com"

        output_path = f"/nrs/liza/cathy_tomos/15f1_top_topop/{folder}/{folder}.aln"


        #convert IMOD to Aretomo3 .aln
        aln = read(aln_path)
        with mrcfile.open(tilt_path) as mrctilt:
            tilt = mrctilt.data
            tilt_shape = tilt.shape
            invert_shape = (tilt_shape[2], tilt_shape[1], tilt_shape[0])

        check = imod_to_aretomo(aln, invert_shape)

        write(check, output_path)

        # convert IMOD .tlt to Aretomo3 _TLT.txt
        os.symlink(f"{folder_path}/{folder}.tlt", f"/nrs/liza/cathy_tomos/15f1_top_topop/{folder}/{folder}_TLT.txt")

def reconstruct_tomos(command_path, path_to_all_folders):
    folder_list = os.listdir(path_to_all_folders)
    for i, folder in enumerate(folder_list):
    # go file by file, reconstructing tomos
        if folder == '.stfolder': # allows for picking up after an error (use debugger to see which folder it got stuck on)
            continue
        cmd = f'ml cuda/cuda-11.3.1\n/nrs/liza/AreTomo3/AreTomo3 -InPrefix /nrs/liza/cathy_tomos/15f1_top_topop/{folder}/20 -InSuffix .mrc -OutDir ./tomos_init -Cmd 2 -Serial 1 -Wbp 1 -FlipVol 1 -VolZ 1600 -AtBin 4.85 4.85 4.85 -Gpu 0 -Cs 0.01'
        cmd = (
        "ml cuda/cuda-11.3.1 &&"
        f"/nrs/liza/AreTomo3/AreTomo3 -InPrefix /nrs/liza/cathy_tomos/15f1_top_topop/{folder}/20 "
        "-InSuffix .mrc -OutDir /nrs/liza/cathy_tomos/tomos_init -Cmd 2 -Serial 1 -Wbp 1 -FlipVol 1 "
        "-VolZ 1600 -AtBin 4 4 4 -Gpu 0 -Cs 0.01"
    )
        with open(command_path, "w") as file:
            file.write(cmd) 

        subprocess.run([
            "gnome-terminal",
            "--wait",
            "--",
            "bash", "-i", "-c", f"{command_path}"
        ])

        print('end')

def open_napari(path_to_all_folders):
    folder_list = os.listdir(path_to_all_folders)
    for i, folder in enumerate(folder_list):
        subprocess.run(f'napari /nrs/liza/cathy_tomos/tomos_init/{folder}_Vol.mrc', shell = True)
        user_input = input("Press Enter to continue")

def run_align_gold(path_to_all_folders, center_path):
    folder_list = os.listdir(path_to_all_folders)
    centers = np.genfromtxt(center_path, delimiter=',', skip_header = 1)
    tuple_centers = [(entry[2],entry[1]) for entry in centers]
    for i, folder in enumerate(folder_list):
        if folder == ".stfolder" or folder == "20240901_AMmilled13-1_43" or folder == "20241031_AMmilled13-1_19":
            continue
        # if i < 41: # used when things break to get back to certain tomogram
        #     continue
        print(folder)
        tomo = f"/nrs/liza/cathy_tomos/tomos_init/{folder}_Vol.mrc"
        tilt = f"/nrs/liza/cathy_tomos/15f1_top_topop/{folder}/{folder}.mrc"
        aln = f"/nrs/liza/cathy_tomos/15f1_top_topop/{folder}/{folder}.aln"
        output_aln = f"/nrs/liza/cathy_tomos/tomos_aligned_oneItr/{folder}_oneItr.aln"
        
        realign_gold(tomo, tilt, aln, output_aln, bin = 4, rel_threshold= 0.3, center_OI = tuple_centers[i], radius_OI=120, tomo_au_model= True, alpha_offset=20)

def duplicate_original_aln(path_to_all_folders):
    folder_list = os.listdir(path_to_all_folders)
    for i, folder in enumerate(folder_list):
        og_aln_path = f"/nrs/liza/cathy_tomos/15f1_top_topop/{folder}/{folder}.aln"
        new_aln_path = f"/nrs/liza/cathy_tomos/15f1_top_topop/{folder}/{folder}_original.aln"

        if os.path.exists(new_aln_path):
            continue
        
        subprocess.run(f"cp {og_aln_path} {new_aln_path}", shell = True)

def symlink_new_aln(path_to_all_folders, outDirName, suffix):
    folder_list = os.listdir(path_to_all_folders)
    for i, folder in enumerate(folder_list):
        og_aln_path = f"/nrs/liza/cathy_tomos/15f1_top_topop/{folder}/{folder}.aln"
        new_aln_path = f"/nrs/liza/cathy_tomos/{outDirName}/{folder}{suffix}.aln"
        subprocess.run(f"ln -sfn {new_aln_path} {og_aln_path}", shell = True)

def reconstruct_realigned_tomos(command_path, path_to_all_folders, outDirFolder):
    folder_list = os.listdir(path_to_all_folders)
    for i, folder in enumerate(folder_list):
    # go file by file, reconstructing tomos
        if folder == ".stfolder" or folder == "20240901_AMmilled13-1_43" or folder == "20241031_AMmilled13-1_19": # allows for picking up after an error (use debugger to see which folder it got stuck on)
            continue
        # cmd = f'ml cuda/cuda-11.3.1\n/nrs/liza/AreTomo3/AreTomo3 -InPrefix /nrs/liza/cathy_tomos/15f1_top_topop/{folder}/20 -InSuffix .mrc -OutDir ./tomos_init -Cmd 2 -Serial 1 -Wbp 1 -FlipVol 1 -VolZ 1600 -AtBin 4.85 4.85 4.85 -Gpu 0 -Cs 0.01'
        cmd = (
        "ml cuda/cuda-11.3.1 &&"
        f"/nrs/liza/AreTomo3/AreTomo3 -InPrefix /nrs/liza/cathy_tomos/15f1_top_topop/{folder}/20 "
        f"-InSuffix .mrc -OutDir /nrs/liza/cathy_tomos/{outDirFolder} -Cmd 2 -Serial 1 -Wbp 1 -FlipVol 1 "
        "-VolZ 1600 -AtBin 4 4 4 -Gpu 0 -Cs 0.01"
    )
        with open(command_path, "w") as file:
            file.write(cmd) 

        subprocess.run([
            "gnome-terminal",
            "--wait",
            "--",
            "bash", "-i", "-c", f"{command_path}"
        ])

        print('end')
        
def apply_alphaOffset_aln(path_to_all_folders):
    # get alphas from tilt.com --> pull aretomo aln --> _fix_alpha_offset --> save alignment + symlink --> reconstruct tomos
    folder_list = os.listdir(path_to_all_folders)
    for i, folder in enumerate(folder_list):
        aln_path = f"/nrs/liza/cathy_tomos/15f1_top_topop/{folder}/fiducial_tracking/{folder}"
        if os.path.exists(f"{aln_path}.xf"):
            print(f"Folder '{aln_path}' exists.")
            tilt_com_path = f"/nrs/liza/cathy_tomos/15f1_top_topop/{folder}/fiducial_tracking/tilt.com"
        else:
            aln_path = f"/nrs/liza/cathy_tomos/15f1_top_topop/{folder}/patch_tracking/{folder}"
            tilt_com_path = f"/nrs/liza/cathy_tomos/15f1_top_topop/{folder}/patch_tracking/tilt.com"
        
        if folder == "20240901_AMmilled13-1_43" or folder == ".stfolder":
            continue
        tilt_file = [line for line in open(tilt_com_path)]
        match = re.findall(r'[0-9]+.[0-9]+', tilt_file[27])
        alphaOffset = float(match[0])


        aln_og_path = f"/nrs/liza/cathy_tomos/15f1_top_topop/{folder}/{folder}_original.aln"
        aln = read(aln_og_path)
        new_aln = fix_alpha_offset(aln, alphaOffset)
        new_aln_path = f"/nrs/liza/cathy_tomos/tomos_alphaOffset_init/{folder}_alphaOffset_init.aln"
        
        write(new_aln, new_aln_path)
        print('end')
    
def apply_alphaOffset_my_aln(path_to_all_folders):
    # get alphas from tilt.com --> pull aretomo aln --> _fix_alpha_offset --> save alignment + symlink --> reconstruct tomos
    folder_list = os.listdir(path_to_all_folders)
    for i, folder in enumerate(folder_list):
        aln_path = f"/nrs/liza/cathy_tomos/15f1_top_topop/{folder}/fiducial_tracking/{folder}"
        if os.path.exists(f"{aln_path}.xf"):
            print(f"Folder '{aln_path}' exists.")
            tilt_com_path = f"/nrs/liza/cathy_tomos/15f1_top_topop/{folder}/fiducial_tracking/tilt.com"
        else:
            aln_path = f"/nrs/liza/cathy_tomos/15f1_top_topop/{folder}/patch_tracking/{folder}"
            tilt_com_path = f"/nrs/liza/cathy_tomos/15f1_top_topop/{folder}/patch_tracking/tilt.com"
        
        if folder == "20240901_AMmilled13-1_43" or folder == ".stfolder" or folder == "20241031_AMmilled13-1_19":
            continue
        tilt_file = [line for line in open(tilt_com_path)]
        match = re.findall(r'[0-9]+.[0-9]+', tilt_file[27])
        alphaOffset = float(match[0])


        aln_og_path = f"/nrs/liza/cathy_tomos/tomos_aligned_oneItr/{folder}_oneItr.aln"
        aln = read(aln_og_path)
        new_aln = fix_alpha_offset(aln, alphaOffset)
        new_aln_path = f"/nrs/liza/cathy_tomos/tomos_alphaOffset_oneItr/{folder}_alphaOffset_oneItr.aln"
        
        write(new_aln, new_aln_path)
        print('end')
    
def open_napari_comparison(path_to_all_folders):
    folder_list = os.listdir(path_to_all_folders)
    for i, folder in enumerate(folder_list):
        subprocess.run(f'napari /nrs/liza/cathy_tomos/tomos_alphaOffset_oneItr/{folder}_Vol.mrc /nrs/liza/cathy_tomos/tomos_alphaOffset_init/{folder}_Vol.mrc', shell = True)
        user_input = input("Press Enter to continue")




if __name__ == "__main__":
    path = "/nrs/liza/cathy_tomos/15f1_top_topop/"
    cmd_path = "/nrs/liza/cathy_tomos/cmd_tomo_init"
    cmd_align_path = "/nrs/liza/cathy_tomos/cmd_tomo_align"
    centers = "/nrs/liza/cathy_tomos/tomo_centers.csv"
    
    # test reconstruction
    aln_path = '/nrs/liza/cathy_tomos/tomos_imod_rerun/20240523_HippWaffle_168/20240523_HippWaffle_168'
    tilt_path = '/nrs/liza/cathy_tomos/15f1_top_topop/20240523_HippWaffle_168/20240523_HippWaffle_168.mrc'
    tilt_com = '/nrs/liza/cathy_tomos/tomos_imod_rerun/20240523_HippWaffle_168/tilt.com'
    imod_aln = read(aln_path)
    with mrcfile.open(tilt_path) as mrctilt:
        tilt = mrctilt.data
        tilt_shape = tilt.shape
        invert_shape = (tilt_shape[2], tilt_shape[1], tilt_shape[0])
    

    new_aretomo_aln = imod_to_aretomo(imod_aln, invert_shape, tilt_com)
    write(new_aretomo_aln, '/nrs/liza/cathy_tomos/test_reconstruction/test_imod_recon/220240523_HippWaffle_168_imod_test.aln')
    ## Rerun Workflow


    # 7.    convert_to_imod(path)
    # 6.    compare_napari_rerun(path)
    # 5.    construct_rerun_tomos(path, cmd_align_path)
    # 4.    symlink_rerun_aln(path)
    # 3.    realign_tomos(path, centers)
    # 2.    construct_tomos(cmd_path, path)
    # 1.    aretomo_alns_init(path)




    ## Original Workflow
    #14. open_napari_comparison(path)
    #13. reconstruct_realigned_tomos(cmd_align_path, path, 'tomos_alphaOffset_oneItr')
    #12. symlink_new_aln(path, 'tomos_alphaOffset_oneItr' ,'_alphaOffset_oneItr')
    #11. apply_alphaOffset_my_aln(path)
    #10. reconstruct_realigned_tomos(cmd_align_path, path, 'tomos_alphaOffset_init')
    #9. symlink_new_aln(path, 'tomos_alphaOffset_init' ,'_alphaOffset_init')
    #8. apply_alphaOffset_aln(path)
    # above was done for getting alpha offset
    #7. reconstruct_realigned_tomos(cmd_align_path, path, 'tomos_aligned_oneItr')
    #6. symlink_new_aln(path, 'tomos_aligned_oneItr' ,'_oneItr')
    #5. duplicate_original_aln(path)
    #4. run_align_gold(path, centers)
    #3. open_napari(path)
    #2. reconstruct_tomos(cmd_path, path)
    #1. create_aretomo_alns(path)


print('end')