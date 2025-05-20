import numpy as np
from cryoet_alignment import read, write 
from cryoet_alignment.io.cryoet_data_portal import Alignment
import mrcfile
import os
import re
import subprocess
from gold_aligner.align_gold import realign_gold
from gold_aligner.fix_alpha_offset import fix_alpha_offset 

def imod_to_aretomo(aln, tilt_shape):
    if hasattr(aln, 'xf') == True:
        imod_aln = Alignment.from_imod(aln)

        aretomo3_alignment = imod_aln.to_aretomo(ts_size=tilt_shape)
    
    return aretomo3_alignment


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
    
    create_aretomo_alns(path)

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