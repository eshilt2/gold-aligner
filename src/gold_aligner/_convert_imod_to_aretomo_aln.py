import numpy as np
from cryoet_alignment import read, write 
from cryoet_alignment.io.cryoet_data_portal import Alignment
import mrcfile
import os
import re
import subprocess
from gold_aligner.align_gold import realign_gold
from gold_aligner.fix_alpha_offset import fix_alpha_offset 

def imod_to_aretomo(aln,            # read (past tense) in alignment file 
                    tilt_shape,     # shape of the tilt series (layers should be 3rd) 
                    tilt_com_path   # path to tilt.com file where alpha offset can be found
                    ):
    if hasattr(aln, 'xf') == True:
        imod_aln = Alignment.from_imod(aln)

        aretomo3_alignment = imod_aln.to_aretomo(ts_size=tilt_shape)

        for line in open(tilt_com_path):
            if re.findall(r'OFFSET*', line) == ['OFFSET']:
                match = re.findall(r'[0-9]+.[0-9]+', line)
                alphaOffset = float(match[0])
                aretomo3_alignment = fix_alpha_offset(aretomo3_alignment, alphaOffset)
    return aretomo3_alignment

def aretomo_to_imod(aln, tilt_shape, pixel_size):
    aretomo_aln = Alignment.from_aretomo3(aln)
    imod_aln = aretomo_aln.to_imod(ts_size=tilt_shape, ts_spacing=pixel_size)

    return imod_aln

def full_aretomo_to_imod(aln_path, tilt_path, output_folder):
    read_alignment = read(aln_path)

    with mrcfile.open(tilt_path) as mrctilt:
            tilt = mrctilt.data
            tilt_shape = tilt.shape
            invert_shape = (tilt_shape[2], tilt_shape[1], tilt_shape[0])
            header = mrctilt.header
    # get pixel_size
    pixel_size = header.cella.x
    
    imod_alignment = aretomo_to_imod(read_alignment, invert_shape, pixel_size)
    
    tomo_name = re.findall(r".*/(.*).mrc",tilt_path)[0]
    output_path = f"{output_folder}/{tomo_name}"

    write(imod_alignment, f"{output_path}")

if __name__ == "__main__":
    full_aretomo_to_imod('/nrs/liza/cathy_tomos/20241030_AMmilled12-1_15_otherPatch/20241030_AMmilled12-1_15_otherPatch_secItr.aln', '/nrs/liza/cathy_tomos/20241030_AMmilled12-1_15_otherPatch/20241030_AMmilled12-1_15.mrc', '/nrs/liza/cathy_tomos/ddw/imod_alignments/other' )
    
    tomo_list = ['20230512-EG18pos2_80','20240624_HippWaffle_86', '20240624_HippWaffle_18', '20240111_WaffleHipp_129', '20240307_AMmilled6_217', '20231026_HippAu_26', '20231017_EGmilled24-2_68',  '20241030_AMmilled12-2_30', '20231017_EGmilled24-2_77', '20241030_AMmilled12-2_78']
    tomo_list = ['20240624_HippWaffle_86']
    for tomo in tomo_list:
        # subprocess.run(f'cp /nrs/elferich/15f1_toptomos/15f1_top_topop/{tomo}/{tomo}_ODD.mrc /nrs/liza/cathy_tomos/ddw/imod_alignments/{tomo}', shell = True)
        full_aretomo_to_imod(f'/nrs/liza/cathy_tomos/tomos_oneItr_rerun/{tomo}_oneItr_rerun.aln', f'/nrs/liza/cathy_tomos/15f1_top_topop/{tomo}/{tomo}.mrc', '/nrs/liza/cathy_tomos/ddw/imod_alignments')
