from cryoet_alignment import read
from cryoet_alignment import write

### Fix alpha tilt in Aretomoe3 alignment file
def fix_alpha_offset(
                   original_aln,                  # class   : class from aretomo3 generated .aln file used for initially aligned tomogram
                   alpha_tilt = None,             # int     : correct Alpha Offset if known
                   ):
    
    old_alpha = original_aln.AlphaOffset
    original_aln.AlphaOffset = alpha_tilt
    dif_alnmt = alpha_tilt - old_alpha
    for algnmt in original_aln.GlobalAlignments:
        algnmt.tilt = algnmt.tilt + dif_alnmt
    return original_aln

def undo_alpha_offset(
        new_aln,
        original_alpha_tilt
):
    old_alpha = new_aln.AlphaOffset
    new_aln.AlphaOffset = original_alpha_tilt
    dif_alnmt = original_alpha_tilt - old_alpha
    for algnmt in new_aln.GlobalAlignments:
        algnmt.tilt = algnmt.tilt + dif_alnmt
    return new_aln




####### fix_alpha_offset test
# aln_path = "/nrs/liza/Aret3_rm_patch_tomo54/20231017_EGmilled24-2_54_original.aln"
# output_aln_path = "/nrs/liza/Aret3_rm_patch_tomo54_alphatilt_fixed/20231017_EGmilled24-2_54.aln"
# aretomo3_alignment = read(aln_path)

# print(aretomo3_alignment.GlobalAlignments[0].tilt)

# new_aln = fix_alpha_tilt(aretomo3_alignment, alpha_tilt = 20)
# write(new_aln, output_aln_path)
# print('end')
# new_alignment = 20
# aretomo3_alignment = read("/nrs/liza/aretomoe3_remove_patch/20231017_EGmilled24-2_68.aln")
# old_alignment = aretomo3_alignment.AlphaOffset
# aretomo3_alignment.AlphaOffset = new_alignment
# dif_alnmt = new_alignment - old_alignment
# for algnmt in aretomo3_alignment.GlobalAlignments:
#     algnmt.tilt = algnmt.tilt + dif_alnmt
# write(aretomo3_alignment, "/nrs/liza/aretomoe3_remove_patch/20231017_EGmilled24-2_68_fixed.aln")
# if os.path.isfile("/nrs/liza/aretomoe3_remove_patch/20231017_EGmilled24-2_68_old.aln") == False:
#     os.rename('/nrs/liza/aretomoe3_remove_patch/20231017_EGmilled24-2_68.aln', '/nrs/liza/aretomoe3_remove_patch/20231017_EGmilled24-2_68_old.aln')
#     os.rename('/nrs/liza/aretomoe3_remove_patch/20231017_EGmilled24-2_68_fixed.aln', '/nrs/liza/aretomoe3_remove_patch/20231017_EGmilled24-2_68.aln')
# else:
#     print("File has already been fixed, please review if you'd like to continue")
# return 
##################################
