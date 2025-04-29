from cryoet_alignment import read
from cryoet_alignment import write

### Fix alpha tilt in Aretomoe3 alignment file
def fix_alpha_offset(original_aln,                      # str : path to aretomo3 generated .aln file used for initially aligned tomogram
                   alpha_tilt = None,             # int : correct Alpha Offset if known
                   ):
    
    old_alpha = original_aln.AlphaOffset
    original_aln.AlphaOffset = alpha_tilt
    dif_alnmt = alpha_tilt - old_alpha
    for algnmt in original_aln.GlobalAlignments:
        algnmt.tilt = algnmt.tilt + dif_alnmt
    return original_aln

