import os
from cryoet_alignment import read
from cryoet_alignment import write

### Fix alpha tilt in Aretomoe3
new_alignment = 20
aretomo3_alignment = read("/nrs/liza/aretomoe3_remove_patch/20231017_EGmilled24-2_68.aln")
old_alignment = aretomo3_alignment.AlphaOffset
aretomo3_alignment.AlphaOffset = new_alignment
dif_alnmt = new_alignment - old_alignment
for algnmt in aretomo3_alignment.GlobalAlignments:
    algnmt.tilt = algnmt.tilt + dif_alnmt
write(aretomo3_alignment, "/nrs/liza/aretomoe3_remove_patch/20231017_EGmilled24-2_68_fixed.aln")
if os.path.isfile("/nrs/liza/aretomoe3_remove_patch/20231017_EGmilled24-2_68_old.aln") == False:
    os.rename('/nrs/liza/aretomoe3_remove_patch/20231017_EGmilled24-2_68.aln', '/nrs/liza/aretomoe3_remove_patch/20231017_EGmilled24-2_68_old.aln')
    os.rename('/nrs/liza/aretomoe3_remove_patch/20231017_EGmilled24-2_68_fixed.aln', '/nrs/liza/aretomoe3_remove_patch/20231017_EGmilled24-2_68.aln')
else:
    print("File has already been fixed, please review if you'd like to continue")