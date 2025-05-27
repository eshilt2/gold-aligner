import mrcfile
import numpy as np
import os
import pandas as pd
import csv

# getting mrc file metadata 
def get_metadata(folder_path):
    folder_list = os.listdir(folder_path)
    metadata_np = np.recarray(len(folder_list), dtype=[
        ('folder', 'U100'),                   # folder name
        ('pixelA', ('f4',3)),                     # pixelA is a float
        ('image_size', ('f4', 3))             # tuple of 3 ints (nx, ny, nz)
    ])    
    for i, folder in enumerate(folder_list):
        tomo_path = f"/nrs/liza/cathy_tomos/15f1_top_topop/{folder}/{folder}.mrc"
        if folder == '.stfolder':
            continue
        mrc = mrcfile.open(tomo_path)
        header = mrc.header
        metadata_np[i].folder = folder
        metadata_np[i].pixelA = (float(header.cella.x), float(header.cella.y), float(header.cella.z))
        metadata_np[i].image_size = (header.nx,header.ny, header.nz)

        mrc.close()
    
    return metadata_np

def export_metadata_to_csv(metadata_np, output_path):
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow([
            'folder',
            'pixelA_x', 'pixelA_y', 'pixelA_z',
            'size_x', 'size_y', 'size_z'
        ])

        # Write each row
        for row in metadata_np:
            writer.writerow([
                row.folder,
                *row.pixelA,
                *row.image_size
            ])

if __name__ == "__main__":
    path = "/nrs/liza/cathy_tomos/15f1_top_topop/"
    metadata = get_metadata(path)
    export_metadata_to_csv(metadata, output_path='cathy_tomos_metadata.csv')
    df = pd.DataFrame(metadata)
