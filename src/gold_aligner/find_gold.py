import mrcfile
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from imodmodel import ImodModel
from imodmodel.models import (
    Contour,
    ContourHeader,
    Mesh,
    IMAT,
    SLAN,
    MeshHeader,
    Object,
    ObjectHeader,
)

### Get AUNPs in tomograph
# with mrcfile.open("/nrs/liza/aretomoe3_tes/20231017_EGmilled24-2_68_Vol.mrc") as mrc:
#     invert_img = mrc.data * -1 # flip black and white
#     peak_coords = peak_local_max(invert_img, min_distance = 3,threshold_rel = .3, exclude_border = (70, 50,50)).tolist() 

#     for entry in peak_coords:
#         entry[0], entry[2] = entry[2], entry[0]
#         peak_coords[peak_coords.index(entry)] = entry
#     peak_coords = np.array(peak_coords)


# modelPeak = ImodModel(objects=[
#     Object(
#         contours=[
#             Contour(
#                 header=ContourHeader(
#                     psize= peak_coords.shape[0],
#                     flags=16,
#                     time=0,
#                     surf=0,
#                 ),
#                 points = peak_coords,
#             )
#         ]
#     )
# ])

# modelPeak.to_file('peak_model.mod')

### Get contours of AUNPs tilt series
total_tilt_points = []
with mrcfile.open("/nrs/liza/aretomoe3_tes/20231017_EGmilled24-2_68.mrc") as mrc_tilt:
    print(mrc_tilt.data.max())
    distance = ndi.distance_transform_edt(mrc_tilt)
    invrt_tilt = (mrc_tilt.data - mrc_tilt.data.max())*-1


    for z in range(0,invrt_tilt.shape[0]):
        working_tilt = invrt_tilt[z,:,:]
        image_max = ndi.maximum_filter(working_tilt, size=5, mode='constant')
        tilt_points = peak_local_max(working_tilt, threshold_rel=0.95, exclude_border=(100,100))      
        for entry in tilt_points:
            entry[0], entry[1] = entry[1], entry[0]
            tilt_points[tilt_points.index(entry)] = entry + [z]

        total_tilt_points = total_tilt_points + tilt_points
        print(len(total_tilt_points))
    
    total_tilt_points = np.array(total_tilt_points)
    print(total_tilt_points)

#     # for entry in tilt_coords:
#     #     entry[0], entry[2] = entry[2], entry[0]
#     #     tilt_coords[tilt_coords.index(entry)] = entry
#     # tilt_coords = np.array(tilt_coords)
#     # print(tilt_coords)
  
modelPeak = ImodModel(objects=[
    Object(
        # header = ObjectHeader(
        #     contsize = len(countours)
        # )
        contours=[
            Contour(
                header=ContourHeader(
                    psize= total_tilt_points.shape[0],
                    flags=16,
                    time=0,
                    surf=0,
                ),
                points = total_tilt_points,
            )
        ]
    )
])

modelPeak.to_file('tilt_model.mod')


### Old Code

# all_coords = []
# peak_all_coords = []
#     for z in range(70,mrc.data.shape[0]-1):
#         current_slice = mrc.data[z, :,:]
#         #print(mrc.data[228, 484, 485])
#         minima = current_slice.min()
#         minimaLoc = np.where(current_slice <=minima*.85) # can play around with the cutoff

#         #using local max
#         invrt_img = current_slice * -1
#         image_max = ndi.maximum_filter(invrt_img, size= 10, mode='constant')
#         peak_coords = peak_local_max(invrt_img, min_distance = 25, exclude_border
#  = 50, threshold_rel = .85).tolist()
        
#         for coord in peak_coords:
#             #print(coord)
#             coord[1], coord[0] = coord[0], coord[1]
#             peak_coords[peak_coords.index(coord)] = coord + [z]
#             #print(coord)

#         peak_all_coords = peak_all_coords + peak_coords
#         for mini in range(len(minimaLoc[0])):
#             one_coord = [x[mini-1] for x in minimaLoc] + [z]
#             one_coord[1], one_coord[0] = one_coord[0], one_coord[1] # <- this sets coord[0] to x, coord[1] to y
#             if one_coord[0] >= 50 & one_coord[0] <= (mrc.data.shape[2]-50) & one_coord[1] >= 50 & one_coord[1] <= (mrc.data.shape[1]-50):
#                 all_coords.append(one_coord)
#             #else:
#                 #print(one_coord)

#         array_coords = np.array(all_coords)
#         flt_min_coords = array_coords.astype(np.float32)

#     peak_array_coords = np.array(peak_all_coords)

# model = ImodModel(objects=[
#     Object(
#         contours=[
#             Contour(
#                 header=ContourHeader(
#                     psize= flt_min_coords.shape[0],
#                     flags=16,
#                     time=0,
#                     surf=0,
#                 ),
#                 points = flt_min_coords,
#             )
#         ]
#     )
# ])

# model.to_file('test_model.mod')

