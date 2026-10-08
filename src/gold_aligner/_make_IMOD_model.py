""" Creates IMOD model when called"""
import pandas as pd
import imodmodel
# from imodmodel import ImodModel
# from imodmodel.models import (
#     Contour,
#     ContourHeader,
#     Mesh,
#     IMAT,
#     SLAN,
#     MeshHeader,
#     Object,
#     ObjectHeader,
# )


def make_imod_model(selected_points,    # array : coordinates of AuNPs (x,y,z)
                    tomo_num,           # str   : taken from tilt image path name during init
                    type_ = "tomo",     # str   : prefix to specify model : tomo, tilt, tilt_conv
                    custom_name = None  # str   : without .mod ending for testing/debugging
                    ):
    modelPeak = ImodModel(objects=[
        Object(
            contours=[
                Contour(
                    header=ContourHeader(
                        psize= selected_points.shape[0],
                        flags=16,
                        time=0,
                        surf=0,
                    ),
                    points = selected_points,
                )
            ]
        )
    ])
    if custom_name == None:
        modelPeak.to_file(f'{type_}{tomo_num}_au.mod')
    else:
        modelPeak.to_file(f'{custom_name}.mod')

def make_IMOD_model_UPDATED(selected_points,        # array : coordinates (x,y,z) 
                           path_and_name           # str   : path and name of output model (omit .mod)
                           ):
    df = pd.DataFrame({'object_id' : [0 for i in range(len(selected_points))],
    'contour_id': [0 for i in range(len(selected_points))],
    'x': selected_points[:,0],
    'y': selected_points[:,1],
    'z': selected_points[:,2]})
    imodmodel.write(df, f'{path_and_name}.mod')
