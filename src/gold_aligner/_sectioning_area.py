import numpy as np

def make_sectioned_area(n, tomo_shape):
    arr = np.zeros(tomo_shape, dtype=int)

    y_splits = np.array_split(np.arange(tomo_shape[1]), n)
    x_splits = np.array_split(np.arange(tomo_shape[2]), n)
    section_centers = {}
    label = 1
    for i, y_idx in enumerate(y_splits):
        for j, x_idx in enumerate(x_splits):
            arr[:, y_idx[:, None], x_idx] = label
            

            y_center = np.mean(y_idx)
            x_center = np.mean(x_idx)

            section_centers[label] = (y_center, x_center)
            label += 1
    return arr, section_centers
