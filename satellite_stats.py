import rasterio
import numpy as np

def get_mean_index(file_path):

    with rasterio.open(file_path) as src:
        data = src.read(1)

    data = data[np.isfinite(data)]

    return {
        "mean": float(np.mean(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data))
    }