import numpy as np
import zarr
from tqdm import tqdm

import itertools
from scipy.spatial import cKDTree

def segment_plate(meta):
    """
    Run Cellpose segmentation per well/position/time, producing nuclear
    (channel 0) and cytoplasmic-sum (channels 2+) masks.
    Works for any metadata object exposing wells, P, T, Z, H, W, zarrfile.
    """

    try:
        from cellpose import models
    except ImportError as e:
        raise ImportError(
            "segment_plate requires cellpose. "
            "Install it with: pip install imaging-pipeline[segmentation]"
        ) from e

    z = zarr.open(meta.zarrfile)
    model = models.CellposeModel(gpu=True)

    for well in meta.wells:
        z[well].zeros(
            name='masks',
            shape=(meta.T, meta.P, meta.Z, 2, meta.H, meta.W),
            chunks=(1, 1, meta.Z, 1, meta.H, meta.W),
            dtype=np.uint16
        )

    coords = list(itertools.product(meta.wells, range(meta.P), range(meta.T)))
    for well, p, t in tqdm(coords):
        z[well]['masks'][t, p, 0, 0], _, _ = model.eval(
            z[well]['images'][t, p, 0, meta.nuc_channel],
            cellprob_threshold=1.5
        )
        if isinstance(meta.cyto_channel, list):
            z[well]['masks'][t, p, 0, 1], _, _ = model.eval(
                np.sum(z[well]['images'][t, p, 0, meta.cyto_channel], axis=0),
                cellprob_threshold=1.5
            )
        else:
            z[well]['masks'][t, p, 0, 1], _, _ = model.eval(
                z[well]['images'][t, p, 0, meta.cyto_channel],
                cellprob_threshold=1.5
            )

def voronoi_mask(centroids, shape):
    """
    centroids: (N, 2) array of (row, col) or (y, x) coordinates
    shape: (H, W) of the output mask
    returns: (H, W) int array where each pixel = index of nearest centroid
    """
    tree = cKDTree(centroids)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    coords = np.column_stack([yy.ravel(), xx.ravel()])
    _, labels = tree.query(coords)
    return labels.reshape(shape)