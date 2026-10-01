import numpy as np
import zarr
from tqdm import tqdm

import itertools
from scipy.spatial import cKDTree

def segment_plate(meta, no_cyto=True):
    """
    Run Tracking software per well/position, producing tracked nuclear mask
    (channel 0) and cytoplasmic-ring (channels 1) masks.
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

    coords = list(itertools.product(meta.wells, range(meta.P)))

    for well, p in tqdm(coords):

        image = z[well][str(p)]['images'][:]

        if no_cyto:
            out_shape = (meta.T, 1, image.shape[-2], image.shape[-1])
        else:
            out_shape = (meta.T, 2, image.shape[-2], image.shape[-1])

        masks = z[well][str(p)].require_array(
            name='masks',
            shape=out_shape,
            chunks = (1, 1, out_shape[-2], out_shape[-1]),
            dtype=np.uint16
        )

        for t in range(meta.T):

            masks[t, 0], _, _ = model.eval(
                image[t, meta.nuc_channel],
                cellprob_threshold=1.5
            )

            if not no_cyto:

                if isinstance(meta.cyto_channel, list):
                    # There are multiple channels asked to be used as source of segmentation
                    # Add them together and send to cellpose for segmentation
                    masks[t, 1], _, _ = model.eval(
                        np.sum(image[t, meta.cyto_channel], axis=0),
                        cellprob_threshold=1.5
                    )
                    
                else:
                    masks[t, 1], _, _ = model.eval(
                        image[t, meta.cyto_channel],
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