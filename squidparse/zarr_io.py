import zarr
import tifffile
from tqdm import tqdm
import numpy as np

def bin_2x2(img):
    """
    Bin a 2D image by 2x2 blocks.
    img: 2D numpy array, shape must be divisible by 2 in both dims
    method: "sum" or "mean"
    """
    h, w = img.shape
    assert h % 2 == 0 and w % 2 == 0, "Dimensions must be divisible by 2"
    
    reshaped = img.reshape(h // 2, 2, w // 2, 2)

    return reshaped.mean(axis=(1, 3)).astype(np.uint16)

def make_zarr(meta):
    """
    Build a Zarr store from any metadata object exposing the common
    AcquisitionMetadata-like interface: wells, T, P, Z, C, H, W,
    zarrfile, zarr_attrs, and path_dataframe().
    """
    z = zarr.open(meta.zarrfile)
    z.attrs['acquisition'] = meta.zarr_attrs

    for well in meta.wells:
        well_group = z.create_group(name=well)
        well_group.zeros(
            name='images',
            shape=(meta.T, meta.P, meta.Z, meta.C, meta.H, meta.W),
            chunks=(1, 1, meta.Z, 1, meta.H, meta.W),
            dtype=np.uint16
        )

    df = meta.path_dataframe()

    for row in tqdm(df.itertuples(index=False), total=len(df)):
        if not row.exist:
            raise FileNotFoundError(f"Missing image: {row.image_dir}")
        if meta.pseudobin:
            z[row.well]['images'][row.T, row.P, row.Z, row.C, :, :] = bin_2x2(tifffile.imread(row.image_dir))
        else:
            z[row.well]['images'][row.T, row.P, row.Z, row.C, :, :] = tifffile.imread(row.image_dir)

