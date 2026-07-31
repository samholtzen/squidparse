import zarr
import tifffile
from tqdm import tqdm
import numpy as np


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
        z[row.well]['images'][row.T, row.P, row.Z, row.C, :, :] = tifffile.imread(row.image_dir)
