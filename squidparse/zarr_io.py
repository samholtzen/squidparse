import zarr
import tifffile
from tqdm import tqdm
import numpy as np
import itertools
from squidparse.utils import fast_rolling_ball, calculate_shift, crop_to_common_overlap

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
    zarrfile, zarr_attrs, zarr_info, and path_dataframe().
    """
    if meta.from_zarr:
        raise ValueError("Metadata was loaded from a zarr; raw images are needed to build one")

    z = zarr.open(meta.zarrfile)
    z.attrs['acquisition'] = meta.zarr_attrs
    z.attrs['squidparse'] = meta.zarr_info

    df = meta.path_dataframe()

    for well, p in tqdm(itertools.product(meta.wells, range(meta.P))):
        well_group = z.require_group(name=well)
        fov_group = well_group.require_group(name=str(p))

        df_fov = df[(df.well == well) & (df.P == p)]

        raw_data = np.zeros((meta.T, meta.C, meta.H, meta.W), dtype=np.uint16)

        for row in df_fov.itertuples(index=False):
            if not row.exist:
                raise FileNotFoundError(f"Missing image: {row.image_dir}")
            if meta.pseudobin:
                binned = bin_2x2(tifffile.imread(row.image_dir))
                raw_data[row.T, row.C] = binned

            else:
                raw_data[row.T, row.C] = tifffile.imread(row.image_dir)

        offsets = fov_group.require_array(name='offsets', shape=(meta.T, 2), dtype=int)
        if offsets.nchunks_initialized > 0:
            shifts = offsets[:]

        else:
            shifts = calculate_shift(np.log(raw_data), nuclear_channel=meta.nuc_channel)
            offsets[:] = shifts

        cropped_data = crop_to_common_overlap(raw_data, shifts)

        for t, c in itertools.product(range(meta.T), range(meta.C)):
            cropped_data[t, c] = fast_rolling_ball(cropped_data[t, c], radius=200, downscale=16)

        image_group = fov_group.require_array(name='images', shape=cropped_data.shape, dtype=np.uint16)
        image_group[:] = cropped_data
