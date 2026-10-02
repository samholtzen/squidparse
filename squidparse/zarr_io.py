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

def _read_fov(df, well, p, meta):
    """Read one FOV's TIFFs as a (T, C, H, W) uint16 array (2x2 binned if meta.pseudobin)."""
    df_fov = df[(df.well == well) & (df.P == p)]

    raw_data = np.zeros((meta.T, meta.C, meta.H, meta.W), dtype=np.uint16)

    for row in df_fov.itertuples(index=False):
        if not row.exist:
            raise FileNotFoundError(f"Missing image: {row.image_dir}")
        if meta.pseudobin:
            binned = bin_2x2(tifffile.imread(row.image_dir))
            valid = (binned > 0).any(axis=0) if getattr(meta, 'stitch', False) else None

            raw_data[row.T, row.C] = fast_rolling_ball(binned, radius=200, downscale=16,
                                                   valid=valid)

        else:
            temp_im = tifffile.imread(row.image_dir)
            valid = (temp_im > 0).any(axis=0) if getattr(meta, 'stitch', False) else None

            raw_data[row.T, row.C] = fast_rolling_ball(temp_im, radius=200, downscale=16,
                                                   valid=valid)

    return raw_data


def _write_stitched_well(well_group, tiles, ys_um, xs_um, meta, correction, blend_pixels, work_dir):
    """Stitch `tiles` (T, P, C, H, W) of one well, then register/crop/store the mosaic as FOV '0'."""
    from squidparse.stitching import stitch_tiles

    mosaic, origin_px = stitch_tiles(
        tiles, ys_um, xs_um, meta.mpp, nuc_channel=meta.nuc_channel,
        correction=correction, blend_pixels=blend_pixels, work_dir=work_dir,
    )
    fov_group = well_group.require_group(name='0')
    _write_fov(fov_group, mosaic, meta)
    fov_group.attrs['stitching'] = dict(
        correction='nominal' if correction in (None, 'none', 'nominal', 'null') else 'tilefusion',
        origin_px=origin_px.tolist(),
        nominal_um=np.stack([ys_um, xs_um], axis=1).tolist(),
        mpp=meta.mpp,
    )


def make_zarr(meta, correction="tilefusion", blend_pixels=(50, 50), work_dir=None):
    """
    Build a Zarr store from any metadata object exposing the common
    AcquisitionMetadata-like interface: wells, T, P, Z, C, H, W,
    zarrfile, zarr_attrs, zarr_info, and path_dataframe().

    If the metadata was created with `stitch=True`, the overlapping FOVs of each well are
    stitched with TileFusion before registration and the mosaic is stored as FOV '0'
    (see `squidparse.stitching.stitch_tiles` for `correction`, `blend_pixels`, `work_dir`).
    """
    if meta.from_zarr:
        raise ValueError("Metadata was loaded from a zarr; raw images are needed to build one")

    z = zarr.open(meta.zarrfile)
    z.attrs['acquisition'] = meta.zarr_attrs
    z.attrs['squidparse'] = meta.zarr_info

    df = meta.path_dataframe()

    if getattr(meta, 'stitch', False):
        for well in tqdm(meta.wells):
            tiles = np.stack([_read_fov(df, well, p, meta) for p in range(meta.n_tiles)], axis=1)
            ys_um, xs_um = meta.fov_positions(well)
            _write_stitched_well(z.require_group(name=well), tiles, ys_um, xs_um,
                                 meta, correction, blend_pixels, work_dir)
        return

    for well, p in tqdm(itertools.product(meta.wells, range(meta.P))):
        well_group = z.require_group(name=well)
        fov_group = well_group.require_group(name=str(p))

        _write_fov(fov_group, _read_fov(df, well, p, meta), meta)


def _registration_movie(raw_data, meta):
    """log of `raw_data` for jitter registration. For stitched mosaics, empty pixels (corners,
    tile gaps) are static and would pin the phase correlation at zero shift, so they are
    flattened to the mean of the nuclear channel first."""
    if not getattr(meta, 'stitch', False):
        return np.log(raw_data)

    movie = np.log(np.maximum(raw_data, 1).astype(np.float32))
    nuc = raw_data[:, meta.nuc_channel]
    empty = nuc == 0
    movie[:, meta.nuc_channel][empty] = movie[:, meta.nuc_channel][~empty].mean()
    return movie


def _write_fov(fov_group, raw_data, meta):
    """Register jitter, crop to the common overlap, subtract background, and store
    `raw_data` (T, C, H, W) as the FOV's `offsets` and `images` arrays."""
    offsets = fov_group.require_array(name='offsets', shape=(meta.T, 2), dtype=int)
    if offsets.nchunks_initialized > 0:
        shifts = offsets[:]

    else:
        shifts = calculate_shift(_registration_movie(raw_data, meta), nuclear_channel=meta.nuc_channel)
        offsets[:] = shifts

    cropped_data = crop_to_common_overlap(raw_data, shifts)

    image_group = fov_group.require_array(name='images', shape=cropped_data.shape, dtype=np.uint16)
    image_group[:] = cropped_data


def make_zarr_nd2(meta, correction="tilefusion", blend_pixels=(50, 50), work_dir=None):
    """
    ND2 equivalent of make_zarr: same Zarr layout (<well>/<p>/images, offsets),
    built from an ND2Metadata object (one ND2 per well, P FOVs each). With `stitch=True`
    on the metadata, FOVs are stitched before registration; see `make_zarr`.
    """
    import nd2
    import dask.array as da

    if meta.from_zarr:
        raise ValueError("Metadata was loaded from a zarr; raw images are needed to build one")

    z = zarr.open(meta.zarrfile)
    z.attrs['acquisition'] = meta.zarr_attrs
    z.attrs['squidparse'] = meta.zarr_info

    def read_fov(arr, p):
        raw_data = np.asarray(arr[:, p]).astype(np.uint16)  # (T, C, Y, X)
        if meta.pseudobin:
            raw_data = np.stack([[bin_2x2(img) for img in frame] for frame in raw_data])
        return raw_data

    for well, path in tqdm(meta.nd2_files.items()):
        well_group = z.require_group(name=well)

        with nd2.ND2File(path) as f:
            dims = tuple(f.sizes)
            if dims != tuple(d for d in 'TPCYX' if d in dims):
                raise ValueError(f"Unexpected ND2 axis order {dims} in {path}")
            arr = f.to_dask()
            for idx, axis in enumerate('TPCYX'):  # add missing singleton axes -> (T, P, C, Y, X)
                if axis not in dims:
                    arr = da.expand_dims(arr, idx)

            if meta.stitch:
                tiles = np.stack([read_fov(arr, p) for p in range(meta.n_tiles)], axis=1)
                ys_um, xs_um = meta.fov_positions(well)
                _write_stitched_well(well_group, tiles, ys_um, xs_um,
                                     meta, correction, blend_pixels, work_dir)
                continue

            for p in range(meta.P):
                _write_fov(well_group.require_group(name=str(p)), read_fov(arr, p), meta)
