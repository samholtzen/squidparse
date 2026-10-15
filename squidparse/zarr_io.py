import zarr
import tifffile
from tqdm import tqdm
import numpy as np
import itertools
from .utils import (estimate_flat_field, apply_flat_field, fast_rolling_ball, calculate_shift,
                    crop_to_common_overlap)

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

def _well_flat_fields(read_image, meta):
    """Flat field of each time point and channel of one well, (T, C, h, w) at reduced
    resolution (see `estimate_flat_field`). Each channel is estimated on its own, from all of
    the well's FOVs at that time point, read with `read_image(t, p, c)`."""
    return np.stack([[estimate_flat_field(read_image(t, p, c) for p in range(meta.n_tiles))
                      for c in range(meta.C)] for t in range(meta.T)])



def _tiff_well_flat_fields(df, well, meta):
    paths = {(r.T, r.P, r.C): r for r in df[df.well == well].itertuples(index=False)}

    def read_image(t, p, c):
        row = paths[(t, p, c)]
        if not row.exist:
            raise FileNotFoundError(f"Missing image: {row.image_dir}")
        img = np.squeeze(tifffile.imread(row.image_dir))
        return bin_2x2(img) if meta.pseudobin else img

    return _well_flat_fields(read_image, meta)


def _read_fov(df, well, p, meta):
    """Read one FOV's raw TIFFs as a (T, C, H, W) uint16 array (2x2 binned if meta.pseudobin)."""
    df_fov = df[(df.well == well) & (df.P == p)]

    raw_data = np.zeros((meta.T, meta.C, meta.H, meta.W), dtype=np.uint16)

    for row in df_fov.itertuples(index=False):
        if not row.exist:
            raise FileNotFoundError(f"Missing image: {row.image_dir}")
        img = np.squeeze(tifffile.imread(row.image_dir))
        raw_data[row.T, row.C] = bin_2x2(img) if meta.pseudobin else img

    return raw_data


def _clean(raw_data, meta, flats=None, confocal=False):
    """Flat-field and rolling-ball background-subtract `raw_data` (T, C, H, W).

    `flats` is (T, C, h, w) at any resolution, e.g. pooled over a well's FOVs. If it is None
    and `confocal` is set, each time point and channel's flat field is estimated from this
    movie's own frame (used on the stitched mosaic, where the FOVs are gone). The flat field is
    divided out first, since the diffuse background follows the illumination too. For
    stitched mosaics, empty pixels are excluded from the background and left at 0.

    Returns the cleaned movie and the (T, C, h, w) flat fields used (None for widefield).
    """
    if flats is None and confocal:
        flats = np.stack([[estimate_flat_field(raw_data[t, c]) for c in range(raw_data.shape[1])]
                          for t in range(raw_data.shape[0])])

    out = np.zeros(raw_data.shape, dtype=np.uint16)
    for t in range(raw_data.shape[0]):
        for c in range(raw_data.shape[1]):
            img = raw_data[t, c]
            valid = (img > 0) if getattr(meta, 'stitch', False) else None
            if flats is not None:
                # out[t, c] = apply_flat_field(img, flats[t,c])
                out[t, c] = img

            else:
                out[t, c] = img
                # out[t, c] = fast_rolling_ball(img, radius=200, downscale=16, valid=valid)
                
    return out


def _write_stitched_well(well_group, tiles, ys_um, xs_um, meta, correction, blend_pixels, work_dir,
                         confocal=False):
    """Stitch the raw `tiles` (T, P, C, H, W) of one well, then flat-field (if `confocal`) and
    background-subtract, register, crop and store the mosaic as FOV '0'."""
    from .stitching import stitch_tiles

    mosaic, origin_px = stitch_tiles(
        tiles, ys_um, xs_um, meta.mpp, nuc_channel=meta.nuc_channel,
        correction=correction, blend_pixels=blend_pixels, work_dir=work_dir,
    )
    del tiles
    cleaned = _clean(mosaic, meta, confocal=confocal)
    del mosaic


    fov_group = well_group.require_group(name='0')
    _register_and_store(fov_group, cleaned, meta)
    fov_group.attrs['stitching'] = dict(
        correction='nominal' if correction in (None, 'none', 'nominal', 'null') else 'tilefusion',
        origin_px=origin_px.tolist(),
        nominal_um=np.stack([ys_um, xs_um], axis=1).tolist(),
        mpp=meta.mpp,
    )


def make_zarr(meta, correction="tilefusion", blend_pixels=(50, 50), work_dir=None, confocal=False):
    """
    Build a Zarr store from any metadata object exposing the common
    AcquisitionMetadata-like interface: wells, T, P, Z, C, H, W,
    zarrfile, zarr_attrs, zarr_info, and path_dataframe().

    If the metadata was created with `stitch=True`, the overlapping FOVs of each well are
    stitched with TileFusion as raw images; the mosaic is then flat-fielded and
    background-subtracted, registered for jitter, and stored as FOV '0'
    (see `squidparse.stitching.stitch_tiles` for `correction`, `blend_pixels`, `work_dir`).

    confocal=True divides every image by an empirical flat field before background
    subtraction; leave False for widefield data. The illumination changes between wells and
    time points, so there is one flat field per (well, time point, channel). For stitched
    wells it is estimated from that channel's mosaic; otherwise from that channel's images
    in all of the well's FOVs at that time point. The fields are saved in each well's group
    as `flatfield`, shape (T, C, h, w) at reduced resolution (see `estimate_flat_field`).
    """
    if meta.from_zarr:
        raise ValueError("Metadata was loaded from a zarr; raw images are needed to build one")

    z = zarr.open(meta.zarrfile)
    z.attrs['acquisition'] = meta.zarr_attrs
    z.attrs['squidparse'] = meta.zarr_info

    df = meta.path_dataframe()
    stitch = getattr(meta, 'stitch', False)

    for well in tqdm(meta.wells):
        well_group = z.require_group(name=well)
        flats = None if stitch or not confocal else _tiff_well_flat_fields(df, well, meta)

        if stitch:
            tiles = np.stack([_read_fov(df, well, p, meta) for p in range(meta.n_tiles)], axis=1)
            ys_um, xs_um = meta.fov_positions(well)
            _write_stitched_well(well_group, tiles, ys_um, xs_um,
                                 meta, correction, blend_pixels, work_dir, confocal)
            continue 

        for p in range(meta.P):
            _write_fov(well_group.require_group(name=str(p)), _read_fov(df, well, p, meta), meta, flats)


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


def _write_fov(fov_group, raw_data, meta, flats=None):
    """Flat-field (with the well's `flats`, if given) and background-subtract `raw_data`
    (T, C, H, W), then register, crop and store it."""
    _register_and_store(fov_group, _clean(raw_data, meta, flats)[0], meta)


def _register_and_store(fov_group, data, meta):
    """Register jitter in `data` (T, C, H, W), crop to the common overlap, and store it as the
    FOV's `offsets` and `images` arrays."""
    offsets = fov_group.require_array(name='offsets', shape=(meta.T, 2), dtype=int)
    if offsets.nchunks_initialized > 0:
        shifts = offsets[:]

    else:
        shifts = calculate_shift(_registration_movie(data, meta), nuclear_channel=meta.nuc_channel)
        offsets[:] = shifts

    cropped_data = crop_to_common_overlap(data, shifts)

    image_group = fov_group.require_array(name='images', shape=cropped_data.shape, dtype=np.uint16)
    image_group[:] = cropped_data
