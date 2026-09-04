import numpy as np
from skimage.registration import phase_cross_correlation
import zarr
from tqdm import tqdm
import itertools
from squidparse.metadata import AcquisitionMetadata

def calculate_shift(
    movie: np.ndarray,
    nuclear_channel: int,
    channel_axis: int = 1,
    time_axis: int = 0,
    reference: str = "first",
):
    """
    Register a multi-channel timelapse movie using a nuclear marker channel
    to correct jitter, then crop all frames to the region that is in-frame
    at every timepoint.

    Parameters
    ----------
    movie : np.ndarray
        Array containing the movie. Must include a time axis and a channel
        axis (other axes, e.g. Y/X or Z/Y/X, are treated as spatial and
        registered together).
    nuclear_channel : int
        Index into the channel axis identifying the nuclear marker, used
        to estimate the per-frame jitter shift.
    channel_axis : int, default 1
        Axis of `movie` corresponding to channels. Default assumes
        shape (T, C, Y, X).
    time_axis : int, default 0
        Axis of `movie` corresponding to time.
    reference : {"first", "previous"}, default "first"
        Whether each frame is registered against the first frame
        (global drift correction) or against the previous frame
        (frame-to-frame, then shifts are accumulated). "first" is
        usually more robust for slow jitter; "previous" can help with
        large cumulative drift but is more sensitive to noise.

    Returns
    -------
    shifts : np.ndarray
        Array of shape (T, n_spatial_dims) with the shift applied to
        each frame (in pixels, per spatial axis), in the same order as
        the spatial axes of the movie (axes other than time/channel).
    """
    movie = np.asarray(movie)
    ndim = movie.ndim

    # Identify spatial axes (everything that isn't time or channel)
    spatial_axes = [a for a in range(ndim) if a not in (time_axis, channel_axis)]
    n_spatial = len(spatial_axes)
    n_t = movie.shape[time_axis]

    # Move axes to a canonical order: (T, C, *spatial)
    order = [time_axis, channel_axis] + spatial_axes
    m = np.moveaxis(movie, order, list(range(ndim)))  # shape (T, C, *spatial)

    nuc = m[:, nuclear_channel]  # shape (T, *spatial)

    # --- Estimate per-frame shifts using the nuclear channel ---
    shifts = np.zeros((n_t, n_spatial), dtype=float)

    if reference == "first":
        ref_img = nuc[0]
        for t in range(1, n_t):
            shift_est, _, _ = phase_cross_correlation(
                ref_img, nuc[t], upsample_factor=1, disambiguate=True
            )
            shifts[t] = shift_est
    elif reference == "previous":
        cumulative = np.zeros(n_spatial)
        for t in range(1, n_t):
            shift_est, _, _ = phase_cross_correlation(
                nuc[t - 1], nuc[t], upsample_factor=1, disambiguate=True
            )
            cumulative = cumulative + shift_est
            shifts[t] = cumulative
    else:
        raise ValueError("reference must be 'first' or 'previous'")


    return shifts


def register_plate(meta: AcquisitionMetadata):
    """
    Compute per-well, per-position jitter offsets across time using the
    nuclear channel, and store them in a Zarr 'offsets' array.
    Works for any metadata object exposing wells, P, T, zarrfile.
    """
    z = zarr.open(meta.zarrfile)

    for well in meta.wells:
        z[well].zeros(name='offsets', shape=(meta.P, meta.T, 2), dtype=int)

    for well, p in tqdm(list(itertools.product(meta.wells, range(meta.P)))):
        z[well]['offsets'][p, :, :] = calculate_shift(
            z[well]['images'][:, p, 0, :1, :, :],
            nuclear_channel=meta.nuc_channel
        )

def crop_to_common_overlap(frames, offsets):
    """
    frames:  array of shape (T, C, H, W)
    offsets: array-like of shape (T, 2), each row (oy, ox) — integer pixel
             offsets relative to a common reference (e.g. frame 0 -> (0,0))

    Returns: array of shape (T, C, overlap_h, overlap_w)
    """
    T, C, H, W = frames.shape
    offsets = np.asarray(offsets)
    oys, oxs = offsets[:, 0], offsets[:, 1]

    overlap_h = H - (oys.max() - oys.min())
    overlap_w = W - (oxs.max() - oxs.min())

    if overlap_h <= 0 or overlap_w <= 0:
        raise ValueError("Offsets exceed frame size — no common overlap.")

    out = np.empty((T, C, overlap_h, overlap_w), dtype=frames.dtype)
    for t in range(T):
        r0 = oys.max() - oys[t]
        c0 = oxs.max() - oxs[t]
        out[t] = frames[t, :, r0:r0+overlap_h, c0:c0+overlap_w]

    return out