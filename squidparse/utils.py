import numpy as np

import zarr

from skimage.transform import resize
from skimage.restoration import rolling_ball
from skimage.filters import gaussian

from skimage.segmentation import expand_labels
from scipy.ndimage import uniform_filter1d

from skimage.registration import phase_cross_correlation

def get_signal(meta, well, channel, compartment, p=None):

    z = zarr.open(meta.zarrfile)
    signal = []

    if p is None:
        for p in range(meta.P):
            signal.append(z[well][str(p)]['features'][f'{compartment}_intensity'][:,:,channel])
        signal = np.hstack(signal)

    else:
        signal=z[well][str(p)]['features'][f'{compartment}_intensity'][:,:,channel]

    nanfilter = (np.sum(np.isnan(signal), axis=0) == 0)

    return signal[:, nanfilter]

def get_cnr(meta, well, channel):

    cyto = get_signal(meta, well, channel, 'cyto')
    nuc = get_signal(meta, well, channel, 'nuc')

    return cyto/nuc

def get_cytoring(nuclear_mask, ring_width=5):
    """
    Given a labeled nuclear mask, return a labeled mask of the
    cytoplasmic ring around each nucleus (dilated region minus nucleus).

    Parameters
    ----------
    nuclear_mask : ndarray (2D or 3D)
        Labeled mask where each nucleus has a unique integer label,
        and background is 0.
    ring_width : int
        Number of pixels/voxels to dilate outward from each nucleus.

    Returns
    -------
    ring_mask : ndarray
        Labeled mask of the same shape, where each ring has the same
        label as its corresponding nucleus, and 0 elsewhere.
    """
    # Dilate labels outward, but stop at the midline between neighboring nuclei
    dilated_nuc = expand_labels(nuclear_mask, distance=3)
    dilated_ring = expand_labels(nuclear_mask, distance=ring_width+3)

    # Ring = dilated region minus the original nucleus
    ring_mask = np.where(dilated_nuc == 0, dilated_ring, 0)

    return ring_mask


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


def make_mask(img, upper_percentile=99.5):
    """True = keep this pixel, False = exclude (bright spot / saturation)."""
    thresh = np.percentile(img, upper_percentile)
    return img < thresh

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
    shifts = np.zeros((n_t, n_spatial), dtype=int)

    if reference == "first":
        ref_img = nuc[0]
        ref_mask = make_mask(ref_img)
        for t in range(1, n_t):
            mov_mask = make_mask(nuc[t])
            shift_est, _, _ = phase_cross_correlation(
                ref_img, nuc[t],
                reference_mask=ref_mask,
                moving_mask=mov_mask,
            )
            shifts[t] = shift_est.astype(int)
    elif reference == "previous":
        cumulative = np.zeros(n_spatial)
        for t in range(1, n_t):
            shift_est, _, _ = phase_cross_correlation(
                nuc[t - 1], nuc[t], upsample_factor=1, disambiguate=True
            )
            cumulative = cumulative + shift_est
            shifts[t] = cumulative.astype(int)
    else:
        raise ValueError("reference must be 'first' or 'previous'")

    return shifts


def fast_illum(image, sigma=200, downscale=4):
    small = resize(image, 
                    (image.shape[0] // downscale, image.shape[1] // downscale),
                    preserve_range=True, anti_aliasing=True)
    
    small_bg = gaussian(small, sigma=sigma, preserve_range=True, mode='reflect')
    
    bg = resize(small_bg, image.shape, preserve_range=True, anti_aliasing=False)
    bg = bg/bg.max()
    
    return image / bg

def fast_rolling_ball(image, radius=50, downscale=4):
    small = resize(image, 
                    (image.shape[0] // downscale, image.shape[1] // downscale),
                    preserve_range=True, anti_aliasing=True)
    
    small_bg = rolling_ball(small, radius=radius / downscale)
    
    bg = resize(small_bg, image.shape, preserve_range=True, anti_aliasing=False) + 1
    return np.clip(image - bg, 1, (2**16)-1)

def movmean(array, size=3):
    arr_smooth = uniform_filter1d(array, size=size, axis=0)
    return arr_smooth