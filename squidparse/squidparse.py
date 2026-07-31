import numpy as np
from skimage.registration import phase_cross_correlation
import yaml
import zarr
import tifffile
import datetime
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from cellpose import models
import itertools
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

class AcquisitionMetadata:
    """
    Parses a single acquisition folder of the form:
        <acqdir>/
            0/, 1/, 2/, ...          <- frame subfolders (T dimension)
            acquisition.yaml
            coordinates.csv
    This is the common unit shared by a standalone timelapse experiment
    and a single ISS round.
    """

    def __init__(self, acqdir):
        self.acqdir = Path(acqdir)
        self._extract_metadata()

    def _extract_metadata(self):
        # sort numerically so frame order is guaranteed, not iterdir()-order
        self.frame_dirs = sorted(
            (d for d in self.acqdir.iterdir() if d.is_dir()),
            key=lambda d: int(d.name)
        )
        self.T = len(self.frame_dirs)

        with open(self.acqdir / 'acquisition.yaml') as f:
            self.metadata = yaml.safe_load(f)

        coord_df = pd.read_csv(self.acqdir / 'coordinates.csv')
        wells_counts = coord_df['region'].value_counts()
        self.wells = wells_counts.keys().to_list()
        self.P = wells_counts.values[0].item()

        self.Z = self.metadata['z_stack']['nz']
        self.channels = [c['name'] for c in self.metadata['channels']]
        self.C = len(self.channels)

        self._get_image_size()

    @property
    def start_time(self):
        return self.metadata['acquisition']['start_time']

    @property
    def zarr_attrs(self):
        return self.metadata

    def _get_image_size(self):
        temp_path = self.image_path(self.wells[0], 0, 0, 0, 0)
        with tifffile.TiffFile(temp_path) as tif:
            self.H, self.W = tif.shaped_metadata[0]['shape']

    def _make_zarrfile_name(self):
        exp_dt = datetime.datetime.fromtimestamp(self.start_time)
        self.zarrfile = exp_dt.strftime('%Y%m%d') + '.zarr'

    def _size(self):
        return (self.T, self.P, self.Z, self.C, self.H, self.W)

    def image_path(self, well, T, P, Z, C):
        if not (0 <= T < self.T and 0 <= P < self.P and 0 <= Z < self.Z
                and 0 <= C < self.C and well in self.wells):
            raise IndexError(f"Coordinate out of range: {well, T, P, Z, C}")

        frame_dir = self.frame_dirs[T]
        channel_name = '_'.join(self.channels[C].split(' '))
        return frame_dir / f'{well}_{P}_{Z}_{channel_name}.tiff'

    def path_dataframe(self):
        all_coords = itertools.product(
            self.wells, range(self.T), range(self.P), range(self.Z), range(self.C)
        )
        df = pd.DataFrame(all_coords, columns=['well', 'T', 'P', 'Z', 'C'])
        df['image_dir'] = [
            self.image_path(r.well, r.T, r.P, r.Z, r.C) for r in df.itertuples()
        ]
        df['exist'] = df['image_dir'].map(lambda p: Path(p).exists())
        return df

    def __repr__(self):
        return (f'Acquisition dir: {self.acqdir}\n'
                f'Channels: {self.channels}\n'
                f'Image dimensions: {self._size()}')

class TimeLapseMetadata(AcquisitionMetadata):
    def __init__(self, expdir):
        super().__init__(expdir)
        self.expdir = self.acqdir
        self.exp_name = self.expdir.stem
        self._make_zarrfile_name()


class ISSMetadata:
    def __init__(self, expdir):
        self.expdir = Path(expdir)
        self.exp_name = self.expdir.stem
        self.zarrfile = f"{self.exp_name}.zarr"

        # parse every round as its own AcquisitionMetadata (T will be 1 for each,
        # since a round folder only ever contains the single '0' frame dir)
        round_dirs = [d for d in self.expdir.iterdir() if d.is_dir()]
        rounds = [AcquisitionMetadata(d) for d in round_dirs]

        # sort rounds chronologically by acquisition start time
        self.rounds = sorted(rounds, key=lambda r: r.start_time)
        self.round_dirs = [r.acqdir for r in self.rounds]
        self.round_meta = [r.metadata for r in self.rounds]

        self.T = len(self.rounds)  # rounds ARE the time dimension for ISS

        # validate consistency across rounds and pull shared attrs from round 0
        self._validate_and_set_shared_attrs()

    def _validate_and_set_shared_attrs(self):
        ref = self.rounds[0]
        for r in self.rounds[1:]:
            if r.channels != ref.channels or r.wells != ref.wells or r.Z != ref.Z:
                raise ValueError(f"Round {r.acqdir} metadata does not match round {ref.acqdir}")

        self.wells, self.P, self.Z = ref.wells, ref.P, ref.Z
        self.channels, self.C = ref.channels, ref.C
        self.H, self.W = ref.H, ref.W

    def _size(self):
        return (self.T, self.P, self.Z, self.C, self.H, self.W)

    def image_path(self, well, T, P, Z, C):
        # T indexes which round; each round's own internal T is always 0
        return self.rounds[T].image_path(well, 0, P, Z, C)
    
    @property
    def zarr_attrs(self):
        return self.round_meta

    def path_dataframe(self):
        all_coords = itertools.product(
            self.wells, range(self.T), range(self.P), range(self.Z), range(self.C)
        )
        df = pd.DataFrame(all_coords, columns=['well', 'T', 'P', 'Z', 'C'])
        df['image_dir'] = [
            self.image_path(r.well, r.T, r.P, r.Z, r.C) for r in df.itertuples()
        ]
        df['exist'] = df['image_dir'].map(lambda p: Path(p).exists())
        return df

    def __repr__(self):
        return (f'Experiment directory: {self.expdir}\n'
                f'Channels: {self.channels}\n'
                f'Image dimensions: {self._size()}')

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


### General functions

def register_plate(meta):
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
            nuclear_channel=0
        )

def segment_plate(meta):
    """
    Run Cellpose segmentation per well/position/time, producing nuclear
    (channel 0) and cytoplasmic-sum (channels 2+) masks.
    Works for any metadata object exposing wells, P, T, Z, H, W, zarrfile.
    """
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
            z[well]['images'][t, p, 0, 0],
            cellprob_threshold=1.5
        )
        z[well]['masks'][t, p, 0, 1], _, _ = model.eval(
            np.sum(z[well]['images'][t, p, 0, 2:], axis=0),
            cellprob_threshold=1.5
        )

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
