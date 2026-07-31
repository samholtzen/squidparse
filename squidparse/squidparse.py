import numpy as np
from skimage.registration import phase_cross_correlation
import yaml
import zarr
import tifffile
import datetime
from pathlib import Path
import pandas as pd
from tqdm import tqdm
# from cellpose import models
import itertools
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

class TimeLapseMetadata():

    def __init__(self, expdir):

        self.expdir = Path(expdir)
        self.exp_name = self.expdir.stem
        self._extract_experiment_metadata()
        self._make_zarrfile_name()

    def __repr__(self):
        
        printstring = f'Experiment directory: {self.expdir}\nChannels: {self.channels}\nImage dimensions: {self._size()}'

        return printstring
    
    def _extract_experiment_metadata(self):

        expdir = Path(self.expdir)
        self.frame_dirs = [frame for frame in expdir.iterdir() if frame.is_dir()]

        with open(expdir / 'acquisition.yaml') as file:
            meta = yaml.safe_load(file)

        self.T = len(self.frame_dirs)

        coord_df = pd.read_csv(expdir / 'coordinates.csv')
        wells_counts = coord_df['region'].value_counts()

        self.wells = wells_counts.keys().to_list()
        self.P = wells_counts.values[0].item()

        self.Z = meta['z_stack']['nz']
        self.channels = [channel['name'] for channel in meta['channels']]

        self.C = len(self.channels)

        self._get_image_size()

        self.metadata = meta

    def _make_zarrfile_name(self):
        exp_datetime = datetime.datetime.fromtimestamp(self.metadata['acquisition']['start_time'])
        self.zarrfile = exp_datetime.strftime('%Y%m%d') + '.zarr'
        
    def _size(self):

        return (self.T, self.P, self.Z, self.C, self.H, self.W)

    def path_dataframe(self):

        all_coords = itertools.product(self.wells, range(self.T), range(self.P), range(self.Z), range(self.C))
        df = pd.DataFrame(all_coords, columns=['well','T','P','Z','C'])

        files = []
        for i, row in df.iterrows():
            files.append(self.image_path(row['well'], row['T'], row['P'], row['Z'], row['C']))
        
        df['image_dir'] = files

        def _img_exist(file):
            return Path(file).exists()
        
        df['exist'] = df['image_dir'].map(_img_exist)
        
        return df

    def _get_image_size(self):

        temp_path = self.image_path(self.wells[0], 0, 0, 0, 0)
        with tifffile.TiffFile(temp_path) as tif:
            self.H, self.W = tif.shaped_metadata[0]['shape']

    def image_path(self, well, T, P, Z, C):
        
        check_coord = (T <= self.T) & (P <= self.P) & (Z <= self.Z) & (C <= self.C) & (well in self.wells)

        if not check_coord:
            raise IndexError
        
        # get directory

        frame_dir = self.frame_dirs[T]
        channel_name = '_'.join(self.channels[C].split(' '))
        
        filedir = frame_dir / f'{well}_{P}_{Z}_{channel_name}.tiff'

        return filedir



class ISSMetadata():

    def __init__(self, expdir):

        self.expdir = Path(expdir)
        self.exp_name = self.expdir.stem
        self.zarrfile = f"{self.exp_name}.zarr"
        self._extract_experiment_metadata()
        self._get_round_attrs()

    def __repr__(self):
        
        printstring = f'Experiment directory: {self.expdir}\nChannels: {self.channels}\nImage dimensions: {self._size()}'

        return printstring

    def plot_fov_well_location(self):
        fig, ax = plt.subplots()


        return fig
    
    def _extract_experiment_metadata(self):

        expdir = Path(self.expdir)
        self.round_dirs = [round for round in expdir.iterdir()]
        test_dir = self.round_dirs[0]

        with open(test_dir / 'acquisition.yaml') as file:
            meta = yaml.safe_load(file)

        self.T = len(self.round_dirs)

        coord_df = pd.read_csv(test_dir / 'coordinates.csv')
        wells_counts = coord_df['region'].value_counts()

        self.wells = wells_counts.keys().to_list()
        self.P = wells_counts.values[0].item()

        self.Z = meta['z_stack']['nz']
        self.channels = [channel['name'] for channel in meta['channels']]

        self.C = len(self.channels)

        self._get_image_size()

    def _size(self):

        return (self.T, self.P, self.Z, self.C, self.H, self.W)

    def path_dataframe(self):

        all_coords = itertools.product(self.wells, range(self.T), range(self.P), range(self.Z), range(self.C))
        df = pd.DataFrame(all_coords, columns=['well','T','P','Z','C'])

        files = []
        for i, row in df.iterrows():
            files.append(self.image_path(row['well'], row['T'], row['P'], row['Z'], row['C']))
        
        df['image_dir'] = files

        def _img_exist(file):
            return Path(file).exists()
        
        df['exist'] = df['image_dir'].map(_img_exist)
        
        return df

    def _get_round_attrs(self):

        round_info = {}

        # Make sure we are processing them in order by sorting on start time for experiments!
        for round in self.round_dirs:

            with open(round / 'acquisition.yaml') as file:
                meta = yaml.safe_load(file)

            round_info[str(round)] = meta['acquisition']['start_time']
        
        sorted_rounds = sorted(round_info, key=round_info.get)

        self.round_meta = []

        for round in sorted_rounds:

            with open(Path(round) / 'acquisition.yaml') as file:
                meta = yaml.safe_load(file)
            
            self.round_meta.append(meta)

    def _get_image_size(self):

        temp_path = self.image_path(self.wells[0], 0, 0, 0, 0)
        with tifffile.TiffFile(temp_path) as tif:
            self.H, self.W = tif.shaped_metadata[0]['shape']

    def image_path(self, well, T, P, Z, C):
        
        check_coord = (T <= self.T) & (P <= self.P) & (Z <= self.Z) & (C <= self.C) & (well in self.wells)

        if not check_coord:
            raise IndexError
        
        # get directory

        round_dir = self.round_dirs[T]
        pseudotime = 0
        channel_name = '_'.join(self.channels[C].split(' '))
        
        filedir = round_dir / str(pseudotime) / f'{well}_{P}_{Z}_{channel_name}.tiff'

        return filedir

### ISS specific functions

def make_iss_zarr(meta: ISSMetadata):    

    z = zarr.open(meta.zarrfile)

    z.attrs['acquisition'] = meta.round_meta

    for well in meta.wells:

        well_group = z.create_group(name=well)
        well_group.zeros(
            name='images',
            shape=(meta.T, meta.P, meta.Z, meta.C, meta.H, meta.W),
            chunks=(1, 1, meta.Z, 1, meta.H, meta.W),
            dtype=np.uint16
        )

    df = meta.path_dataframe()

    for _, row in tqdm(df.iterrows()):

        _well, _t, _p, _z, _c, _path, _ = row.values
        z[_well]['images'][_t, _p, _z, _c, :, :] = tifffile.imread(_path)

### Movie specific functions

def make_movie_zarr(meta: TimeLapseMetadata, exp_name = None):    

    z = zarr.open(meta.zarrfile)

    z.attrs['acquisition'] = meta.metadata

    for well in meta.wells:

        well_group = z.create_group(name=well)
        well_group.zeros(
            name='images',
            shape=(meta.T, meta.P, meta.Z, meta.C, meta.H, meta.W),
            chunks=(1, 1, meta.Z, 1, meta.H, meta.W),
            dtype=np.uint16
        )

    df = meta.path_dataframe()

    for _, row in tqdm(df.iterrows()):

        _well, _t, _p, _z, _c, _path, _ = row.values
        z[_well]['images'][_t, _p, _z, _c, :, :] = tifffile.imread(_path)


### General functions

def register_plate(meta):

    z = zarr.open(meta.zarrfile)

    for well in meta.wells:
        z[well].zeros(name='offsets', shape=(meta.P, meta.T, 2), dtype=int)
    
    for well, _p in tqdm(itertools.product(meta.wells, range(meta.P))):
        z[well]['offsets'][_p, :, :] = calculate_shift(
            z[well]['images'][:, _p, 0, :1, :, :],
            nuclear_channel=0
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

# def segment_plate(meta):

#     z = zarr.open(meta.zarrfile)
#     model = models.CellposeModel(gpu=True)

#     for well in meta.wells:

#         arr = z[well].zeros(
#             name = 'masks',
#             shape = (meta.T, meta.P, meta.Z, 2, meta.H, meta.W),
#             chunks = (1, 1, meta.Z, 1, meta.H, meta.W),
#             dtype=np.uint16
#         )
    
#     for well, _p, _t in tqdm(itertools.product(meta.wells, range(meta.P), range(meta.T))):
#         z[well]['masks'][_t, _p, 0, 0], _, _ = model.eval(
#             z[well]['images'][_t, _p, 0, 0],
#             cellprob_threshold=1.5
#         )

#         z[well]['masks'][_t, _p, 0, 1], _, _ = model.eval(
#             np.sum(z[well]['images'][_t, _p, 0, 2:], axis=0)
#         )

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
