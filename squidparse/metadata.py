import re
import numpy as np
import yaml
import tifffile
import datetime
import pandas as pd
import itertools
import zarr
from pathlib import Path


def _is_zarr(path):
    return Path(path).suffix == '.zarr'


def _read_zarr_info(zarrpath):
    """
    Recover the descriptive info normally parsed from a raw acquisition
    folder, from a zarr written by make_zarr. Uses the 'squidparse' attrs
    when present; for older stores falls back to the raw 'acquisition' attrs
    plus the shapes of the stored arrays.
    """
    zarrpath = Path(zarrpath)
    if not zarrpath.exists():
        raise FileNotFoundError(f"Zarr store not found: {zarrpath}")
    z = zarr.open(str(zarrpath), mode='r')
    info = dict(z.attrs.get('squidparse', {}))
    acquisition = z.attrs.get('acquisition')
    info['acquisition'] = acquisition
    wells = info.get('wells') or [name for name, _ in z.groups()]
    well = z[wells[0]]
    fov_names = [k for k, _ in well.groups()]
    # stored images are cropped to the registration overlap, so report their
    # shape (of the first FOV) rather than the raw frame size
    T, C, H, W = well[fov_names[0]]['images'].shape
    info.update(T=T, C=C, H=H, W=W)
    if 'wells' in info:
        return info

    # legacy store: reconstruct from acquisition attrs
    first = acquisition[0] if isinstance(acquisition, list) else acquisition
    info.update(
        wells=wells, P=len(fov_names),
        Z=first['z_stack']['nz'],
        channels=[c['name'] for c in first['channels']],
        mpp=first['objective']['pixel_size_um'],
    )
    return info


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

    def __init__(
            self, 
            acqdir,
            pseudobin=None,
            nuc_channel=None,
            cyto_channel=None,
            stitch=False,
    ):
        """
        acqdir may be a raw acquisition folder or a zarr store written by
        make_zarr. For a zarr, descriptive info is read from the store and
        unspecified (None) pseudobin/nuc_channel/cyto_channel take the
        stored values; image_path()/path_dataframe() are unavailable.

        stitch=True stitches the P overlapping FOVs of each well into one mosaic when
        make_zarr runs (before registration). The metadata then reports P = 1, since the
        zarr holds a single stitched FOV '0' per well; `n_tiles` keeps the raw FOV count.
        """
        self.acqdir = Path(acqdir)
        self.from_zarr = _is_zarr(self.acqdir)

        if self.from_zarr:
            self._extract_from_zarr(pseudobin, nuc_channel, cyto_channel)
        else:
            self.pseudobin = bool(pseudobin)
            self.nuc_channel = 0 if nuc_channel is None else nuc_channel
            self.cyto_channel = 1 if cyto_channel is None else cyto_channel
            self.stitch = bool(stitch)
            self._extract_metadata()
            self.n_tiles = self.P
            if self.stitch:
                self.P = 1
            self._make_zarrfile_name()

    def _extract_from_zarr(self, pseudobin, nuc_channel, cyto_channel):
        info = _read_zarr_info(self.acqdir)
        acq = info['acquisition']
        if isinstance(acq, list):
            raise ValueError(f"{self.acqdir} holds multiple rounds; use ISSMetadata")

        self.metadata = acq
        self.wells = info['wells']
        self.T, self.P, self.Z, self.C = info['T'], info['P'], info['Z'], info['C']
        self.H, self.W = info['H'], info['W']
        self.channels = info['channels']
        self.mpp = info['mpp']
        self.pseudobin = info.get('pseudobin', False) if pseudobin is None else pseudobin
        self.nuc_channel = info.get('nuc_channel', 0) if nuc_channel is None else nuc_channel
        self.cyto_channel = info.get('cyto_channel', 1) if cyto_channel is None else cyto_channel
        self.stitch = info.get('stitch', False)
        self.n_tiles = info.get('n_tiles', self.P)
        self.zarrfile = str(self.acqdir)

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
        self.coord_df = coord_df
        wells_counts = coord_df['region'].value_counts()
        self.wells = wells_counts.keys().to_list()
        self.P = wells_counts.values[0].item()
        self.n_tiles = self.P

        self.Z = self.metadata['z_stack']['nz']
        self.channels = [c['name'] for c in self.metadata['channels']]
        self.C = len(self.channels)

        if self.pseudobin:
            self.mpp = self.metadata['objective']['pixel_size_um']*2
        else:
            self.mpp = self.metadata['objective']['pixel_size_um']

        self._get_image_size()

    @property
    def start_time(self):
        return self.metadata['acquisition']['start_time']

    @property
    def zarr_attrs(self):
        return self.metadata

    @property
    def zarr_info(self):
        """Everything needed to rebuild this object from the zarr alone."""
        return dict(
            kind='acquisition',
            wells=self.wells, T=self.T, P=self.P, Z=self.Z, C=self.C,
            H=self.H, W=self.W, channels=self.channels, mpp=self.mpp,
            pseudobin=self.pseudobin, nuc_channel=self.nuc_channel,
            cyto_channel=self.cyto_channel, stitch=self.stitch, n_tiles=self.n_tiles,
        )

    def fov_positions(self, well):
        """Stage (y_um, x_um) of each of the well's FOVs, in P order (from coordinates.csv)."""
        rows = self.coord_df[self.coord_df['region'] == well]
        return rows['y (mm)'].to_numpy() * 1000, rows['x (mm)'].to_numpy() * 1000

    def _get_image_size(self):

        temp_path = self.image_path(self.wells[0], 0, 0, 0, 0)

        with tifffile.TiffFile(temp_path) as tif:
            H, W= tif.shaped_metadata[0]['shape']

        if self.pseudobin:
            self.H, self.W = H//2, W//2
        else:
            self.H, self.W = H, W

    def _make_zarrfile_name(self):
        exp_dt = datetime.datetime.fromtimestamp(self.start_time)
        self.zarrfile = exp_dt.strftime('%Y%m%d%H%M%S') + '.zarr'

    def _size(self):
        return (self.T, self.P, self.Z, self.C, self.H, self.W)

    def image_path(self, well, T, P, Z, C):
        if self.from_zarr:
            raise RuntimeError("Raw image paths are unavailable: metadata was loaded from a zarr")
        if not (0 <= T < self.T and 0 <= P < self.n_tiles and 0 <= Z < self.Z
                and 0 <= C < self.C and well in self.wells):
            raise IndexError(f"Coordinate out of range: {well, T, P, Z, C}")

        frame_dir = self.frame_dirs[T]
        channel_name = '_'.join(self.channels[C].split(' '))
        return frame_dir / f'{well}_{P}_{Z}_{channel_name}.tiff'

    def path_dataframe(self):
        if self.from_zarr:
            raise RuntimeError("Raw image paths are unavailable: metadata was loaded from a zarr")
        all_coords = itertools.product(
            self.wells, range(self.T), range(self.n_tiles), range(self.Z), range(self.C)
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
    def __init__(self, expdir, pseudobin=None, nuc_channel=None, cyto_channel=None, stitch=False):
        super().__init__(expdir, pseudobin=pseudobin,
                         nuc_channel=nuc_channel, cyto_channel=cyto_channel, stitch=stitch)
        self.expdir = self.acqdir
        self.exp_name = self.expdir.stem
        if not self.from_zarr:
            self._make_zarrfile_name()


class ISSMetadata:
    def __init__(
            self, 
            expdir, 
            pseudobin=None,
            nuc_channel=None,
            cyto_channel=None
    ):
        """
        expdir may be a raw experiment folder (one subfolder per round) or a
        zarr store written by make_zarr; see AcquisitionMetadata.
        """
        self.expdir = Path(expdir)
        self.exp_name = self.expdir.stem
        self.from_zarr = _is_zarr(self.expdir)

        if self.from_zarr:
            self._extract_from_zarr(pseudobin, nuc_channel, cyto_channel)
            return

        self.pseudobin = bool(pseudobin)
        self.zarrfile = f"{self.exp_name}.zarr"
        self.nuc_channel = 0 if nuc_channel is None else nuc_channel
        self.cyto_channel = 1 if cyto_channel is None else cyto_channel

        # parse every round as its own AcquisitionMetadata (T will be 1 for each,
        # since a round folder only ever contains the single '0' frame dir)
        round_dirs = [d for d in self.expdir.iterdir() if d.is_dir()]
        rounds = [AcquisitionMetadata(d, pseudobin=self.pseudobin) for d in round_dirs]

        # sort rounds chronologically by acquisition start time
        self.rounds = sorted(rounds, key=lambda r: r.start_time)
        self.round_dirs = [r.acqdir for r in self.rounds]
        self.round_meta = [r.metadata for r in self.rounds]

        self.T = len(self.rounds)  # rounds ARE the time dimension for ISS

        # validate consistency across rounds and pull shared attrs from round 0
        self._validate_and_set_shared_attrs()

    def _extract_from_zarr(self, pseudobin, nuc_channel, cyto_channel):
        info = _read_zarr_info(self.expdir)
        acq = info['acquisition']
        if not isinstance(acq, list):
            raise ValueError(f"{self.expdir} holds a single acquisition; use AcquisitionMetadata")

        self.round_meta = acq
        self.rounds = None  # per-round AcquisitionMetadata only exist for raw folders
        self.round_dirs = None
        self.start_times = [m['acquisition']['start_time'] for m in acq]
        self.T, self.P, self.Z, self.C = info['T'], info['P'], info['Z'], info['C']
        self.H, self.W = info['H'], info['W']
        self.wells, self.channels, self.mpp = info['wells'], info['channels'], info['mpp']
        self.pseudobin = info.get('pseudobin', False) if pseudobin is None else pseudobin
        self.nuc_channel = info.get('nuc_channel', 0) if nuc_channel is None else nuc_channel
        self.cyto_channel = info.get('cyto_channel', 1) if cyto_channel is None else cyto_channel
        self.zarrfile = str(self.expdir)

    def _validate_and_set_shared_attrs(self):
        ref = self.rounds[0]
        for r in self.rounds[1:]:
            if r.channels != ref.channels or r.wells != ref.wells or r.Z != ref.Z:
                raise ValueError(f"Round {r.acqdir} metadata does not match round {ref.acqdir}")

        self.wells, self.P, self.Z = ref.wells, ref.P, ref.Z
        self.channels, self.C = ref.channels, ref.C
        self.H, self.W = ref.H, ref.W
        self.mpp = ref.mpp
        self.start_times = [r.start_time for r in self.rounds]

    def _size(self):
        return (self.T, self.P, self.Z, self.C, self.H, self.W)

    def image_path(self, well, T, P, Z, C):
        if self.from_zarr:
            raise RuntimeError("Raw image paths are unavailable: metadata was loaded from a zarr")
        # T indexes which round; each round's own internal T is always 0
        return self.rounds[T].image_path(well, 0, P, Z, C)
    
    @property
    def zarr_attrs(self):
        return self.round_meta

    def path_dataframe(self):
        if self.from_zarr:
            raise RuntimeError("Raw image paths are unavailable: metadata was loaded from a zarr")
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