import yaml
import tifffile
import datetime
import pandas as pd
import itertools
from pathlib import Path


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