# Imaging Pipeline

## Overview

This pipeline reads microscopy image data from disk, stores it in Zarr format, and processes it. The pipeline supports two experiment types:

- A time-lapse experiment. This is one acquisition folder with one frame subfolder for each time point.
- An ISS experiment. This is a parent folder that contains many round folders. Each round folder is one acquisition folder.

## Folder Structure

### Single Acquisition Folder

A single acquisition folder has this structure:

```
<acqdir>/
    0/, 1/, 2/, ...          <- frame subfolders (T dimension)
    acquisition.yaml
    coordinates.csv
```

Each numbered subfolder holds the images for one time point. The file `acquisition.yaml` holds acquisition metadata. The file `coordinates.csv` holds well and position data.

### ISS Experiment Folder

An ISS experiment folder has this structure:

```
<expdir>/
    Round_1_<timestamp>/     <- one acquisition folder
    Round_2_<timestamp>/     <- one acquisition folder
    ...
```

Each round folder is a single acquisition folder, as described above. In an ISS experiment, each round has exactly one frame subfolder. The pipeline treats each round as one time point.

## Metadata Classes

### AcquisitionMetadata

`AcquisitionMetadata` reads one acquisition folder. It reads the well list, the position count, the Z count, and the channel list. It also finds the image size.

Use this class for a single time-lapse experiment through the `TimeLapseMetadata` subclass.

### TimeLapseMetadata

`TimeLapseMetadata` extends `AcquisitionMetadata`. It adds a Zarr file name based on the acquisition start time.

Create a `TimeLapseMetadata` object with one call:

```python
meta = TimeLapseMetadata(expdir)
```

### ISSMetadata

`ISSMetadata` reads an ISS experiment folder. It creates one `AcquisitionMetadata` object for each round folder. It sorts the rounds by acquisition start time. It checks that all rounds share the same wells, channels, and Z count. If a round does not match, `ISSMetadata` raises a `ValueError`.

Create an `ISSMetadata` object with one call:

```python
meta = ISSMetadata(expdir)
```

## Common Interface

Both `TimeLapseMetadata` and `ISSMetadata` expose the same attributes and methods. This lets the pipeline functions work with either class.

Attributes:
- `wells`: list of well names
- `T`, `P`, `Z`, `C`, `H`, `W`: array dimensions
- `zarrfile`: output Zarr file name
- `zarr_attrs`: metadata to store in the Zarr file

Methods:
- `image_path(well, T, P, Z, C)`: return the file path for one image
- `path_dataframe()`: return a table of every image coordinate and path
- `_size()`: return the shape `(T, P, Z, C, H, W)`

## Pipeline Functions

### make_zarr(meta)

`make_zarr` builds a Zarr store from a metadata object. It creates one group for each well. Each group holds an `images` array. The function reads each TIFF file and writes it into the array.

If an expected image file does not exist, `make_zarr` raises a `FileNotFoundError`.

```python
make_zarr(meta)
```

### register_plate(meta)

`register_plate` finds the frame-to-frame jitter for each well and position. It uses the nuclear channel to estimate the shift. It stores the shifts in an `offsets` array in the Zarr store.

```python
register_plate(meta)
```

### calculate_shift(movie, nuclear_channel, channel_axis=1, time_axis=0, reference="first")

`calculate_shift` estimates the pixel shift between frames in a movie. It uses phase cross-correlation on the nuclear channel.

Set `reference` to `"first"` to compare each frame to the first frame. Set `reference` to `"previous"` to compare each frame to the frame before it. Use `"first"` for slow drift. Use `"previous"` for large cumulative drift.

The function returns an array of shape `(T, n_spatial_dims)`.

### crop_to_common_overlap(frames, offsets)

`crop_to_common_overlap` crops a stack of frames to the area that is in view at every time point. It uses the offsets from `calculate_shift`.

If the offsets are larger than the frame, the function raises a `ValueError`.

### segment_plate(meta)

`segment_plate` runs Cellpose on each well, position, and time point. It creates two masks for each image: a nuclear mask and a cytoplasm mask. It stores the masks in a `masks` array in the Zarr store.

```python
segment_plate(meta)
```

### voronoi_mask(centroids, shape)

`voronoi_mask` builds a Voronoi label mask from a list of centroids. For each pixel, the function finds the nearest centroid. The function returns an array of centroid indices with shape `shape`.

## Typical Workflow

1. Create a metadata object for your experiment.
2. Call `make_zarr(meta)` to build the Zarr store from the source images.
3. Call `register_plate(meta)` to compute frame alignment offsets.
4. Call `segment_plate(meta)` to compute nuclear and cytoplasm masks.
5. Use `crop_to_common_overlap` and `voronoi_mask` as needed for downstream analysis.

Example for a time-lapse experiment:

```python
meta = TimeLapseMetadata(expdir)
make_zarr(meta)
register_plate(meta)
segment_plate(meta)
```

Example for an ISS experiment:

```python
meta = ISSMetadata(expdir)
make_zarr(meta)
register_plate(meta)
segment_plate(meta)
```

## Dependencies

- numpy
- scikit-image
- pyyaml
- zarr
- tifffile
- pandas
- tqdm
- cellpose
- scipy
- matplotlib