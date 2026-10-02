"""Stitch the overlapping FOVs of a well into one mosaic with TileFusion.

Called from `make_zarr` / `make_zarr_nd2` when the metadata was created with `stitch=True`,
before jitter registration, so the zarr stores the stitched mosaic as the well's single FOV.
Needs the optional `tilefusion` package (`pip install ".[stitch]"`).
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np


def _write_tilefusion_input(tiles, ys_um, xs_um, mpp, out_dir):
    """Write `tiles` (P, C, H, W) as the (1, P, C, H, W) zarr3 store TileFusion reads,
    with per-tile stage positions in `per_index_metadata`."""
    import tensorstore as ts

    shutil.rmtree(out_dir, ignore_errors=True)
    n_p, n_c, H, W = tiles.shape

    per_index = {"0": {str(p): {"0": {"stage_position": [0.0, float(ys_um[p]), float(xs_um[p])]}}
                       for p in range(n_p)}}
    config = {
        "driver": "zarr3",
        "kvstore": {"driver": "file", "path": str(out_dir)},
        "metadata": {
            "shape": [1, n_p, n_c, H, W],
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [1, 1, 1, H, W]}},
            "chunk_key_encoding": {"name": "default"},
            "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
            "data_type": "uint16",
            "dimension_names": ["t", "p", "c", "y", "x"],
            "attributes": {
                "per_index_metadata": per_index,
                "deskewed_voxel_size_um": [1.0, mpp, mpp],
                "channels": [f"ch{i}" for i in range(n_c)],
            },
        },
    }
    store = ts.open(config, create=True, open=True).result()
    store[0].write(tiles).result()


def _fuse(input_dir, fused_dir, nuc_channel, blend_pixels):
    """Run TileFusion on `input_dir`; return ((C, Y, X) mosaic, (P, 2) tile positions in µm).

    Registration is cached in `metrics.json` next to `input_dir` (TileFusion.run() loads it
    when present), so only the first call per well registers the tiles.
    """
    import tensorstore as ts
    from tilefusion import TileFusion

    shutil.rmtree(fused_dir, ignore_errors=True)
    # multiscale_factors=(): skip the pyramid, only full resolution is needed
    with TileFusion(input_dir, output_path=fused_dir, channel_to_use=nuc_channel,
                    blend_pixels=blend_pixels, multiscale_factors=()) as tf:
        tf.run()
        positions = np.array(tf.tile_positions, dtype=np.float64)
        sy, sx = tf.unpadded_shape

    fused = ts.open({"driver": "zarr3", "kvstore": {"driver": "file",
                     "path": str(Path(fused_dir) / "scale0" / "image")}}).result()
    mosaic = fused[0, :, 0, :sy, :sx].read().result()  # fused store is (T, C, Z, Y, X)
    return mosaic, positions


def stitch_tiles(tiles, ys_um, xs_um, mpp, nuc_channel=0, correction="tilefusion",
                 blend_pixels=(50, 50), work_dir=None):
    """Stitch the FOVs of one well, one time point at a time.

    tiles:      (T, P, C, H, W) uint16
    ys_um/xs_um: (P,) stage position of each FOV in µm
    mpp:        µm per pixel of `tiles`
    correction: "tilefusion" refines the stage positions by phase-correlation registration
                (done once, on the first time point, and reused for every frame);
                None / "nominal" fuses at the raw stage positions.
    work_dir:   where temporary stores go (default: system temp); removed afterwards.

    Returns (mosaic (T, C, Y, X) uint16, origin_px (P, 2) of each FOV as (row, col) in the mosaic).
    """
    T, P = tiles.shape[:2]
    if P == 1:
        return tiles[:, 0], np.zeros((1, 2))

    nominal = correction is None or str(correction).lower() in ("none", "nominal", "null")

    with tempfile.TemporaryDirectory(dir=work_dir) as tmp:
        input_dir, fused_dir = Path(tmp, 'tf_input.zarr'), Path(tmp, 'fused.ome.zarr')
        if nominal:
            # an empty metrics.json makes run() skip registration and keep the stage positions
            Path(tmp, 'metrics.json').write_text('{}')

        for t in range(T):
            _write_tilefusion_input(tiles[t], ys_um, xs_um, mpp, input_dir)
            frame, positions = _fuse(input_dir, fused_dir, nuc_channel, blend_pixels)
            if t == 0:
                mosaic = np.zeros((T, *frame.shape), dtype=np.uint16)
                origin_px = (positions - positions.min(axis=0)) / mpp
            mosaic[t] = frame

    return mosaic, origin_px
