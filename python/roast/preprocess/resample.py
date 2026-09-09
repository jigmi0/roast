"""Resampling of the MRI to 1 mm isotropic resolution.

This replaces ``spm_reslice``: the target grid is derived from the bounding box
of the input exactly as ``spm_get_bbox`` does, and the volume is resampled with
a high-order B-spline.  SPM offers a 7th-degree spline; SciPy tops out at 5,
which is the order used here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage

from ..io.nifti import NiftiVolume, spm_mat_to_srow
from ..utils.logging import get_logger

__all__ = ["bounding_box", "reslice_to_grid", "resamp_to_one_mm"]

logger = get_logger()
_SPLINE_ORDER = 5


def bounding_box(mat: np.ndarray, dim) -> np.ndarray:
    """World-space bounding box of a volume (port of ``spm_get_bbox(V, 'fv')``)."""
    d1, d2, d3 = (int(v) for v in dim)
    corners = np.array([
        [1, 1, 1], [1, 1, d3], [1, d2, 1], [1, d2, d3],
        [d1, 1, 1], [d1, 1, d3], [d1, d2, 1], [d1, d2, d3],
    ], dtype=float)
    world = (np.asarray(mat, dtype=float)
             @ np.column_stack([corners, np.ones(len(corners))]).T)[:3].T
    return np.vstack([world.min(axis=0), world.max(axis=0)])


def reslice_to_grid(volume: NiftiVolume, source_mat: np.ndarray,
                    target_mat: np.ndarray, target_dim) -> np.ndarray:
    """Sample ``volume`` on the voxel grid described by ``target_mat``/``target_dim``.

    Both matrices follow the SPM convention (1-based voxel to world).
    """
    target_dim = np.asarray(target_dim, dtype=int)
    grid = np.meshgrid(*[np.arange(1, n + 1, dtype=float) for n in target_dim],
                       indexing="ij")
    ones = np.ones(grid[0].size)
    target_voxels = np.vstack([g.ravel() for g in grid] + [ones])
    source_voxels = np.linalg.solve(np.asarray(source_mat, dtype=float),
                                    np.asarray(target_mat, dtype=float) @ target_voxels)
    coords = source_voxels[:3] - 1.0            # SciPy indexes from zero
    data = np.asarray(volume.img, dtype=float)
    resampled = ndimage.map_coordinates(data, coords, order=_SPLINE_ORDER,
                                        mode="constant", cval=0.0, prefilter=True)
    return resampled.reshape(target_dim)


def resamp_to_one_mm(mri, do_resamp: bool):
    """Resample ``mri`` to 1 mm isotropic resolution when requested.

    Returns ``(path, do_resamp)``; ``do_resamp`` is cleared when the input
    already has 1 mm isotropic voxels.
    """
    mri = Path(mri)
    if not do_resamp:
        return mri, False

    volume = NiftiVolume.load(mri)
    geom = volume.geom()
    pixdim = np.array([geom.mat[0, 0], geom.mat[1, 1], geom.mat[2, 2]])

    if np.all(np.abs(pixdim) == 1):
        logger.warning("The MRI %s already has a 1 mm isotropic resolution. "
                       "No need to resample.", mri)
        return mri, False

    out_path = mri.parent / (mri.stem + "_1mm" + mri.suffix)
    if out_path.exists():
        logger.warning("%s has already been resampled to 1mm resolution and saved as %s. "
                       "ROAST will use that file as the input.", mri, out_path)
        return out_path, True

    logger.info("Resampling %s to 1 mm isotropic resolution...", mri)
    voxel_size = np.array([1.0, 1.0, 1.0])
    bbox = bounding_box(geom.mat, geom.dim)

    target_mat = np.eye(4)
    target_mat[:3, :3] = np.diag(voxel_size)
    target_mat[:3, 3] = bbox[0] - voxel_size
    target_dim = np.ceil(np.linalg.solve(target_mat,
                                         np.append(bbox[1], 1.0))[:3] - 0.1).astype(int)

    resampled = reslice_to_grid(volume, geom.mat, target_mat, target_dim)

    out = volume.copy()
    dtype = np.dtype(out.header.get_data_dtype())
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        resampled = np.clip(np.round(resampled), info.min, info.max)
    out.img = resampled.astype(dtype)
    dims = np.asarray(out.header["dim"]).copy()
    dims[1:4] = target_dim
    out.header["dim"] = dims
    pix = out.pixdim.copy()
    pix[1:4] = voxel_size
    out.header["pixdim"] = pix
    out.set_srow(spm_mat_to_srow(target_mat), force_sform=True)
    out.save(out_path)

    logger.info("%s has been resampled to 1 mm isotropic resolution, and is saved as:\n%s\n"
                "It'll be used as the input for ROAST.", mri, out_path)
    return out_path, True
