"""Rigid-body coregistration of the T2 image to the T1 image space.

The MATLAB pipeline calls SPM's ``spm_coreg`` (normalised mutual information,
followed by a B-spline reslice).  The same scheme is implemented here directly:
NMI is maximised over the six rigid-body parameters on progressively finer
sampling grids, then the source volume is resampled onto the reference grid.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from scipy import ndimage, optimize

from ..io.nifti import NiftiVolume
from ..utils.logging import get_logger
from .resample import reslice_to_grid

__all__ = ["rigid_matrix", "coregister", "realign_t2"]

logger = get_logger()
_SAMPLING = (4.0, 2.0)          # SPM's eoptions.sep
_BINS = 64


def rigid_matrix(params) -> np.ndarray:
    """Build a rigid-body transform from ``[tx ty tz rx ry rz]`` (radians)."""
    tx, ty, tz, rx, ry, rz = (float(p) for p in params)
    translate = np.eye(4)
    translate[:3, 3] = (tx, ty, tz)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rot_x = np.array([[1, 0, 0, 0], [0, cx, sx, 0], [0, -sx, cx, 0], [0, 0, 0, 1.0]])
    rot_y = np.array([[cy, 0, sy, 0], [0, 1, 0, 0], [-sy, 0, cy, 0], [0, 0, 0, 1.0]])
    rot_z = np.array([[cz, sz, 0, 0], [-sz, cz, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])
    return translate @ rot_x @ rot_y @ rot_z


def _normalised_mutual_information(a, b, bins=_BINS):
    """NMI of two intensity vectors; larger is better."""
    joint, _, _ = np.histogram2d(a, b, bins=bins)
    joint = ndimage.gaussian_filter(joint, 1.0)      # SPM smooths the histogram
    total = joint.sum()
    if total <= 0:
        return 0.0
    joint /= total
    p_a = joint.sum(axis=1)
    p_b = joint.sum(axis=0)

    def entropy(p):
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum())

    h_joint = entropy(joint.ravel())
    if h_joint == 0:
        return 0.0
    return (entropy(p_a) + entropy(p_b)) / h_joint


def coregister(ref: NiftiVolume, src: NiftiVolume, separations=_SAMPLING) -> np.ndarray:
    """Estimate the rigid transform (in world space) mapping ``src`` onto ``ref``."""
    ref_geom, src_geom = ref.geom(), src.geom()
    ref_data = np.asarray(ref.img, dtype=float)
    src_data = np.asarray(src.img, dtype=float)

    params = np.zeros(6)
    for sep in separations:
        step = np.maximum((sep / np.abs(ref_geom.voxel_size)).astype(int), 1)
        grid = np.meshgrid(*[np.arange(1, n + 1, s, dtype=float)
                             for n, s in zip(ref_geom.dim, step)], indexing="ij")
        sample_voxels = np.vstack([g.ravel() for g in grid]
                                  + [np.ones(grid[0].size)])
        ref_values = ndimage.map_coordinates(ref_data, sample_voxels[:3] - 1.0,
                                             order=1, mode="constant", cval=0.0)
        world = ref_geom.mat @ sample_voxels

        def cost(p, world=world, ref_values=ref_values):
            moved = np.linalg.solve(src_geom.mat, rigid_matrix(p) @ world)
            values = ndimage.map_coordinates(src_data, moved[:3] - 1.0, order=1,
                                             mode="constant", cval=0.0)
            return -_normalised_mutual_information(ref_values, values)

        result = optimize.minimize(cost, params, method="Powell",
                                   options={"xtol": 1e-3, "ftol": 1e-4, "maxiter": 40})
        params = result.x
    return rigid_matrix(params)


def realign_t2(src, ref):
    """Align the T2 image to the T1 image space, when it is not already aligned.

    Returns the path of the image ROAST should use (the original when the two
    already share a grid).
    """
    src, ref = Path(src), Path(ref)
    ref_volume = NiftiVolume.load(ref)
    src_volume = NiftiVolume.load(src)

    def same_grid(a: NiftiVolume, b: NiftiVolume) -> bool:
        return (np.array_equal(np.asarray(a.header["dim"])[:5],
                               np.asarray(b.header["dim"])[:5])
                and np.array_equal(a.pixdim[:4], b.pixdim[:4])
                and np.allclose(a.srow(), b.srow()))

    if same_grid(ref_volume, src_volume):
        return src

    out_path = src.parent / (ref.stem + "_T2_aligned" + src.suffix)
    if out_path.exists() and same_grid(ref_volume, NiftiVolume.load(out_path)):
        logger.warning("Original T2 %s is not aligned to T1 image space, but an aligned "
                       "T2 has been found (%s). ROAST will use that file as the input.",
                       src, out_path)
        return out_path

    logger.warning("T2 image %s is not aligned to T1 image space. ROAST will align it now.",
                   src)
    logger.info("Aligning T2 to T1...")

    # Keep a copy under the reference's name, mirroring the MATLAB pipeline which
    # lets SPM rewrite the header of its working copy.
    working_copy = src.parent / (ref.stem + "_T2" + src.suffix)
    if working_copy.resolve() != src.resolve():
        shutil.copyfile(src, working_copy)

    transform = coregister(ref_volume, src_volume)
    ref_geom, src_geom = ref_volume.geom(), src_volume.geom()
    resampled = reslice_to_grid(src_volume, transform @ src_geom.mat,
                                ref_geom.mat, ref_geom.dim)

    out = src_volume.copy()
    dtype = np.dtype(out.header.get_data_dtype())
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        resampled = np.clip(np.round(resampled), info.min, info.max)
    out.img = resampled.astype(dtype)
    dims = np.asarray(out.header["dim"]).copy()
    dims[1:4] = ref_geom.dim
    out.header["dim"] = dims
    out.header["pixdim"] = ref_volume.pixdim
    out.set_srow(ref_volume.srow(), force_sform=True)
    out.save(out_path)

    logger.info("Aligned T2 is saved as %s, and will be used by ROAST.", out_path)
    return out_path
