"""Re-orientation of the input MRI (and of point clouds) into RAS."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..io.nifti import NiftiVolume
from ..utils.logging import get_logger

__all__ = ["orientation_of", "convert_to_ras", "convert_to_ras_point_cloud"]

logger = get_logger()


def orientation_of(volume: NiftiVolume):
    """Work out how the volume is oriented with respect to RAS.

    Returns ``(matrix, perm_order, flip_tag)``: the voxel-to-world matrix, the
    permutation that brings the axes into RAS order (0-based) and the sign of
    each axis (negative meaning the volume is flipped along it).
    """
    matrix = volume.orientation_matrix()
    orient = matrix[:3, :3]
    axis_of = np.argmax(np.abs(orient), axis=0)      # dominant world axis per column
    perm_order = np.argsort(axis_of, kind="stable")
    flip_tag = np.array([np.sign(orient[axis_of[i], i]) for i in range(3)])
    return matrix, perm_order, flip_tag


def convert_to_ras(mri):
    """Re-orient an MRI into RAS if needed.

    Returns ``(path, is_non_ras)``; the re-oriented volume is written next to
    the input with an ``_ras`` suffix and re-used if it already exists.
    """
    mri = Path(mri)
    volume = NiftiVolume.load(mri)
    matrix, perm_order, flip_tag = orientation_of(volume)
    orient = matrix[:3, :3].copy()
    axis_of = np.argmax(np.abs(orient), axis=0)

    if np.array_equal(perm_order, [0, 1, 2]) and np.all(flip_tag > 0):
        return mri, False

    out_path = mri.parent / (mri.stem + "_ras" + mri.suffix)
    if out_path.exists():
        logger.warning("Input MRI %s is not in RAS orientation, and has been re-oriented "
                       "into RAS and saved as %s. ROAST will use that file as the input.",
                       mri, out_path)
        return out_path, True

    logger.warning("Input MRI %s is not in RAS orientation. ROAST will re-orient it "
                   "into RAS now...", mri)

    img = volume.img
    size = np.array(img.shape[:3], dtype=int)
    resolution = volume.resolution.copy()

    origin = np.round(np.linalg.inv(matrix) @ np.array([0.0, 0.0, 0.0, 1.0]) + 1)[:3]

    for i in range(3):
        if flip_tag[i] < 0:
            img = np.flip(img, axis=i)
            origin[i] = size[i] - origin[i] + 1
            orient[axis_of[i], i] = abs(orient[axis_of[i], i])

    img = np.transpose(img, perm_order)
    origin = origin[perm_order]
    orient = orient[:, perm_order]

    out = volume.copy()
    out.img = np.ascontiguousarray(img)
    dims = np.asarray(out.header["dim"]).copy()
    dims[1:4] = size[perm_order]
    out.header["dim"] = dims
    pixdim = out.pixdim.copy()
    pixdim[0] = abs(pixdim[0])
    pixdim[1:4] = resolution[perm_order]
    out.header["pixdim"] = pixdim

    srow = np.eye(4)
    srow[:3, :3] = orient
    srow[:3, 3] = -orient @ origin
    out.set_srow(srow)
    out.save(out_path)

    logger.info("%s is now in RAS orientation and saved as:\n%s\n"
                "It'll be used as the input for ROAST.", mri, out_path)
    return out_path, True


def convert_to_ras_point_cloud(mri, data):
    """Apply the same flips and permutation to a point cloud of voxel coordinates.

    Returns ``(data, perm_order)`` with 1-based coordinates and a 0-based
    permutation order.
    """
    volume = NiftiVolume.load(mri)
    _, perm_order, flip_tag = orientation_of(volume)
    dims = np.asarray(volume.header["dim"], dtype=float)

    data = np.array(data, dtype=float, copy=True)
    for j in range(3):
        if flip_tag[j] < 0:
            data[:, j] = dims[j + 1] - data[:, j] + 1        # note the +1
    return data[:, perm_order], perm_order
