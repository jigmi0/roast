"""Tissue mask helpers shared by the segmentation steps."""

from __future__ import annotations

import numpy as np

from ..utils.matlab import size_of_object

__all__ = ["binary_mask_generate", "brain_crop", "size_of_object"]


def binary_mask_generate(*tissues):
    """Turn tissue probability maps into disjoint binary masks by a max operation.

    Accepts one to six probability volumes and returns
    ``(empty, mask1, ..., maskN)``.  ``empty`` marks the voxels that belong to
    no tissue at all, i.e. where every probability is zero.  The same routine
    is used to re-derive the masks after one of them has been edited.
    """
    tissues = [t for t in tissues if t is not None]
    if not tissues:
        raise ValueError("at least one tissue map is required")

    shape = np.asarray(tissues[0]).shape
    stack = np.empty((len(tissues) + 1,) + shape, dtype=float)
    stack[0] = 0.0                                    # the "no tissue" channel
    for i, tissue in enumerate(tissues, start=1):
        stack[i] = np.asarray(tissue, dtype=float)

    winner = np.argmax(stack, axis=0)
    masks = [winner == i for i in range(len(tissues) + 1)]
    return tuple(masks)


def brain_crop(mask) -> np.ndarray:
    """Bounding box of the brain, derived from the white matter mask.

    Returns a ``2x3`` array of 1-based voxel coordinates: the first row holds
    the lower bounds and the second the upper bounds, with the columns ordered
    R/L, A/P and S/I.
    """
    white = np.asarray(mask) == 1
    thresh = (0, 0, 100)
    bbox = np.zeros((2, 3), dtype=int)
    profiles = (white.sum(axis=(1, 2)), white.sum(axis=(0, 2)), white.sum(axis=(0, 1)))
    for axis, (profile, limit) in enumerate(zip(profiles, thresh)):
        indices = np.flatnonzero(profile > limit) + 1     # 1-based
        if indices.size == 0:
            bbox[:, axis] = (1, white.shape[axis])
        else:
            bbox[0, axis] = indices.min()
            bbox[1, axis] = indices.max()
    return bbox
