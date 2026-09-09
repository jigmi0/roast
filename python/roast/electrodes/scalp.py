"""Morphological clean-up of the scalp mask before electrodes are placed."""

from __future__ import annotations

import numpy as np

from ..utils.logging import get_logger
from ..utils.matlab import find, imclose, imfill_holes, imopen, ind2sub

__all__ = ["clean_scalp"]

logger = get_logger()


def clean_scalp(scalp_in, scalp_surface):
    """Close, fill and open the scalp so it has a smooth outer surface.

    Returns ``(scalp_opened, scalp_filled)``: the opened mask is what the local
    surface normals are computed from, the filled one is what the electrodes
    are pushed against.
    """
    logger.info("cleaning the hair for gel injection...")
    scalp_in = np.asarray(scalp_in).astype(bool).copy()
    surface = np.asarray(scalp_surface, dtype=int)

    # Force the bottom-most slice closed: MRIs with a limited field of view are
    # often cut off in the middle of the face.
    r0, r1 = surface[:, 0].min(), surface[:, 0].max()
    a0, a1 = surface[:, 1].min(), surface[:, 1].max()
    s0 = surface[:, 2].min()
    scalp_in[r0 - 1:r1, a0 - 1:a1, s0 - 1] = True

    centroid = np.round(np.mean(ind2sub(scalp_in.shape, find(scalp_in)), axis=0)).astype(int)

    size_se = 0
    scalp_filled = None
    while True:
        size_se += 10
        scalp_filled = imfill_holes(imclose(scalp_in, np.ones((size_se,) * 3)))
        if scalp_filled[centroid[0] - 1, centroid[1] - 1, centroid[2] - 1]:
            break

    nx, ny, nz = scalp_in.shape
    size_se = 0
    while True:
        size_se += 10
        if size_se > 30:
            # Force the image boundaries "open" so the opening can terminate.
            scalp_filled[:, :, 0] = False
            scalp_filled[:, :, nz - 1] = False
            scalp_filled[:, 0, :] = False
            scalp_filled[:, ny - 1, :] = False
            scalp_filled[0, :, :] = False
            scalp_filled[nx - 1, :, :] = False
        scalp_out = imopen(scalp_filled, np.ones((size_se,) * 3))
        touching = np.concatenate([scalp_out[:, :, nz - 1].ravel(),
                                   scalp_out[:, 0, :].ravel(),
                                   scalp_out[0, :, :].ravel(),
                                   scalp_out[nx - 1, :, :].ravel()])
        if not touching.any():
            break
    return scalp_out, scalp_filled
