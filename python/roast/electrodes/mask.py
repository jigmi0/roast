"""Conversion of the electrode/gel point clouds into labelled 3-D masks."""

from __future__ import annotations

import numpy as np

from ..utils.logging import get_logger

__all__ = ["generate_elec_mask"]

logger = get_logger()

_OUT_OF_BOUNDS = ("Electrode {name} goes out of image boundary. ROAST cannot proceed "
                  "without a properly placed electrode. Please expand the input MRI by "
                  "specifying the 'zeroPadding' option.")


def generate_elec_mask(all_coords, shape, names, do_warn: bool) -> np.ndarray:
    """Rasterise every electrode point cloud into one labelled volume.

    Each electrode gets its own label (1..N) so that the mesher and the solver
    can address them individually.  Overlapping electrodes are rejected: they
    would make the boundary conditions ambiguous.
    """
    shape = tuple(int(s) for s in shape)
    volume = np.zeros(shape, dtype=np.int32)
    occupied = np.zeros(shape, dtype=bool)

    for i, coords in enumerate(all_coords):
        name = names[i]
        if coords is None or len(coords) == 0:
            raise ValueError(_OUT_OF_BOUNDS.format(name=name))
        coords = np.asarray(coords)
        inside = ((coords[:, 0] > 0) & (coords[:, 0] <= shape[0])
                  & (coords[:, 1] > 0) & (coords[:, 1] <= shape[1])
                  & (coords[:, 2] > 0) & (coords[:, 2] <= shape[2]))
        if not inside.any():
            raise ValueError(_OUT_OF_BOUNDS.format(name=name))
        if inside.sum() < len(coords) and do_warn:
            logger.warning("Part of the electrode %s goes out of image boundary. ROAST "
                           "can continue but results may not be accurate. It is "
                           "recommended that you expand the input MRI by specifying the "
                           "'zeroPadding' option.", name)
        coords = coords[inside].astype(np.int64) - 1        # to 0-based indices
        index = (coords[:, 0], coords[:, 1], coords[:, 2])
        if occupied[index].any():
            previous = names[i - 1] if i > 0 else "another electrode"
            raise ValueError(
                f"Electrode {name} overlaps with Electrode {previous}. ROAST cannot "
                "continue as overlapping electrodes will confuse ROAST when setting up "
                "the boundary conditions for the model. To avoid overlapping, please do "
                "not place two electrodes too close to each other, or reduce the size of "
                "any neighboring electrodes.")
        volume[index] = i + 1
        occupied[index] = True
    return volume
