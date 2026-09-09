"""Placement of the four optional neck electrodes (nk1..nk4)."""

from __future__ import annotations

import numpy as np

from ..geometry.pointcloud import map_to_points, project_to_closest_surface_points
from ..utils.matlab import prctile

__all__ = ["place_neck_elec"]


def place_neck_elec(scalp, scalp_surface, landmarks, ind_need):
    """Project the requested neck electrodes onto the scalp surface.

    ``landmarks`` follow the order nasion, inion, right, left, front neck, back
    neck; ``ind_need`` holds 0-based indices into
    ``[front, back, left, right]``.  Returns ``(coords, center)``.
    """
    landmarks = np.asarray(landmarks, dtype=float)
    front_neck, back_neck = landmarks[4], landmarks[5]
    center = (front_neck + back_neck) / 2.0

    half_width = round(np.asarray(scalp).shape[0] / 2)
    candidates = np.array([
        front_neck,
        back_neck,
        [center[0] - half_width, center[1], center[2]],   # left neck
        [center[0] + half_width, center[1], center[2]],   # right neck
    ], dtype=float)
    candidates = candidates[np.asarray(ind_need, dtype=int)]

    cosine, order = project_to_closest_surface_points(candidates, scalp_surface, center)
    scalp_surface = np.asarray(scalp_surface, dtype=float)
    idx = np.zeros(len(candidates), dtype=int)
    for i in range(len(idx)):
        cut = prctile(cosine[:, i], 99.99)
        selected = order[cosine[:, i] > cut, i]
        test_points = scalp_surface[selected]
        _, farthest = map_to_points(center, test_points, "farthest")
        idx[i] = selected[int(farthest[0])]
        # The farthest of the well-aligned candidates is the point on the outer
        # surface of the scalp, i.e. the electrode location.
    return scalp_surface[idx], center
