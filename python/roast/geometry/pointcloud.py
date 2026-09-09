"""Point-cloud operations on the segmented head.

All *coordinates* are 1-based voxel coordinates (the MATLAB convention that the
SPM affines assume); all *indices* returned by these helpers are 0-based, so
they can be used directly to index NumPy arrays.
"""

from __future__ import annotations

import numpy as np

from ..utils.matlab import find, imdilate, imerode, ind2sub

__all__ = ["mask_to_edge_point_cloud", "map_to_points",
           "project_to_closest_surface_points", "get_data_around_target"]


def mask_to_edge_point_cloud(mask, method: str = "erode", selem=None):
    """Detect the edge of a 3-D mask and return it as a point cloud.

    ``method='erode'`` gives the outermost layer of the mask itself,
    ``'dilate'`` the first layer just outside it.  Returns
    ``(edge_points, processed_mask)`` with 1-based coordinates.
    """
    if selem is None:
        selem = np.ones((3, 3, 3))
    mask = np.asarray(mask).astype(bool)
    if method == "erode":
        processed = imerode(mask, selem)
        edge = mask & ~processed
    elif method == "dilate":
        processed = imdilate(mask, selem)
        edge = processed & ~mask
    else:
        raise ValueError("Please enter either erode or dilate for the Method.")
    return ind2sub(edge.shape, find(edge)).astype(float), processed


def map_to_points(input_points, goal_points, criterion: str = "closest",
                  num_of_pts: int | None = None):
    """Map one point cloud onto another using the Euclidean distance.

    Returns ``(distances, indices)``; ``indices`` are 0-based positions in
    ``goal_points``.  For the ``closest``/``farthest`` criteria both have one
    entry per input point, for ``closer``/``farther`` they are
    ``(num_of_pts, n_input)`` arrays.
    """
    input_points = np.atleast_2d(np.asarray(input_points, dtype=np.float32))
    goal_points = np.atleast_2d(np.asarray(goal_points, dtype=np.float32))
    if input_points.shape[1] != goal_points.shape[1]:
        raise ValueError("You cannot map points that are in spaces of different dimensions!")

    # (n_goal, n_input) distance matrix, as in the MATLAB implementation.
    diff = goal_points[:, None, :] - input_points[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    order = np.argsort(dist, axis=0, kind="stable")
    dist_sorted = np.take_along_axis(dist, order, axis=0)

    n_goal = goal_points.shape[0]
    if criterion == "closest":
        return dist_sorted[0, :], order[0, :]
    if criterion == "farthest":
        return dist_sorted[-1, :], order[-1, :]
    if criterion in ("closer", "farther"):
        if num_of_pts is None:
            raise ValueError("Please specify how many points to map to in num_of_pts.")
        if num_of_pts > n_goal:
            raise ValueError("Number of points exceed size of goal point cloud.")
        if criterion == "closer":
            return dist_sorted[:num_of_pts, :], order[:num_of_pts, :]
        return dist_sorted[-num_of_pts:, :], order[-num_of_pts:, :]
    raise ValueError("Please specify either closest, farthest, closer or farther "
                     "as the mapping criterion.")


def project_to_closest_surface_points(points, surface, surface_center):
    """Project points onto a surface along the rays leaving ``surface_center``.

    Returns ``(cosine_sorted, index_on_surface)``, both ``(n_surface, n_points)``
    and sorted so that the best aligned surface point comes first.  Indices are
    0-based positions in ``surface``.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    surface = np.atleast_2d(np.asarray(surface, dtype=float))
    center = np.asarray(surface_center, dtype=float).reshape(1, -1)

    vec_p = points - center
    vec_p = (vec_p / np.linalg.norm(vec_p, axis=1, keepdims=True)).astype(np.float32)
    vec_s = surface - center
    vec_s = (vec_s / np.linalg.norm(vec_s, axis=1, keepdims=True)).astype(np.float32)

    # cosine of the angle between every surface point and every query point
    cosine = vec_s @ vec_p.T
    order = np.argsort(-cosine, axis=0, kind="stable")
    return np.take_along_axis(cosine, order, axis=0), order


def get_data_around_target(data, target, radius, grid=None):
    """Mean of ``data`` inside a sphere of ``radius`` voxels around ``target``.

    Used when the target voxel itself falls outside the brain mask.  ``target``
    is a 1-based voxel coordinate; NaNs are ignored.
    """
    data = np.asarray(data, dtype=float)
    target = np.asarray(target, dtype=float).reshape(3)
    if grid is None:
        grid = np.ogrid[1:data.shape[0] + 1, 1:data.shape[1] + 1, 1:data.shape[2] + 1]
    x, y, z = grid
    dist2 = ((x - target[0]) ** 2 + (y - target[1]) ** 2 + (z - target[2]) ** 2)
    inside = dist2 < radius ** 2
    values = data[inside]
    if values.size == 0 or np.all(np.isnan(values)):
        return np.nan
    return float(np.nanmean(values))
