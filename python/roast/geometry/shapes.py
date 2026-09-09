"""Point-cloud generators for the electrode and gel geometry.

Electrodes are modelled by sampling their surface densely enough that rounding
the samples to the voxel grid fills the corresponding mask without holes
(``density`` is expressed in points per voxel).
"""

from __future__ import annotations

import numpy as np

from ..utils.matlab import unique_rows

__all__ = ["draw_line", "draw_cuboid", "cylinder_between", "draw_cylinder"]


def draw_line(p0, direction, length, density, dtype=np.float32) -> np.ndarray:
    """Sample a straight line starting at ``p0`` along ``direction``."""
    p0 = np.asarray(p0, dtype=float).reshape(3)
    direction = np.asarray(direction, dtype=float).reshape(3)
    samples = int(round(length * density))
    steps = np.arange(samples + 1, dtype=float)[:, None]
    coord = (p0[None, :] + steps * direction[None, :] / density).astype(dtype)
    return unique_rows(coord)


def draw_cuboid(center, dim, long_axis, short_axis, normal_axis, density,
                dtype=np.float32) -> np.ndarray:
    """Sample the volume of a cuboid, used to model pad electrodes.

    ``dim`` is ``[length width thickness]`` along ``long_axis``, ``short_axis``
    and ``normal_axis`` respectively.
    """
    center = np.asarray(center, dtype=float).reshape(3)
    dim = np.asarray(dim, dtype=float).reshape(3)
    long_axis = np.asarray(long_axis, dtype=float).reshape(3)
    short_axis = np.asarray(short_axis, dtype=float).reshape(3)
    normal_axis = np.asarray(normal_axis, dtype=float).reshape(3)

    corner = (center - long_axis * dim[0] / 2 - short_axis * dim[1] / 2
              - normal_axis * dim[2] / 2)
    n_short = int(round(dim[1] * density + 1))
    n_normal = int(round(dim[2] * density + 1))

    pieces = []
    for s in range(n_short):
        for n in range(n_normal):
            start = corner + short_axis * s / density + normal_axis * n / density
            pieces.append(draw_line(start, long_axis, dim[0], density, dtype))
    return np.vstack(pieces) if pieces else np.zeros((0, 3), dtype=dtype)


def cylinder_between(radii, n_theta: int, p1, p2, rng=None):
    """Facet coordinates of an ``n_theta``-sided cylinder running from ``p1`` to
    ``p2`` with the per-segment radii in ``radii``.

    Port of ``cylinder2P`` (Luigi Barone / Per Sundqvist).  The reference frame
    around the cylinder axis is arbitrary; a seeded generator is used so that
    repeated runs produce identical point clouds.
    """
    radii = np.atleast_1d(np.asarray(radii, dtype=float))
    p1 = np.asarray(p1, dtype=float).reshape(3)
    p2 = np.asarray(p2, dtype=float).reshape(3)
    if radii.size == 1:
        radii = np.array([radii[0], radii[0]])
    m = radii.size
    n_theta = max(int(n_theta), 2)
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta)

    axis = p2 - p1
    axis = axis / np.sqrt(axis @ axis)                # unit vector along the axis
    rng = np.random.default_rng(0) if rng is None else rng
    helper = rng.random(3)
    x2 = axis - helper / (helper @ axis)              # orthogonal to the axis
    x2 = x2 / np.sqrt(x2 @ x2)
    x3 = np.cross(axis, x2)
    x3 = x3 / np.sqrt(x3 @ x3)

    t = np.linspace(0.0, 1.0, m)[:, None]
    base = p1[None, :] + t * (p2 - p1)[None, :]                # (m, 3)
    ring = (np.cos(theta)[None, :, None] * x2[None, None, :]
            + np.sin(theta)[None, :, None] * x3[None, None, :])  # (1, n, 3)
    points = base[:, None, :] + radii[:, None, None] * ring       # (m, n, 3)
    return points[..., 0], points[..., 1], points[..., 2]


def draw_cylinder(inner_radius, outer_radius, top, bottom, density,
                  dtype=np.float32) -> np.ndarray:
    """Sample a (possibly hollow) cylinder: disc electrodes use
    ``inner_radius=0``, ring electrodes a positive inner radius."""
    top = np.asarray(top, dtype=float).reshape(3)
    bottom = np.asarray(bottom, dtype=float).reshape(3)
    height = float(np.linalg.norm(top - bottom))
    step = 1.0 / density
    radii = np.arange(inner_radius + step, outer_radius + step / 2, step)

    pieces = []
    for radius in radii:
        n_segments = max(int(round(height * density)), 1)
        n_theta = max(int(round(2 * np.pi * radius * density)), 2)
        x, y, z = cylinder_between(np.full(n_segments, radius), n_theta, top, bottom)
        pieces.append(np.column_stack([x.ravel(order="F"), y.ravel(order="F"),
                                       z.ravel(order="F")]).astype(dtype))
    return np.vstack(pieces) if pieces else np.zeros((0, 3), dtype=dtype)
