"""Scalp topography of an electrode montage.

A stand-in for EEGLAB's ``topoplot``: the electrode positions are read from the
polar coordinates in a ``.loc`` file, the values are interpolated inside the
head circle and drawn together with a head, nose and ear outline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from ..config import elec_loc_file

__all__ = ["read_loc_positions", "topoplot"]


def read_loc_positions(path=None):
    """Read ``(names, x, y)`` from an EEGLAB ``.loc`` file.

    The file stores polar coordinates; as in EEGLAB the polar angle is measured
    from the nose, which points up in the plot.
    """
    path = Path(path) if path is not None else elec_loc_file()
    names, theta, radius = [], [], []
    with open(path, "r") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) < 4:
                continue
            theta.append(float(fields[1]))
            radius.append(float(fields[2]))
            names.append(fields[3].replace(".", ""))
    theta = np.deg2rad(np.asarray(theta))
    radius = np.asarray(radius)
    # EEGLAB's convention: x points to the nose, y to the left ear.
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return names, x, y


def topoplot(values, loc_file=None, axis=None, cmap="jet", clim=None,
             grid_scale: int = 200, plot_radius: float = 0.9, label: str = ""):
    """Draw a scalp map of ``values``, one per electrode in the ``.loc`` file."""
    names, x, y = read_loc_positions(loc_file)
    values = np.asarray(values, dtype=float).ravel()
    if len(values) != len(names):
        raise ValueError(f"expected {len(names)} values, got {len(values)}")

    if axis is None:
        axis = plt.gca()
    grid = np.linspace(-plot_radius, plot_radius, grid_scale)
    grid_x, grid_y = np.meshgrid(grid, grid)
    points = np.column_stack([y, x])
    interpolated = griddata(points, values, (grid_x, grid_y), method="cubic")
    # Fill the rim outside the electrode hull, as topoplot does, then mask the
    # area outside the head.
    outside_hull = np.isnan(interpolated)
    if outside_hull.any():
        interpolated[outside_hull] = griddata(points, values, (grid_x, grid_y),
                                              method="nearest")[outside_hull]
    interpolated[np.hypot(grid_x, grid_y) > plot_radius] = np.nan

    if clim is None:
        clim = (float(np.min(values)), float(np.max(values)))
    image = axis.pcolormesh(grid_x, grid_y, interpolated, cmap=cmap,
                            vmin=clim[0], vmax=clim[1], shading="auto")

    angle = np.linspace(0, 2 * np.pi, 200)
    axis.plot(plot_radius * np.cos(angle), plot_radius * np.sin(angle), "k",
              linewidth=2)
    nose = plot_radius * np.array([[-0.08, 0.99], [0.0, 1.12], [0.08, 0.99]])
    axis.plot(nose[:, 0], nose[:, 1], "k", linewidth=2)
    for side in (-1, 1):
        ear_angle = np.linspace(-np.pi / 2, np.pi / 2, 40)
        axis.plot(side * (plot_radius + 0.09 * plot_radius * np.cos(ear_angle)),
                  0.18 * plot_radius * np.sin(ear_angle), "k", linewidth=2)

    axis.plot(y, x, "k.", markersize=3)
    axis.set_aspect("equal")
    axis.set_axis_off()
    if label:
        axis.set_title(label)
    return image
