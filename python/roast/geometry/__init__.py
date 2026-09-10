"""Geometric primitives shared by the electrode code.

* :mod:`~roast.geometry.shapes` - sampling lines, cuboids and cylinders densely
  enough to fill the voxel grid once rounded (pads, discs and rings).
* :mod:`~roast.geometry.pointcloud` - the point-cloud operations on the head:
  mask edges, nearest-neighbour mapping and projection onto the scalp.
* :mod:`~roast.geometry.spline` - the natural cubic spline used to measure the
  nasion-to-inion arc along the scalp.
"""

from .shapes import draw_line, draw_cuboid, cylinder_between, draw_cylinder
from .pointcloud import (mask_to_edge_point_cloud, map_to_points,
                         project_to_closest_surface_points, get_data_around_target)
from .spline import ncs2dapprox, spline_curve

__all__ = [
    "draw_line", "draw_cuboid", "cylinder_between", "draw_cylinder",
    "mask_to_edge_point_cloud", "map_to_points",
    "project_to_closest_surface_points", "get_data_around_target",
    "ncs2dapprox", "spline_curve",
]
