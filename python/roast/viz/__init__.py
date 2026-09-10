"""Figures and interactive views.

* :mod:`~roast.viz.slices` - the clickable three-plane slice viewer, with voxel
  and MNI coordinates and an optional vector-field overlay.
* :mod:`~roast.viz.surfaces` - 3-D renderings of the MRI, the segmentation and
  the placed electrodes.
* :mod:`~roast.viz.results` - the field rendered on a tissue surface, and the
  slice views of the results.
* :mod:`~roast.viz.topoplot` - the scalp topography of a montage.
* :mod:`~roast.viz.landmarks` - the GUIs for inspecting and hand-picking the
  anatomical landmarks.
"""

from .slices import SliceViewer, sliceshow, roast_colormap
from .surfaces import (view_mri, view_seg, view_electrodes, add_isosurface,
                       add_mesh_surface, SEGMENTATION_COLORS)
from .topoplot import topoplot, read_loc_positions
from .results import visualize_res, render_field_on_mesh
from .landmarks import get_landmarks_manual, check_landmarks, reorder_landmarks

__all__ = [
    "SliceViewer", "sliceshow", "roast_colormap",
    "view_mri", "view_seg", "view_electrodes", "add_isosurface", "add_mesh_surface",
    "SEGMENTATION_COLORS", "topoplot", "read_loc_positions",
    "visualize_res", "render_field_on_mesh",
    "get_landmarks_manual", "check_landmarks", "reorder_landmarks",
]
