"""Volumetric meshing of the segmented head.

* :mod:`~roast.mesh.iso2mesh` - the front-end to the CGAL mesher bundled with
  iso2mesh, plus the node reordering that follows it.
* :mod:`~roast.mesh.build` - mesh the tissues together with the electrodes and
  the gel, and write the mesh for the solver.
"""

from .iso2mesh import cgalv2m, sort_mesh, cgalmesh_path
from .build import mesh_by_iso2mesh, region_names

__all__ = ["cgalv2m", "sort_mesh", "cgalmesh_path", "mesh_by_iso2mesh", "region_names"]
