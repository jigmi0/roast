"""Finite-element solution of the volume conductor model.

* :mod:`~roast.solver.prepare` - find the outer surface of every electrode and
  append it to the mesh as the region the current is injected through.
* :mod:`~roast.solver.getdp` - write the getDP problem definition and run the
  solver.
* :mod:`~roast.solver.post` - interpolate the node values back onto the voxel
  grid, or stack them into the lead field.
"""

from .prepare import prepare_for_getdp, free_boundary, load_elec_areas
from .getdp import solve_by_getdp, write_pro_file, getdp_path, getdp_command
from .post import post_getdp, interpolate_to_grid

__all__ = ["prepare_for_getdp", "free_boundary", "load_elec_areas",
           "solve_by_getdp", "write_pro_file", "getdp_path", "getdp_command",
           "post_getdp", "interpolate_to_grid"]
