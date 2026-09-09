"""Finite-element solution of the volume conductor model."""

from .prepare import prepare_for_getdp, free_boundary, load_elec_areas
from .getdp import solve_by_getdp, write_pro_file, getdp_path
from .post import post_getdp, interpolate_to_grid

__all__ = ["prepare_for_getdp", "free_boundary", "load_elec_areas",
           "solve_by_getdp", "write_pro_file", "getdp_path",
           "post_getdp", "interpolate_to_grid"]
