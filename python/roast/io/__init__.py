"""File input/output for ROAST.

* :mod:`~roast.io.nifti` - reading and writing NIfTI volumes with the voxel data
  and the header left untouched, plus the SPM ``mat``/``dim`` geometry.
* :mod:`~roast.io.matfile` - the ``.mat`` files exchanged with SPM (the
  ``_seg8.mat`` mapping) and the result files ROAST writes.
* :mod:`~roast.io.meshfile` - the mesh and solver formats: INRImage in, Medit
  out, Gmsh for the solver, and the getDP node tables that come back.
* :mod:`~roast.io.caps` - the electrode layout files (``capInfo.xlsx``,
  ``elec72.loc`` and the per-subject custom locations).
"""

from .nifti import (NiftiVolume, load_untouch, ImageGeom, geom_from_nifti,
                    srow_to_spm_mat, spm_mat_to_srow)
from .matfile import load_mat, save_mat, load_spm_mapping, save_spm_mapping
from .meshfile import save_inr, read_medit, save_msh, read_pos
from .caps import (CapInfo, read_cap_info, read_elec_loc, read_custom_locations,
                   custom_locations_path, sheet_for_cap_type)

__all__ = [
    "NiftiVolume", "load_untouch", "ImageGeom", "geom_from_nifti",
    "srow_to_spm_mat", "spm_mat_to_srow",
    "load_mat", "save_mat", "load_spm_mapping", "save_spm_mapping",
    "save_inr", "read_medit", "save_msh", "read_pos",
    "CapInfo", "read_cap_info", "read_elec_loc", "read_custom_locations",
    "custom_locations_path", "sheet_for_cap_type",
]
