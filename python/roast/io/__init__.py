"""File input/output for ROAST (NIfTI, MATLAB, mesh and layout files)."""

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
