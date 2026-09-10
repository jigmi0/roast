"""Getting the MRI into the shape the rest of the pipeline expects.

* :mod:`~roast.preprocess.orientation` - detect the orientation of a volume and
  re-orient it (or a point cloud) into RAS.
* :mod:`~roast.preprocess.resample` - resample to 1 mm isotropic, and the
  bounding-box/reslice helpers that also serve the T2 alignment.
* :mod:`~roast.preprocess.padding` - add empty slices so electrodes near the
  edge of the field of view still fit.
* :mod:`~roast.preprocess.realign` - rigid-body coregistration of the T2 to the
  T1, maximising normalised mutual information.
* :mod:`~roast.preprocess.header` - the ``_MNI`` copies of the images and the
  renaming of the SPM outputs.
"""

from .orientation import convert_to_ras, convert_to_ras_point_cloud, orientation_of
from .resample import resamp_to_one_mm, reslice_to_grid, bounding_box
from .padding import zero_padding
from .realign import realign_t2, coregister, rigid_matrix
from .header import align_header_to_mni, rename_spm_res

__all__ = [
    "convert_to_ras", "convert_to_ras_point_cloud", "orientation_of",
    "resamp_to_one_mm", "reslice_to_grid", "bounding_box",
    "zero_padding", "realign_t2", "coregister", "rigid_matrix",
    "align_header_to_mni", "rename_spm_res",
]
