"""Turning an MRI into six tissue masks, and locating it in MNI space.

* :mod:`~roast.segment.spm` - the default backend: drives the bundled SPM12
  through a MATLAB or Octave interpreter.
* :mod:`~roast.segment.touchup` - the automated clean-up of the SPM output
  (smoothing, CSF continuity, stray voxels, unassigned voxels, patching).
* :mod:`~roast.segment.multiaxial` - the alternative backend, a deep CNN that
  segments the whole head and needs no MATLAB.
* :mod:`~roast.segment.niftyreg` - affine registration to MNI152, used with the
  Multiaxial backend, which estimates no mapping of its own.
* :mod:`~roast.segment.masks` - mask arithmetic shared by the above.
"""

from .masks import binary_mask_generate, brain_crop, size_of_object
from .touchup import seg_touchup
from .spm import start_seg, SPMNotAvailable, matlab_command, spm_path
from .multiaxial import run_multiaxial, multiaxial_python
from .niftyreg import run_nifty_reg, reg_aladin_path

__all__ = [
    "binary_mask_generate", "brain_crop", "size_of_object", "seg_touchup",
    "start_seg", "SPMNotAvailable", "matlab_command", "spm_path",
    "run_multiaxial", "multiaxial_python", "run_nifty_reg", "reg_aladin_path",
]
