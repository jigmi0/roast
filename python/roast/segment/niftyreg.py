"""Affine registration of the individual MRI to MNI152 using NiftyReg.

Used together with the Multiaxial segmentation, which does not estimate a
mapping to the MNI space itself.  The result is stored in the same layout SPM12
uses for its ``_seg8.mat`` file, so the rest of the pipeline does not care which
segmentation backend produced it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from ..config import arch, etpm_file, example_dir, lib_dir
from ..io.matfile import save_spm_mapping
from ..io.nifti import geom_from_nifti
from ..utils.logging import get_logger

__all__ = ["reg_aladin_path", "run_nifty_reg"]

logger = get_logger()

_PLATFORM_DIR = {"win64": "win", "glnxa64": "linux", "maci64": "mac"}


def reg_aladin_path() -> Path:
    """Location of the bundled ``reg_aladin`` binary for this platform."""
    system = arch()
    binary = lib_dir() / "NiftyReg" / _PLATFORM_DIR[system] / "reg_aladin"
    if system == "win64":
        binary = binary.with_suffix(".exe")
    elif binary.exists():
        binary.chmod(0o755)
    return binary


def run_nifty_reg(input_mri) -> Path:
    """Register ``input_mri`` to the MNI152 head and save the mapping.

    Returns the path of the ``_niftyReg.mat`` file, which holds ``Affine``
    (MRI-to-MNI, inverted to match the SPM convention) together with the
    geometries of the MRI and of the tissue probability map.
    """
    input_mri = Path(input_mri)
    directory = input_mri.parent
    name = input_mri.stem

    output_mri = directory / (name + "_niftyReg.nii")
    reference = example_dir() / "MNI152_T1_1mm.nii"
    forward_file = directory / (name + "_tmp_forward.txt")

    command = [str(reg_aladin_path()), "-ref", str(reference), "-flo", str(input_mri),
               "-aff", str(forward_file), "-res", str(output_mri), "-voff"]
    logger.info("This may take a couple of minutes ...")
    result = subprocess.run(command)
    if result.returncode != 0:
        raise RuntimeError("niftyReg failed")

    affine = np.loadtxt(forward_file)
    forward_file.unlink(missing_ok=True)
    output_mri.unlink(missing_ok=True)
    affine = np.linalg.inv(affine)          # to be consistent with the SPM format

    out_path = directory / (name + "_niftyReg.mat")
    save_spm_mapping(out_path, geom_from_nifti(input_mri),
                     geom_from_nifti(etpm_file()), affine)
    return out_path
