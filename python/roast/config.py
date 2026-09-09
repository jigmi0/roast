"""Paths, constants and defaults shared across the ROAST pipeline.

The Python implementation lives in ``<repo>/python/roast`` and resolves the
bundled assets (``data/``, ``lib/``, ``example/``) relative to the repository
root.  Set the ``ROAST_ROOT`` environment variable to override the location,
e.g. when the package is installed outside of the repository.
"""

from __future__ import annotations

import os
import platform
from pathlib import Path

import numpy as np

__all__ = [
    "roast_root", "data_dir", "lib_dir", "example_dir",
    "cap_info_file", "elec_loc_file", "etpm_file",
    "NUM_OF_TISSUE", "TISSUE_NAMES", "TISSUE_INDEX",
    "LANDMARKS_IN_TPM", "LANDMARK_NAMES",
    "DEFAULT_CONDUCTIVITIES", "DEFAULT_MESH_OPTIONS", "DEFAULT_ELEC_SIZE",
    "CAP_TYPES", "ELEC_TYPES", "OPT_TYPES", "ORIENT_KEYWORDS",
    "arch", "exe_suffix",
]


def roast_root() -> Path:
    """Root directory of the ROAST distribution (holds data/, lib/, example/)."""
    env = os.environ.get("ROAST_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    # <root>/python/roast/config.py -> <root>
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    return roast_root() / "data"


def lib_dir() -> Path:
    return roast_root() / "lib"


def example_dir() -> Path:
    return roast_root() / "example"


def cap_info_file() -> Path:
    return data_dir() / "capInfo.xlsx"


def elec_loc_file() -> Path:
    return data_dir() / "elec72.loc"


def etpm_file() -> Path:
    return data_dir() / "eTPM.nii"


# --------------------------------------------------------------------------
# Tissue model.  Hard coded across ROAST: masks are labelled 1..6 in this order
# (note white before gray, which differs from the SPM output order).
# --------------------------------------------------------------------------
NUM_OF_TISSUE = 6
TISSUE_NAMES = ("WHITE", "GRAY", "CSF", "BONE", "SKIN", "AIR")
TISSUE_INDEX = {name.lower(): i + 1 for i, name in enumerate(TISSUE_NAMES)}

# Landmarks in eTPM.nii, in 1-based voxel coordinates of the template.
# Rows: nasion, inion, right, left, front neck, back neck, scalp centre, then
# the nine 10-10 electrodes on the central sagittal line.
LANDMARKS_IN_TPM = np.array([
    [61.0, 139.0, 98.0],       # nasion
    [61.0, 9.0, 100.0],        # inion
    [11.0, 62.0, 93.0],        # right
    [111.0, 63.0, 93.0],       # left (eTPM is in LAS orientation)
    [61.0, 113.0, 7.0],        # front neck
    [61.0, 7.0, 20.0],         # back neck
    [61.1698, 74.8445, 128.6539],   # scalp centre
    [61.1266, 6.7765, 128.3046],    # nine 10-10 electrodes on the
    [61.0821, 13.1721, 153.1825],   # central sagittal line
    [61.0449, 26.7791, 176.6905],
    [61.0294, 49.4432, 192.0928],
    [61.0444, 76.4435, 193.3487],
    [61.0751, 101.3033, 185.8634],
    [61.1154, 122.7582, 172.3329],
    [61.1658, 137.4929, 151.4222],
    [61.2171, 141.6240, 126.5092],
])

LANDMARK_NAMES = ("Nasion", "Inion", "Right Ear", "Left Ear",
                  "Front neck", "Back neck")

# Literature conductivity values in S/m.
DEFAULT_CONDUCTIVITIES = {
    "white": 0.126,
    "gray": 0.276,
    "csf": 1.65,
    "bone": 0.01,
    "skin": 0.465,
    "air": 2.5e-14,
    "gel": 0.3,
    "electrode": 5.9e7,
}

# iso2mesh / CGAL mesher defaults (higher resolution since ROAST v3).
DEFAULT_MESH_OPTIONS = {
    "radbound": 5,
    "angbound": 30,
    "distbound": 0.3,
    "reratio": 3,
    "maxvol": 10,
}

# Electrode sizes in mm.
DEFAULT_ELEC_SIZE = {
    "disc": (6.0, 2.0),          # [radius height]
    "pad": (50.0, 30.0, 3.0),    # [length width height]
    "ring": (4.0, 6.0, 2.0),     # [innerRadius outerRadius height]
}

CAP_TYPES = ("1020", "1010", "1005", "biosemi", "egi")
ELEC_TYPES = ("disc", "pad", "ring")
OPT_TYPES = ("unconstrained-wls", "wls-l1", "wls-l1per", "unconstrained-lcmv",
             "lcmv-l1", "lcmv-l1per", "max-l1", "max-l1per")
ORIENT_KEYWORDS = ("radial-in", "radial-out", "right", "left", "anterior",
                   "posterior", "right-anterior", "right-posterior",
                   "left-anterior", "left-posterior", "optimal")

NECK_ELECTRODES = ("nk1", "nk2", "nk3", "nk4")


def arch() -> str:
    """MATLAB-style architecture string, used to pick bundled binaries."""
    system = platform.system()
    if system == "Windows":
        return "win64"
    if system == "Linux":
        return "glnxa64"
    if system == "Darwin":
        return "maci64"
    raise RuntimeError(f"Unsupported operating system: {system}")


def exe_suffix() -> str:
    """Suffix used by the iso2mesh binaries shipped under ``lib/iso2mesh/bin``."""
    return {"win64": ".exe", "glnxa64": ".mexa64", "maci64": ".mexmaci64"}[arch()]
