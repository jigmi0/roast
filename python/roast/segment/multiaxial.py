"""Multiaxial segmentation, a deep CNN that segments the whole head.

See Birnbaum et al. 2025 (https://arxiv.org/abs/2501.18716).  The network and
its inference script ship under ``lib/multiaxial``; this module prepares the
conda environment on first use and then runs the script, mirroring
``runMultiaxial``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from ..config import arch, lib_dir, roast_root
from ..utils.logging import get_logger

__all__ = ["multiaxial_python", "run_multiaxial"]

logger = get_logger()

_ENV_DIR = {"win64": "multiaxialEnv", "glnxa64": "multiaxialEnvLinux",
            "maci64": "multiaxialEnvMac"}
_SETUP = {"win64": "setupWindows.bat", "glnxa64": "setupLinux.sh",
          "maci64": "setupMac.sh"}


def multiaxial_python(install: bool = True) -> Path:
    """Path of the Python interpreter of the Multiaxial environment.

    The environment is created by the bundled setup script the first time it is
    needed, exactly as the MATLAB version does.
    """
    system = arch()
    base = lib_dir() / "multiaxial"
    env = base / _ENV_DIR[system]

    if not env.is_dir():
        if not install:
            raise FileNotFoundError(f"Multiaxial environment {env} does not exist.")
        setup = base / _SETUP[system]
        logger.info("Setting up the Multiaxial environment (first run only)...")
        if system != "win64":
            setup.chmod(0o755)
        # The shell scripts build their paths from ``$(pwd)/lib/multiaxial``, so
        # they must be started from the repository root, as the MATLAB code does.
        result = subprocess.run([str(setup)], cwd=str(roast_root()))
        if result.returncode != 0:
            raise RuntimeError(f"Multiaxial setup script {setup} failed.")
    else:
        logger.info("Environment already exists. Skipping setup...")

    if system == "win64":
        # A duplicate of this DLL makes the runtime abort on Windows.
        duplicate = env / "Library" / "bin" / "libiomp5md.dll"
        if duplicate.exists():
            duplicate.unlink()
        return env / "python.exe"
    return env / "bin" / "python3"


def run_multiaxial(t1) -> None:
    """Run the Multiaxial segmentation on ``t1``.

    The script writes ``<name>_multiaxial_masks.nii`` next to the input image.
    """
    t1 = Path(t1)
    script = lib_dir() / "multiaxial" / "SEGMENT.py"
    interpreter = multiaxial_python()
    result = subprocess.run([str(interpreter), str(script), str(t1)])
    if result.returncode != 0:
        raise RuntimeError("Error running the Multiaxial segmentation script.")
    logger.info("Multiaxial segmentation finished successfully.")
