"""SPM12 segmentation.

SPM12 is a MATLAB toolbox, so the segmentation itself cannot be re-implemented
in Python; what the MATLAB ``start_seg`` did - assemble the batch and run it -
is what this module does, driving the copy of SPM12 bundled under ``lib/spm12``
through a MATLAB (or GNU Octave) interpreter.

Configuration, in order of precedence:

``ROAST_MATLAB_CMD``
    Command used to start the interpreter (default ``matlab``; ``octave`` also
    works with the bundled SPM12).
``ROAST_SPM_PATH``
    Location of SPM12 (default ``<repo>/lib/spm12``).

If no interpreter is available, use the ``multiaxial=True`` segmentation
backend instead, which is pure Python.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from ..config import etpm_file, lib_dir
from ..utils.logging import get_logger

__all__ = ["SPMNotAvailable", "matlab_command", "spm_path", "start_seg"]

logger = get_logger()

# Number of Gaussians per tissue, and the warping regularisation, straight from
# the MATLAB batch.
_NGAUS = (1, 1, 2, 3, 4, 2)
_WARP_REG = (0, 0.001, 0.5, 0.05, 0.2)


class SPMNotAvailable(RuntimeError):
    """Raised when no MATLAB/Octave interpreter can be found to run SPM12."""


def matlab_command() -> str:
    return os.environ.get("ROAST_MATLAB_CMD", "matlab")


def spm_path() -> Path:
    return Path(os.environ.get("ROAST_SPM_PATH", str(lib_dir() / "spm12")))


def _batch_script(t1: Path, t2: Path | None, template: Path, norm: bool) -> str:
    """Generate the MATLAB batch that ``start_seg`` used to build in memory."""
    native = "[1 1]" if norm else "[1 0]"
    warped = "[1 0]" if norm else "[0 0]"
    lines = [
        f"addpath('{spm_path()}');",
        "spm('defaults','fmri');",
        "spm_jobman('initcfg');",
        "clear matlabbatch;",
        f"matlabbatch{{1}}.spm.spatial.preproc.channel.vols = {{'{t1}'}};",
        "matlabbatch{1}.spm.spatial.preproc.channel.biasreg = 0.001;",
        "matlabbatch{1}.spm.spatial.preproc.channel.biasfwhm = 60;",
        "matlabbatch{1}.spm.spatial.preproc.channel.write = [0 0];",
    ]
    if t2 is not None:
        lines += [
            f"matlabbatch{{1}}.spm.spatial.preproc.channel(2).vols = {{'{t2}'}};",
            "matlabbatch{1}.spm.spatial.preproc.channel(2).biasreg = 0.001;",
            "matlabbatch{1}.spm.spatial.preproc.channel(2).biasfwhm = 60;",
            "matlabbatch{1}.spm.spatial.preproc.channel(2).write = [0 0];",
        ]
    for tissue, ngaus in enumerate(_NGAUS, start=1):
        lines += [
            f"matlabbatch{{1}}.spm.spatial.preproc.tissue({tissue}).tpm = "
            f"{{'{template},{tissue}'}};",
            f"matlabbatch{{1}}.spm.spatial.preproc.tissue({tissue}).ngaus = {ngaus};",
            f"matlabbatch{{1}}.spm.spatial.preproc.tissue({tissue}).native = {native};",
            f"matlabbatch{{1}}.spm.spatial.preproc.tissue({tissue}).warped = {warped};",
        ]
    reg = " ".join(str(v) for v in _WARP_REG)
    lines += [
        f"matlabbatch{{1}}.spm.spatial.preproc.warp.reg = [{reg}];",
        "matlabbatch{1}.spm.spatial.preproc.warp.affreg = 'mni';",
        "matlabbatch{1}.spm.spatial.preproc.warp.samp = 3;",
        "matlabbatch{1}.spm.spatial.preproc.warp.write = [0 0];",
        "matlabbatch{1}.spm.spatial.preproc.warp.mrf = 0;",
        "matlabbatch{1}.spm.spatial.preproc.warp.cleanup = 0;",
        "matlabbatch{1}.spm.spatial.preproc.warp.fwhm = 0;",
        "spm_jobman('run',matlabbatch);",
    ]
    return "\n".join(lines)


def start_seg(t1, t2=None, template=None, norm: bool = False) -> Path:
    """Segment ``t1`` (optionally aided by ``t2``) with SPM12.

    Writes ``c1``..``c6`` probability maps plus the ``_seg8.mat`` mapping next
    to the input image, and returns the path of the ``_seg8.mat`` file.
    """
    t1 = Path(t1).resolve()
    t2 = Path(t2).resolve() if t2 else None
    template = Path(template).resolve() if template else etpm_file().resolve()

    if t1.suffix == ".hdr":
        t1 = t1.with_suffix(".img")
    if t2 is not None and t2.suffix == ".hdr":
        t2 = t2.with_suffix(".img")

    executable = shutil.which(matlab_command())
    if executable is None:
        raise SPMNotAvailable(
            f"'{matlab_command()}' was not found on PATH, so SPM12 cannot be run. "
            "Set ROAST_MATLAB_CMD to your MATLAB (or Octave) executable, or run the "
            "pipeline with multiaxial=True to use the deep-learning segmentation, "
            "which needs no MATLAB.")

    script = _batch_script(t1, t2, template, norm)
    with tempfile.TemporaryDirectory() as workdir:
        script_file = Path(workdir) / "roast_segment.m"
        script_file.write_text(script)
        logger.info("Running SPM12 segmentation on %s (this takes a while)...", t1)
        if "octave" in Path(executable).name:
            command = [executable, "--no-gui", "--quiet", str(script_file)]
        else:
            command = [executable, "-batch", f"run('{script_file}')", "-nodisplay"]
        result = subprocess.run(command, cwd=str(t1.parent))
    if result.returncode != 0:
        raise RuntimeError("SPM12 segmentation failed; check the messages above.")

    seg8 = t1.parent / (t1.stem + "_seg8.mat")
    if not seg8.exists():
        raise RuntimeError(f"SPM12 finished but {seg8} was not produced.")
    return seg8
