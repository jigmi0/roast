"""Header bookkeeping: MNI-aligned copies and renaming of the SPM outputs."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from ..io.nifti import NiftiVolume, spm_mat_to_srow

__all__ = ["align_header_to_mni", "rename_spm_res"]


def _update_affine(mri, voxel_to_world, out_name: str) -> Path:
    """Write a copy of ``mri`` whose header maps voxels straight into MNI space."""
    mri = Path(mri)
    volume = NiftiVolume.load(mri)
    volume.set_srow(spm_mat_to_srow(np.asarray(voxel_to_world, dtype=float)),
                    force_sform=True)
    return volume.save(mri.parent / (out_name + "_MNI" + mri.suffix))


def align_header_to_mni(t1, t2, seg_out, mri2mni) -> None:
    """Save ``_MNI`` copies of the T1, the T2 and the segmentation.

    ``mri2mni`` maps 1-based MRI voxel coordinates to MNI coordinates, so the
    voxel-to-world mapping of the copies lands directly in MNI space.
    """
    seg_out = Path(seg_out)
    directory = seg_out.parent
    seg_name = seg_out.stem

    if Path(t1).exists():
        _update_affine(t1, mri2mni, seg_name)

    if t2:
        t2 = Path(t2)
        if t2.exists():
            _update_affine(t2, mri2mni, t2.stem)

    masks = directory / (seg_name + "_masks.nii")
    if masks.exists():
        _update_affine(masks, mri2mni, seg_name + "_masks")


def rename_spm_res(src, tar) -> None:
    """Rename the SPM segmentation outputs from the input name to the model name."""
    src, tar = Path(src), Path(tar)
    directory = src.parent
    src_name, tar_name = src.stem, tar.stem

    for tissue in range(1, 7):
        shutil.move(str(directory / f"c{tissue}{src_name}.nii"),
                    str(directory / f"c{tissue}{tar_name}.nii"))
    for suffix in ("_rmask.mat", "_seg8.mat"):
        shutil.move(str(directory / (src_name + suffix)),
                    str(directory / (tar_name + suffix)))
