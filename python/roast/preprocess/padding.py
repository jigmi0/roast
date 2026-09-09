"""Zero-padding of the MRI, so that electrodes near the edge of the field of
view still fit inside the volume."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..io.nifti import NiftiVolume
from ..utils.logging import get_logger

__all__ = ["zero_padding"]

logger = get_logger()


def zero_padding(mri, pad_num: int):
    """Add ``pad_num`` empty slices on all six sides of the volume.

    The header origin is shifted accordingly so that the world coordinates of
    the anatomy stay put.  Returns the path of the padded volume.
    """
    mri = Path(mri)
    if "_padded" in mri.name:
        logger.warning("%s has already been zero-padded. Nothing will happen here. If you "
                       "meant to add empty slices on the MRI, please provide MRI name "
                       "without the _padded suffix.", mri)
        return mri

    if mri.parent.name == "example" and mri.name.startswith("nyhead_"):
        out_path = mri.parent / f"nyhead_padded{pad_num}{mri.stem[6:]}{mri.suffix}"
    else:
        out_path = mri.parent / f"{mri.stem}_padded{pad_num}{mri.suffix}"

    if out_path.exists():
        logger.warning("%s has already been zero-padded by %d empty slices in the six "
                       "directions and saved as %s. ROAST will use that file as the input.",
                       mri, pad_num, out_path)
        return out_path

    logger.info("Padding %s by %d empty slices in all the six directions...", mri, pad_num)
    volume = NiftiVolume.load(mri)
    new_shape = tuple(np.asarray(volume.img.shape[:3]) + 2 * pad_num)
    img = np.zeros(new_shape, dtype=volume.img.dtype)
    img[pad_num:-pad_num, pad_num:-pad_num, pad_num:-pad_num] = volume.img
    volume.img = img
    dims = np.asarray(volume.header["dim"]).copy()
    dims[1:4] = new_shape
    volume.header["dim"] = dims

    srow = volume.srow()
    origin = np.linalg.inv(srow) @ np.array([0.0, 0.0, 0.0, 1.0])
    origin = origin[:3] + pad_num
    srow[:3, 3] = -srow[:3, :3] @ origin
    volume.set_srow(srow)
    volume.save(out_path)

    logger.info("%s has been zero-padded, and is saved as:\n%s\n"
                "It'll be used as the input for ROAST.", mri, out_path)
    return out_path
