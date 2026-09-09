"""Automated clean-up of the SPM12 segmentation.

Port of ``segTouchup``, which merges the ``mysegment`` and ``autoPatching``
routines of ROAST 2.1.  See Huang et al. 2013 (DOI 10.1088/1741-2560/10/6/066004)
and Huang et al. 2017 (DOI 10.1101/217331) for the rationale behind each rule.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..io.matfile import load_mat
from ..io.nifti import NiftiVolume
from ..utils.logging import get_logger
from ..utils.matlab import (bwareaopen, filter_slices, fspecial_gaussian, imclose,
                            imdilate, imerode, imfill_holes, size_of_object)
from .masks import binary_mask_generate

__all__ = ["seg_touchup"]

logger = get_logger()

# Sigma of the 5x5 Gaussian applied to each tissue before binarising.
_SMOOTHING_SIGMA = {"gray": 0.2, "white": 0.1, "csf": 0.1,
                    "bone": 0.4, "skin": 1.0, "air": 1.0}


def _neighbour_offsets() -> np.ndarray:
    """The 18 face- and edge-neighbours of a voxel (corners and centre omitted)."""
    grid = np.stack(np.meshgrid(*[np.arange(-1, 2)] * 3, indexing="ij"), axis=-1)
    offsets = grid.reshape(-1, 3)
    keep = np.abs(offsets).sum(axis=1)
    return offsets[(keep == 1) | (keep == 2)]


def seg_touchup(spm_out, seg_out, is_smooth: bool = True, conn: int = 18) -> Path:
    """Clean up the six SPM tissue maps and write the combined mask volume.

    ``spm_out`` names the volume the ``c1``..``c6`` maps belong to and
    ``seg_out`` the volume the ``_masks.nii`` output is named after.
    """
    logger.info("loading data...")
    spm_out, seg_out = Path(spm_out), Path(seg_out)
    directory = spm_out.parent
    spm_name, seg_name = spm_out.stem, seg_out.stem

    order = ("gray", "white", "csf", "bone", "skin", "air")   # SPM's c1..c6 order
    volumes = {name: NiftiVolume.load(directory / f"c{i}{spm_name}.nii")
               for i, name in enumerate(order, start=1)}
    data = {name: volume.img for name, volume in volumes.items()}

    if is_smooth:
        logger.info("smoothing masks...")
        for name in order:
            kernel = fspecial_gaussian(5, _SMOOTHING_SIGMA[name])
            data[name] = filter_slices(data[name], kernel)

    logger.info("creating binary masks...")
    empty, gray, white, csf, bone, skin, air = binary_mask_generate(
        data["gray"], data["white"], data["csf"], data["bone"], data["skin"], data["air"])

    logger.info("fixing CSF continuity...")
    selem = np.ones((3, 3, 3))
    contin = (empty & imdilate(csf, selem)) | (imdilate(bone, selem) & gray)
    csf = csf | contin
    # No tissue is removed here, so no new empty voxels are created.
    _, csf, bone, gray = binary_mask_generate(csf, bone, gray)

    logger.info("removing disconnected voxels...")
    for name, mask, min_components in (("gray", gray, 2), ("white", white, 2),
                                       ("csf", csf, 4), ("bone", bone, 2),
                                       ("skin", skin, 2)):
        sizes, _ = size_of_object(mask, conn)
        if len(sizes) >= min_components:
            # Keep only the largest component(s); the threshold is one voxel above
            # the size of the first component that should be dropped.
            cleaned = bwareaopen(mask, int(sizes[min_components - 1]) + 1, conn)
            if name == "gray":
                gray = cleaned
            elif name == "white":
                white = cleaned
            elif name == "csf":
                csf = cleaned
            elif name == "bone":
                bone = cleaned
            else:
                skin = cleaned
    air = bwareaopen(air, 20)

    logger.info("generating and labeling empty voxels...")
    empty = binary_mask_generate(gray, white, csf, bone, skin, air)[0]

    # Relabel every unassigned voxel to its nearest tissue: the Gaussian blur acts
    # as a distance measure and the max operation picks the closest tissue.
    kernel = fspecial_gaussian(5, 1)
    while np.any(empty):
        filtered = {}
        for name, mask in (("air", air), ("gray", gray), ("white", white),
                           ("csf", csf), ("bone", bone), ("skin", skin)):
            filtered[name] = filter_slices((mask.astype(np.uint8) * 255), kernel)
        _, f_air, f_gray, f_white, f_csf, f_bone, f_skin = binary_mask_generate(
            filtered["air"], filtered["gray"], filtered["white"],
            filtered["csf"], filtered["bone"], filtered["skin"])

        gray = (empty & f_gray) | gray
        white = (empty & f_white) | white
        csf = (empty & f_csf) | csf
        bone = (empty & f_bone) | bone
        skin = (empty & f_skin) | skin
        air = (empty & f_air) | air
        assigned = ((empty & f_gray) | (empty & f_white) | (empty & f_csf)
                    | (empty & f_bone) | (empty & f_skin) | (empty & f_air))
        empty = empty ^ assigned

    logger.info("removing outside air...")
    outside = ~air
    outside = imerode(imfill_holes(imclose(outside, np.ones((10, 10, 10)))),
                      np.ones((12, 12, 12)))
    air = air & outside

    rmask_file = directory / (spm_name + "_rmask.mat")
    if not rmask_file.exists():
        raise FileNotFoundError(
            f"{rmask_file} not found; it is written by the SPM segmentation step and "
            "holds the exceptions (skull holes, eyeballs, white-matter exclusions) "
            "used by the automated touch-up.")
    rmask = load_mat(rmask_file)
    holes_vol = np.asarray(rmask["holes_vol"], dtype=float)
    eyes_vol = np.asarray(rmask["eyes_vol"], dtype=float)
    wm_exclude_vol = np.asarray(rmask["WMexclude_vol"], dtype=float)

    # Tissue labels: white, gray, csf, bone, skin, air (white before gray, which
    # differs from the SPM output order; changed after ROAST v2.1).
    all_mask = np.zeros(white.shape, dtype=np.uint8)
    all_mask[white] = 1
    all_mask[gray] = 2
    all_mask[csf] = 3
    all_mask[bone] = 4
    all_mask[skin] = 5
    all_mask[air] = 6

    offsets = _neighbour_offsets()
    shape = all_mask.shape

    def interior(coords):
        return np.all((coords > 0) & (coords < np.asarray(shape) - 1), axis=1)

    # -- patch gray matter -------------------------------------------------
    # White matter touching CSF/bone/skin/air becomes gray matter.  The brain
    # stem and the tissue around the ventricles have no gray matter, so those
    # voxels are excluded.
    logger.info("patching gray matter...")
    wm_all = all_mask == 1
    wm_keep = wm_all & imdilate(wm_exclude_vol > 1e-4, np.ones((10, 10, 10)))
    coords = np.argwhere(wm_all ^ wm_keep)
    coords = coords[interior(coords)]
    if coords.size:
        for offset in offsets:
            neighbours = coords + offset
            values = all_mask[tuple(neighbours.T)]
            wrong = np.isin(values, (3, 4, 5, 6))
            if np.any(wrong):
                target = coords[wrong]
                all_mask[tuple(target.T)] = 2          # the voxel itself, not the neighbour

    # -- patch CSF ---------------------------------------------------------
    # Gray/white matter touching bone/skin/air turns those neighbours into CSF.
    logger.info("patching CSF...")
    coords = np.argwhere((all_mask == 1) | (all_mask == 2))
    coords = coords[interior(coords)]
    if coords.size:
        for offset in offsets:
            neighbours = coords + offset
            values = all_mask[tuple(neighbours.T)]
            wrong = np.isin(values, (4, 5, 6))
            if np.any(wrong):
                target = neighbours[wrong]
                all_mask[tuple(target.T)] = 3

    # -- patch bone --------------------------------------------------------
    # CSF touching skin/air turns those neighbours into bone; the eyeballs are
    # excluded (1e-4 mirrors the "tiny" cut-off of spm_load_mask).
    logger.info("patching bone...")
    coords = np.argwhere(all_mask == 3)
    if coords.size:
        eyes = eyes_vol[tuple(coords.T)] > 1e-4
        coords = coords[~eyes]
        coords = coords[interior(coords)]
    if coords.size:
        for offset in offsets:
            neighbours = coords + offset
            values = all_mask[tuple(neighbours.T)]
            wrong = np.isin(values, (5, 6))
            if np.any(wrong):
                target = neighbours[wrong]
                all_mask[tuple(target.T)] = 4
    all_mask[(all_mask == 4) & (holes_vol > 1e-4)] = 5

    out = volumes["white"]
    out.img = all_mask
    out.set_scaling(1.0, 0.0)       # so that viewers do not rescale the labels
    out.header["descrip"] = b"tissue masks"
    out.header.set_data_dtype(np.uint8)
    out_path = directory / (seg_name + "_masks.nii")
    out.save(out_path)

    for tissue in range(1, 7):
        (directory / f"c{tissue}{spm_name}.nii").unlink(missing_ok=True)
    rmask_file.unlink(missing_ok=True)
    return out_path
