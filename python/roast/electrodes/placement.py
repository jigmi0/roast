"""Placement of all requested electrodes on the scalp surface.

This is the third step of the pipeline: it turns electrode *names* into
electrode and gel *masks* in the MRI voxel space.  Everything happens in RAS
orientation, which ROAST enforces up front since v3.0.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import NUM_OF_TISSUE
from ..geometry.pointcloud import (map_to_points, mask_to_edge_point_cloud,
                                   project_to_closest_surface_points)
from ..io.caps import read_cap_info, read_custom_locations
from ..io.nifti import NiftiVolume
from ..preprocess.orientation import convert_to_ras_point_cloud
from ..utils.logging import get_logger
from .cap import fit_cap_to_individual
from .mask import generate_elec_mask
from .model import place_and_model_electrodes
from .neck import place_neck_elec
from .scalp import clean_scalp

__all__ = ["electrode_placement"]

logger = get_logger()

_LOCAL_SURFACE_POINTS = 100


def _custom_electrode_coords(subj, options):
    """Read the user-defined locations and bring them into the model's voxel space."""
    coords = read_custom_locations(subj)[1]
    if options.is_non_ras:
        coords, perm = convert_to_ras_point_cloud(subj, coords)
    else:
        perm = np.array([0, 1, 2])
    if options.resamp:
        resolution = NiftiVolume.load(subj).resolution[perm]
        coords = coords * resolution[None, :]
    if options.zero_pad > 0:
        coords = coords + options.zero_pad
    return coords


def electrode_placement(subj, template, geom, landmarks, elec_needed, options, uni_tag):
    """Place the electrodes and write the electrode and gel masks.

    Returns the two masks as :class:`~roast.io.nifti.NiftiVolume` objects; both
    are labelled 1..N so that each electrode stays individually addressable.
    """
    subj = Path(subj)
    directory = subj.parent
    subj_name = subj.stem

    # Fill in the whole head first, to avoid complications with the scalp mask.
    scalp = np.asarray(template.img) > 0
    scalp_surface, _ = mask_to_edge_point_cloud(scalp, "erode", np.ones((3, 3, 3)))

    elec_para = options.elec_para
    ind_p, ind_n, ind_c = elec_para[0].ind_p, elec_para[0].ind_n, elec_para[0].ind_c
    landmarks = np.asarray(landmarks, dtype=float)

    # -- fit the cap on the individual head --------------------------------
    if ind_p.size:
        cap_type = str(elec_para[0].cap_type).lower()
        cap_info = read_cap_info(cap_type)
        coord_p, center_p = fit_cap_to_individual(
            scalp, scalp_surface, landmarks, geom, cap_info, ind_p,
            is_biosemi=(cap_type == "biosemi"), is_egi=(cap_type == "egi"))
    else:
        coord_p, center_p = np.zeros((0, 3)), None

    if ind_n.size:
        if np.any(landmarks[4:6, 2] <= 0):
            raise ValueError(
                "MRI does not cover the neck, so cannot place electrodes on the neck. "
                "Consider using 'zeroPadding' option to extend the input MRI and turn "
                "off Multiaxial option.")
        coord_n, center_n = place_neck_elec(scalp, scalp_surface, landmarks, ind_n)
    else:
        coord_n, center_n = np.zeros((0, 3)), None

    if ind_c.size:
        coords = _custom_electrode_coords(subj, options)[ind_c]
        _, nearest = map_to_points(coords, scalp_surface, "closest")
        coord_c = scalp_surface[nearest]
    else:
        coord_c = np.zeros((0, 3))

    # -- clean the head up so electrodes sit on a smooth surface -----------
    scalp_clean, scalp_filled = clean_scalp(scalp, scalp_surface)
    if (scalp_filled[:, :, [0, -1]].any() or scalp_filled[:, [0, -1], :].any()
            or scalp_filled[[0, -1], :, :].any()):
        logger.warning("Scalp touches image boundary. Electrodes may go out of image "
                       "boundary. ROAST can continue but results may not be accurate. "
                       "It is recommended that you expand the input MRI by specifying "
                       "the 'zeroPadding' option.")
    scalp_clean_surface, _ = mask_to_edge_point_cloud(scalp_clean, "erode",
                                                      np.ones((3, 3, 3)))

    # -- collect the neighbouring surface points of every electrode --------
    logger.info("calculating gel amount for each electrode...")
    ranges = []
    if ind_p.size:
        _, order = project_to_closest_surface_points(coord_p, scalp_clean_surface, center_p)
        ranges.append(order[:_LOCAL_SURFACE_POINTS, :].T)
    if ind_n.size:
        _, order = project_to_closest_surface_points(coord_n, scalp_clean_surface, center_n)
        ranges.append(order[:_LOCAL_SURFACE_POINTS, :].T)
    if ind_c.size:
        _, order = map_to_points(coord_c, scalp_clean_surface, "closer",
                                 _LOCAL_SURFACE_POINTS)
        ranges.append(order.T)

    electrode_coord = np.vstack([c for c in (coord_p, coord_n, coord_c) if len(c)])
    elec_range = np.vstack(ranges)

    # -- model the electrodes ----------------------------------------------
    # mean() handles anisotropic resolution; turning on 'resampling' is better.
    resolution = geom.mean_resolution
    elec_coords, gel_coords = place_and_model_electrodes(
        electrode_coord, elec_range, scalp_clean_surface, scalp_filled,
        elec_needed, elec_para, resolution, uni_tag)

    # -- rasterise and clean up --------------------------------------------
    logger.info("constructing electrode and gel volume to be exported...")
    volume_elec_labels = generate_elec_mask(elec_coords, scalp.shape, elec_needed, True)
    volume_gel_labels = generate_elec_mask(gel_coords, scalp.shape, elec_needed, False)

    logger.info("final clean-up...")
    volume_elec = volume_elec_labels > 0
    volume_gel = volume_gel_labels > 0
    volume_gel = volume_gel & ~(volume_gel & volume_elec)   # gel under the metal only
    for tissue in range(1, NUM_OF_TISSUE + 1):
        volume_gel = volume_gel & ~(volume_gel & (np.asarray(template.img) == tissue))

    logger.info("saving placed electrodes...")
    elec = template.copy()
    elec.img = (volume_elec_labels * volume_elec).astype(np.uint8)
    elec.header["descrip"] = b"electrode mask"
    elec.header.set_data_dtype(np.uint8)
    elec.save(directory / f"{subj_name}_{uni_tag}_mask_elec.nii")

    gel = template.copy()
    gel.img = (volume_gel_labels * volume_gel).astype(np.uint8)
    gel.header["descrip"] = b"gel mask"
    gel.header.set_data_dtype(np.uint8)
    gel.save(directory / f"{subj_name}_{uni_tag}_mask_gel.nii")

    return elec, gel
