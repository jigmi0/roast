"""Modelling of the electrodes and of the conducting gel underneath them.

Every electrode is turned into two point clouds - one for the metal, one for
the gel - by sampling the corresponding solid (a cuboid for pads, a cylinder
for discs and rings) along the local surface normal of the scalp.
"""

from __future__ import annotations

import numpy as np

from ..config import NECK_ELECTRODES
from ..geometry.pointcloud import mask_to_edge_point_cloud
from ..geometry.shapes import draw_cuboid, draw_cylinder
from ..utils.logging import get_logger
from ..utils.matlab import intersect_rows, unique_rows

__all__ = ["place_and_model_electrodes", "expand_elec_para"]

logger = get_logger()

_ORIENTATION_KEYWORDS = {"lr": (1.0, 0.0, 0.0), "ap": (0.0, 1.0, 0.0),
                         "si": (0.0, 0.0, 1.0)}
_POINTS_PER_VOXEL = 2


def expand_elec_para(paras, num_of_elec: int):
    """Turn a single uniform parameter set into one entry per electrode."""
    if len(paras) != 1:
        return list(paras)
    return [paras[0].row(i) for i in range(num_of_elec)]


def _local_normal(local_surface, elec_loc, scalp_filled):
    """Outward surface normal at an electrode location.

    The normal is the least-variance direction of the neighbouring scalp points;
    its sign is fixed by walking along it until one end leaves the head.
    """
    covariance = np.cov(np.asarray(local_surface, dtype=float), rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    normal = eigenvectors[:, int(np.argmin(eigenvalues))]
    normal = normal / np.linalg.norm(normal)

    shape = np.asarray(scalp_filled.shape)
    length = 1
    while True:
        inside = np.round(elec_loc - length * normal).astype(int)
        outside = np.round(elec_loc + length * normal).astype(int)
        in_bounds = (np.all(np.minimum(inside, outside) > 0)
                     and np.all(np.maximum(inside, outside) <= shape))
        if not in_bounds:
            break
        a = bool(scalp_filled[inside[0] - 1, inside[1] - 1, inside[2] - 1])
        b = bool(scalp_filled[outside[0] - 1, outside[1] - 1, outside[2] - 1])
        if a != b:
            if not a:
                normal = -normal
            break
        length += 1
    return normal


def _describe(name, index, total, para):
    if "custom" in name.lower() or name.lower() in NECK_ELECTRODES:
        logger.info("placing electrode %s (%d out of %d)...", name, index + 1, total)
    else:
        logger.info("placing electrode %s (%d out of %d) in the %s layout...",
                    name, index + 1, total, str(para.cap_type).lower())


def place_and_model_electrodes(elec_loc, elec_range, scalp_clean_surf, scalp_filled,
                               elec_placing, elec_para, res, uni_tag=""):
    """Build the electrode and gel point clouds for every requested electrode.

    ``elec_loc`` holds the 1-based voxel coordinates on the scalp, ``elec_range``
    the 0-based indices of the surrounding surface points used to estimate the
    local normal, and ``res`` the MRI resolution in mm.
    """
    logger.info("placing electrodes...")
    elec_loc = np.atleast_2d(np.asarray(elec_loc, dtype=float))
    elec_range = np.atleast_2d(np.asarray(elec_range, dtype=int))
    scalp_clean_surf = np.asarray(scalp_clean_surf, dtype=float)
    scalp_filled = np.asarray(scalp_filled).astype(bool).copy()
    num_of_elec = elec_loc.shape[0]
    elec_para = expand_elec_para(elec_para, num_of_elec)

    # Pad electrodes are cut out of shells sitting on top of the scalp; build one
    # shell per distinct pad thickness.
    pad_height = np.zeros(len(elec_para))
    for i, para in enumerate(elec_para):
        if str(para.elec_type).lower() == "pad":
            pad_height[i] = para.elec_size[0, 2] / res
    gel_layer, elec_layer, layer_of = {}, {}, {}
    for height in np.unique(pad_height[pad_height > 0]):
        selem = np.ones((int(round(height)),) * 3)
        gel_layer[height], dilated = mask_to_edge_point_cloud(scalp_filled, "dilate", selem)
        elec_layer[height], _ = mask_to_edge_point_cloud(dilated, "dilate", selem)
    for i, height in enumerate(pad_height):
        if height > 0:
            layer_of[i] = height

    nx, ny, nz = scalp_filled.shape
    scalp_filled[:, :, 0] = False
    scalp_filled[:, :, nz - 1] = False
    scalp_filled[:, 0, :] = False
    scalp_filled[:, ny - 1, :] = False
    scalp_filled[0, :, :] = False
    scalp_filled[nx - 1, :, :] = False

    elec_all_coord = [None] * num_of_elec
    gel_all_coord = [None] * num_of_elec

    for i, para in enumerate(elec_para):
        local_surface = scalp_clean_surf[elec_range[i, :], :]
        normal = _local_normal(local_surface, elec_loc[i, :], scalp_filled)
        elec_type = str(para.elec_type).lower()
        _describe(elec_placing[i], i, num_of_elec, para)

        if elec_type == "pad":
            pad_length = para.elec_size[0, 0] / res
            pad_width = para.elec_size[0, 1] / res
            # Bigger electrodes need a thicker slab to intersect with.
            dim_try = np.mean([pad_length, pad_width])

            orientation = para.elec_ori
            if isinstance(orientation, str):
                orientation = np.array(_ORIENTATION_KEYWORDS[orientation.lower()])
            orientation = np.asarray(orientation, dtype=float).ravel()

            ori_short = np.cross(normal, orientation)
            ori_short = ori_short / np.linalg.norm(ori_short)
            ori_long = np.cross(normal, ori_short)
            ori_long = ori_long / np.linalg.norm(ori_long)

            slab = draw_cuboid(elec_loc[i, :], [pad_length, pad_width, dim_try],
                               ori_long, ori_short, normal, _POINTS_PER_VOXEL)
            slab = unique_rows(np.round(slab))
            height = layer_of[i]
            gel_coord = intersect_rows(slab, gel_layer[height].astype(slab.dtype))[0]
            elec_coord = intersect_rows(slab, elec_layer[height].astype(slab.dtype))[0]

        elif elec_type in ("disc", "ring"):
            if elec_type == "disc":
                radius_in, radius_out = 0.0, para.elec_size[0, 0] / res
                height = para.elec_size[0, 1] / res
                dim_try = radius_out
            else:
                radius_in = para.elec_size[0, 0] / res
                radius_out = para.elec_size[0, 1] / res
                height = para.elec_size[0, 2] / res
                dim_try = np.mean([radius_out, radius_in])

            gel_out = elec_loc[i, :] + 2 * height * normal
            electrode = gel_out + height * normal
            gel_in = gel_out - dim_try * normal
            gel_coord = np.round(draw_cylinder(radius_in, radius_out, gel_in, gel_out,
                                               _POINTS_PER_VOXEL))
            elec_coord = np.round(draw_cylinder(radius_in, radius_out, gel_out, electrode,
                                                _POINTS_PER_VOXEL))
            gel_coord = unique_rows(gel_coord)
            elec_coord = unique_rows(elec_coord)
        else:
            raise ValueError(f"Unsupported electrode type: {para.elec_type}")

        gel_all_coord[i] = gel_coord
        elec_all_coord[i] = elec_coord

    return elec_all_coord, gel_all_coord
