"""Fitting a standard EEG cap (10-05, BioSemi or EGI) onto an individual head.

The template coordinates are transformed onto the subject's scalp and then, for
the non-EGI layouts, the position of the cap is optimised so that the
electrodes along the central sagittal line are evenly spaced - a tenth of the
nasion-to-inion arc apart, as the 10-10 system prescribes.  The EGI layout is
projected directly without that adjustment.
"""

from __future__ import annotations

import numpy as np

from ..geometry.pointcloud import map_to_points, project_to_closest_surface_points
from ..geometry.spline import ncs2dapprox, spline_curve
from ..utils.logging import get_logger
from ..utils.matlab import edge_sobel, find, imclose, imfill_holes, imopen, ind2sub

__all__ = ["fit_cap_to_individual", "CENTRAL_ELECTRODES", "CENTRAL_ELECTRODES_BIOSEMI"]

logger = get_logger()

# The nine 10-10 electrodes on the central sagittal line, and their BioSemi
# equivalents (the "Aid" names are the auxiliary positions of that layout).
CENTRAL_ELECTRODES = ("Oz", "POz", "Pz", "CPz", "Cz", "FCz", "Fz", "AFz", "Fpz")
CENTRAL_ELECTRODES_BIOSEMI = ("A19", "POzAid", "A6", "CPzAid", "A1", "FCzAid",
                              "E17", "AFzAid", "E12")

_THETA = 23.0          # angular spacing of the 10-10 system, in degrees
_ADJUST_FACTORS = np.arange(1.0, 0.499, -0.05)


def _central_sagittal_outline(scalp, landmarks, central_sag):
    """Trace the scalp outline on the central sagittal slice.

    Returns ``(yi, zi, arc_length)``: the sampled outline between inion and
    nasion and the length of the scalp arc between them.
    """
    nasion, inion = landmarks[0], landmarks[1]
    slice_img = np.asarray(scalp[int(central_sag) - 1, :, :]).astype(bool).copy()

    # Make sure the slice can be closed completely, so its edge is detected
    # correctly: fill the first column that contains any scalp.
    columns = np.flatnonzero(slice_img.sum(axis=0) > 0)
    if columns.size:
        col = columns[0]
        rows = np.flatnonzero(slice_img[:, col] > 0)
        slice_img[rows[0]:rows[-1] + 1, col] = True

    centroid = np.round(np.mean(ind2sub(slice_img.shape, find(slice_img)),
                                axis=0)).astype(int)
    size_se = 0
    while True:
        size_se += 8
        filled = imfill_holes(imclose(slice_img, np.ones((size_se, size_se))))
        if filled[centroid[0] - 1, centroid[1] - 1]:
            break

    outline = edge_sobel(imopen(filled, np.ones((3, 3))))
    subs = ind2sub(outline.shape, find(outline))       # column-major, like MATLAB
    r_c, c_c = subs[:, 0].astype(float), subs[:, 1].astype(float)

    ind_inion = np.flatnonzero(c_c == inion[2])
    ind_nasion = np.flatnonzero(c_c == nasion[2])
    if ind_inion.size == 0 or ind_nasion.size == 0:
        raise ValueError("Could not trace the scalp outline on the central sagittal "
                         "slice; check the landmarks and the segmentation.")
    top = int(np.argmax(c_c))

    # Walk the outline from the inion up over the vertex and back down to the
    # nasion, so the samples are ordered along the curve.
    right_up = np.flatnonzero((c_c >= c_c[ind_inion[0]]) & (r_c < r_c[top]))
    right_up = right_up[np.argsort(c_c[right_up], kind="stable")]
    right_down = np.flatnonzero((c_c >= c_c[ind_nasion[-1]]) & (r_c >= r_c[top]))
    right_down = right_down[np.argsort(-c_c[right_down], kind="stable")]
    index = np.concatenate([right_up, right_down])

    bx, by, breaks = ncs2dapprox(r_c[index], c_c[index])
    n_samples = int(breaks[-1])
    yi, zi = spline_curve(breaks, bx, by, np.linspace(1, n_samples, n_samples))
    arc_length = float(np.sum(np.sqrt(np.diff(yi) ** 2 + np.diff(zi) ** 2)))
    return yi, zi, arc_length


def fit_cap_to_individual(scalp, scalp_surface, landmarks, geom, cap_info,
                          ind_need, is_biosemi: bool = False, is_egi: bool = False):
    """Place the electrodes of a standard EEG system on the subject's scalp.

    ``ind_need`` holds the 0-based indices (into ``cap_info``) of the electrodes
    to place; pass an empty sequence to get only the central sagittal
    electrodes, which is what the manual-landmark workflow uses to re-estimate
    the registration.  Returns ``(coords, center)`` in 1-based voxel space.
    """
    landmarks = np.asarray(landmarks, dtype=float)
    nasion, inion, right, left = landmarks[0], landmarks[1], landmarks[2], landmarks[3]
    ind_need = np.asarray(ind_need, dtype=int).ravel()

    logger.info("measuring head size...")
    nasion_inion = float(np.linalg.norm(inion - nasion))
    line_center = (inion + nasion) / 2.0

    central_sag = None
    if not is_egi:
        central_sag = int(round(line_center[0]))
        yi, zi, arc_length = _central_sagittal_outline(scalp, landmarks, central_sag)

    logger.info("wearing the cap...")
    names = list(cap_info.names)
    if not is_egi:
        central = CENTRAL_ELECTRODES_BIOSEMI if is_biosemi else CENTRAL_ELECTRODES
        ind_central = np.array([names.index(name) for name in central], dtype=int)
    else:
        ind_central = np.zeros(0, dtype=int)

    ind_fit = np.concatenate([ind_central, ind_need]).astype(int)
    # Account for the MRI resolution, so non-1 mm and anisotropic MRIs work too.
    template = cap_info.coords[ind_fit] / geom.voxel_size[None, :]

    alpha = np.deg2rad((360.0 - 10.0 * _THETA) / 2.0)
    height = (nasion_inion / 2.0) / np.tan(alpha)
    # Where the centre of the electrode sphere sits; exact for 10-10 and
    # BioSemi, approximate for EGI.

    s = right - left
    s = s / np.linalg.norm(s)
    c = nasion - inion
    c = c / np.linalg.norm(c)
    a = np.cross(s, c)
    a = a / np.linalg.norm(a)

    factors = _ADJUST_FACTORS if not is_egi else np.array([1.0])
    if not is_egi:
        logger.info("adjust the cap for optimized position...this will take a while...")
    errors = np.zeros(len(factors))
    centers = np.zeros((len(factors), 3))
    coords = np.zeros((template.shape[0], 3, len(factors)))

    scalp_surface = np.asarray(scalp_surface, dtype=float)
    scale = round(max(np.asarray(scalp).shape) / 2)

    for n, factor in enumerate(factors):
        if not is_egi:
            logger.info("Iteration No. %d...", n + 1)
        center = line_center + height * factor * a
        centers[n] = center

        affine = np.eye(4)
        affine[:3, 0] = scale * s
        affine[:3, 1] = scale * c
        affine[:3, 2] = scale * a
        affine[:3, 3] = center
        adjusted = np.column_stack([template, np.ones(len(template))])
        adjusted[:, 2] = adjusted[:, 2] * factor      # squash the cap vertically
        transformed = (affine @ adjusted.T).T[:, :3]

        cosine, order = project_to_closest_surface_points(transformed, scalp_surface,
                                                          center)
        idx = np.zeros(len(transformed), dtype=int)
        for i in range(len(idx)):
            # Taking the maximum works better than a percentile for the
            # high-density layouts (10-05, BioSemi).
            selected = order[cosine[:, i] == cosine[0, i], i]
            test_points = scalp_surface[selected]
            _, farthest = map_to_points(center, test_points, "farthest")
            idx[i] = selected[int(farthest[0])]
        interp = scalp_surface[idx]
        coords[:, :, n] = interp

        if is_egi:
            continue

        # Score this cap position by how evenly the central electrodes are spaced.
        center_points = np.vstack([inion, interp[:len(ind_central)], nasion])
        center_fit = np.column_stack([np.full(len(yi), float(central_sag)), yi, zi])
        saved = 0
        distances = np.zeros(len(center_points) - 1)
        for ii in range(1, len(center_points)):
            delta = center_fit - center_points[ii][None, :]
            nearest = int(np.argmin(np.sqrt(np.sum(delta * delta, axis=1))))
            # Walking backwards leaves an empty span, and hence a zero distance,
            # exactly as the MATLAB indexing does.
            span = slice(saved, nearest + 1)
            distances[ii - 1] = np.sum(np.sqrt(np.diff(yi[span]) ** 2
                                               + np.diff(zi[span]) ** 2))
            saved = nearest
        errors[n] = np.sum(np.abs(distances - arc_length / 10.0))

    best = 0 if is_egi else int(np.argmin(errors))
    if ind_need.size:
        electrode_coord = coords[len(ind_central):, :, best]
    else:
        electrode_coord = coords[:len(ind_central), :, best]
    return electrode_coord, centers[best]
