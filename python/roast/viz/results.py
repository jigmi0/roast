"""Visualisation of the simulation and targeting results.

The 3-D renderings are done in world space so that left and right cannot be
confused; the slice views stay in voxel space.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from ..config import NUM_OF_TISSUE
from ..io.meshfile import read_pos
from ..segment.masks import brain_crop
from ..utils.logging import get_logger
from ..utils.matlab import interp1, prctile
from .slices import roast_colormap, sliceshow
from .surfaces import add_mesh_surface, _equalise

__all__ = ["render_field_on_mesh", "visualize_res"]

logger = get_logger()


def _to_world(node, geom):
    """Pseudo-world mesh coordinates back to voxel, then out to world space."""
    node = np.array(node, dtype=float, copy=True)
    for i in range(3):
        node[:, i] = node[:, i] / geom.mat[i, i]
    homogeneous = np.column_stack([node[:, :3], np.ones(len(node))])
    node[:, :3] = (geom.mat @ homogeneous.T).T[:, :3]
    return node


def render_field_on_mesh(node, elem, face, color, in_current, num_of_gel,
                         surface_index, elec_indices, fig_name, label,
                         upper_percentile=None):
    """Render one scalar field on a tissue surface, with the electrodes on top.

    The electrodes are painted with the value that corresponds to the current
    injected there, so a single colour bar tells both stories.
    """
    elem = np.asarray(elem, dtype=int)
    face = np.asarray(face, dtype=int)
    in_current = np.asarray(in_current, dtype=float).ravel()

    tissue_faces = face[face[:, 3] == surface_index, :3]
    tissue_elems = elem[elem[:, 4] == surface_index, :4]
    values = np.array(color, dtype=float, copy=True)

    used = np.unique(tissue_elems) - 1
    low = float(np.nanmin(values[used]))
    high = (float(np.nanmax(values[used])) if upper_percentile is None
            else float(prctile(values[used], upper_percentile)))
    data_range = (low, high)

    current_range = (float(in_current.min()), float(in_current.max()))
    for_elec = interp1(current_range, data_range, in_current[elec_indices])
    for i, index in enumerate(elec_indices):
        nodes = np.unique(elem[elem[:, 4] == NUM_OF_TISSUE + num_of_gel + index + 1, :4])
        values[nodes - 1] = for_elec[i]

    elec_faces = face[np.isin(face[:, 3],
                              NUM_OF_TISSUE + num_of_gel + np.asarray(elec_indices) + 1), :3]

    figure = plt.figure(fig_name + ". Move your mouse to rotate.", figsize=(9, 7),
                        facecolor="w")
    axis = figure.add_subplot(111, projection="3d")
    mesh = add_mesh_surface(axis, node, tissue_faces, values, clim=data_range)
    add_mesh_surface(axis, node, elec_faces, values, clim=data_range)
    _equalise(axis, np.asarray(node, dtype=float)[:, :3])
    axis.view_init(elev=20, azim=-70)
    axis.set_axis_off()

    mappable = plt.cm.ScalarMappable(cmap="jet",
                                     norm=plt.Normalize(*data_range))
    bar = figure.colorbar(mappable, ax=axis, fraction=0.03, pad=0.02)
    bar.set_label(label, fontsize=14)
    current_bar = figure.colorbar(
        plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(*current_range)),
        ax=axis, fraction=0.03, pad=0.10, location="left")
    current_bar.set_label("Injected current (mA)", fontsize=14)
    return figure, mesh


def visualize_res(subj, mask, mri2mni, node, elem, face, in_current, geom, uni_tag,
                  vol_all=None, ef_mag=None, ef_all=None, xopt=None,
                  target_coord=None, surface_index: int = 2):
    """Show the results of ``roast`` (voltage and field) or of ``roast_target``.

    Pass ``vol_all`` for a simulation; pass ``xopt`` (the node values of the
    optimised field) and ``target_coord`` for a targeting run.
    """
    subj = Path(subj)
    directory = subj.parent
    subj_name = subj.stem
    is_roast = vol_all is not None
    in_current = np.asarray(in_current, dtype=float).ravel()
    num_of_gel = len(in_current)

    logger.info("generating 3D renderings...")
    node = _to_world(node, geom)
    figures = []

    if is_roast:
        elec_indices = np.arange(num_of_gel)
        ids, values = read_pos(directory / f"{subj_name}_{uni_tag}_v.pos", 1)
        color = np.full(len(node), np.nan)
        color[ids - 1] = values[:, 0] - values[:, 0].min()
        figures.append(render_field_on_mesh(
            node, elem, face, color, in_current, num_of_gel, surface_index,
            elec_indices, f"Voltage in Simulation: {uni_tag}", "Voltage (mV)")[0])

        ids, values = read_pos(directory / f"{subj_name}_{uni_tag}_e.pos", 3)
        color = np.full(len(node), np.nan)
        color[ids - 1] = np.sqrt(np.sum(values ** 2, axis=1))
        figures.append(render_field_on_mesh(
            node, elem, face, color, in_current, num_of_gel, surface_index,
            elec_indices, f"Electric field in Simulation: {uni_tag}",
            "Electric field (V/m)", upper_percentile=95)[0])
    else:
        elec_indices = np.flatnonzero(np.abs(in_current) > 1e-3)
        xopt = np.asarray(xopt, dtype=float)
        color = np.full(len(node), np.nan)
        color[xopt[:, 0].astype(int) - 1] = np.sqrt(np.sum(xopt[:, 1:4] ** 2, axis=1))
        figures.append(render_field_on_mesh(
            node, elem, face, color, in_current, num_of_gel, surface_index,
            elec_indices, f"Electric field in Targeting: {uni_tag}",
            "Electric field (V/m)", upper_percentile=95)[0])

    logger.info("generating slice views...")
    mask_img = np.asarray(mask.img)
    brain = (mask_img == 1) | (mask_img == 2)
    brain_only = np.where(brain, 1.0, np.nan)
    cmap = roast_colormap()
    bbox = brain_crop(mask_img)

    if is_roast:
        sliceshow(vol_all * brain_only, np.round(bbox.mean(axis=0)).astype(int), cmap,
                  None, "Voltage (mV)",
                  f"Voltage in Simulation: {uni_tag}. Click anywhere to navigate.",
                  None, mri2mni, bbox)

    ef_all = np.asarray(ef_all, dtype=float) * brain_only[..., None]
    ef_mag = np.asarray(ef_mag, dtype=float) * brain_only
    finite = ef_mag[~np.isnan(ef_mag)]
    clim = (float(np.min(finite)), float(prctile(finite, 95)))

    if is_roast:
        sliceshow(ef_mag, np.round(bbox.mean(axis=0)).astype(int), cmap, clim,
                  "Electric field (V/m)",
                  f"Electric field in Simulation: {uni_tag}. Click anywhere to navigate.",
                  ef_all, mri2mni, bbox)
    else:
        for i, coord in enumerate(np.atleast_2d(target_coord)):
            sliceshow(ef_mag, coord.astype(int), cmap, clim, "Electric field (V/m)",
                      f"Electric field at Target {i + 1} in Targeting: {uni_tag}. "
                      "Click anywhere to navigate.", ef_all, mri2mni, bbox)
    return figures
