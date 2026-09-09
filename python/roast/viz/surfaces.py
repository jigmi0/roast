"""3-D renderings of the head, the segmentation and the placed electrodes."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import ndimage

from ..io.nifti import NiftiVolume
from .slices import sliceshow

__all__ = ["SEGMENTATION_COLORS", "view_mri", "view_seg", "view_electrodes",
           "add_isosurface", "add_mesh_surface"]

#: Anatomically inspired colours for the six tissue labels (plus background).
SEGMENTATION_COLORS = np.array([
    [0.0, 0.0, 0.0],                        # background
    [1.0, 1.0, 1.0],                        # white matter
    [0.7, 0.7, 0.7],                        # gray matter
    [105 / 255, 175 / 255, 255 / 255],      # CSF
    [241 / 255, 214 / 255, 145 / 255],      # bone
    [177 / 255, 122 / 255, 101 / 255],      # skin
    [0.6863, 0.8824, 0.6863],               # air cavities
])


def view_mri(t1, t2, mri2mni):
    """Show the T1 (and the T2, when given) in the slice viewer."""
    viewers = [sliceshow(NiftiVolume.load(t1).img, color="gray", mri2mni=mri2mni,
                         fig_name="MRI: Click anywhere to navigate.")]
    if t2:
        viewers.append(sliceshow(NiftiVolume.load(t2).img, color="gray", mri2mni=mri2mni,
                                 fig_name="MRI: T2. Click anywhere to navigate."))
    return viewers


def view_seg(mask, mri2mni):
    """Show the tissue segmentation in the slice viewer."""
    return sliceshow(np.asarray(mask.img), color=ListedColormap(SEGMENTATION_COLORS),
                     label="Tissue index", mri2mni=mri2mni,
                     fig_name="Segmentation. Click anywhere to navigate.")


def add_isosurface(axis, volume, level=0.5, color="w", alpha=1.0, step=2):
    """Draw the iso-surface of a volume into a 3-D axis.

    ``step`` down-samples the marching-cubes grid; the default keeps large head
    volumes interactive.
    """
    from skimage import measure

    volume = np.asarray(volume, dtype=float)
    if not volume.any():
        return None
    verts, faces, _, _ = measure.marching_cubes(volume, level, step_size=step)
    mesh = Poly3DCollection(verts[faces], alpha=alpha)
    mesh.set_facecolor(color)
    mesh.set_edgecolor("none")
    axis.add_collection3d(mesh)
    return verts


def add_mesh_surface(axis, node, faces, values=None, cmap="jet", clim=None, alpha=1.0):
    """Draw a triangulated surface, optionally coloured by per-node values."""
    node = np.asarray(node, dtype=float)[:, :3]
    faces = np.asarray(faces, dtype=int)[:, :3] - 1        # to 0-based
    triangles = node[faces]
    mesh = Poly3DCollection(triangles, alpha=alpha)
    if values is None:
        mesh.set_facecolor("0.7")
    else:
        values = np.asarray(values, dtype=float)
        face_values = np.nanmean(values[faces], axis=1)
        if clim is None:
            clim = (np.nanmin(face_values), np.nanmax(face_values))
        span = clim[1] - clim[0]
        normalised = np.clip((face_values - clim[0]) / (span if span else 1.0), 0, 1)
        normalised = np.nan_to_num(normalised)
        mesh.set_facecolor(plt.get_cmap(cmap)(normalised))
    mesh.set_edgecolor("none")
    axis.add_collection3d(mesh)
    return mesh


def _equalise(axis, points):
    """Give the three axes the same scale, so the head is not distorted."""
    points = np.asarray(points, dtype=float)
    low, high = points.min(axis=0), points.max(axis=0)
    center = (low + high) / 2
    radius = float(np.max(high - low)) / 2 or 1.0
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)


def view_electrodes(mask, elec, gel, landmarks, geom, uni_tag, step: int = 2):
    """Render skin, brain, electrodes, gel and the anatomical landmarks together.

    Tissue labels follow the ROAST convention (5 = skin, 2 = gray matter); the
    landmarks are drawn in red and labelled.
    """
    mask_img = np.asarray(mask.img)
    skin = ndimage.gaussian_filter((mask_img == 5).astype(np.float32), 1)
    brain = ndimage.gaussian_filter((mask_img == 2).astype(np.float32), 1)
    elec_img = ndimage.gaussian_filter((np.asarray(elec.img) > 0).astype(np.float32), 1)
    gel_img = ndimage.gaussian_filter((np.asarray(gel.img) > 0).astype(np.float32), 1)

    figure = plt.figure(f"Electrode placement in Simulation: {uni_tag}",
                        figsize=(12, 8), facecolor="white")
    axis = figure.add_subplot(111, projection="3d")

    add_isosurface(axis, skin, 0.5, (229 / 255, 181 / 255, 161 / 255), 0.2, step)
    add_isosurface(axis, brain, 0.5, (1.0, 0.6, 0.8), 1.0, step)
    add_isosurface(axis, elec_img, 0.5, "blue", 0.8, 1)
    add_isosurface(axis, gel_img, 0.5, "green", 0.8, 1)

    if landmarks is not None and len(landmarks):
        landmarks = np.asarray(landmarks, dtype=float)[:4]
        axis.scatter(landmarks[:, 0], landmarks[:, 1], landmarks[:, 2],
                     s=120, c="red", depthshade=False)
        for point, name in zip(landmarks, ("Nasion", "Inion", "Right Ear", "Left Ear")):
            axis.text(point[0], point[1], point[2], "  " + name, color="red",
                      fontsize=12, fontweight="bold")

    _equalise(axis, np.argwhere(mask_img > 0))
    axis.set_box_aspect(np.abs(geom.voxel_size))
    axis.view_init(elev=20, azim=-70)
    axis.set_axis_off()
    axis.set_title(f"Electrode placement in Simulation: {uni_tag}")
    return figure
