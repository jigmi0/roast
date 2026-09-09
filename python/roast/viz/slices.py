"""Interactive orthogonal slice viewer.

Port of ``sliceshow``: three orthogonal views of a volume that the user can
click through, optionally with a vector field drawn on top and with the voxel
and MNI coordinates of the current position editable.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.widgets import TextBox

__all__ = ["SliceViewer", "sliceshow", "roast_colormap"]

_ARROW_STEP = 5          # draw one arrow every five voxels


def roast_colormap(n: int = 2 ** 11) -> ListedColormap:
    """The colour map ROAST uses for its result overlays: jet with white below."""
    jet = plt.get_cmap("jet")(np.linspace(0, 1, n))
    colors = np.vstack([[1.0, 1.0, 1.0, 1.0], jet])
    cmap = ListedColormap(colors)
    cmap.set_bad("white")
    return cmap


def _as_cmap(color):
    if color is None:
        cmap = plt.get_cmap("jet").copy()
    elif isinstance(color, str):
        cmap = plt.get_cmap(color).copy()
    elif isinstance(color, ListedColormap):
        cmap = color
    else:
        colors = np.asarray(color, dtype=float)
        cmap = ListedColormap(colors)
    cmap.set_bad("white")
    return cmap


class SliceViewer:
    """Three orthogonal slices through a volume, navigable by clicking.

    Coordinates are 1-based voxel coordinates of the *uncropped* volume, which
    is also what the MNI mapping expects.
    """

    #: which two position components each panel spans
    _AXES = ((0, 2), (1, 2), (0, 1))

    def __init__(self, img, pos=None, color=None, clim=None, label=None,
                 fig_name="", vec_img=None, mri2mni=None, bbox=None):
        img = np.asarray(img, dtype=float)
        if img.ndim != 3:
            raise ValueError("At least give us a volume to display")

        if bbox is None:
            self.offset = np.array([0, 0, 0])
            self.img = img
        else:
            bbox = np.asarray(bbox, dtype=int)
            self.offset = bbox[0] - 1
            self.img = img[bbox[0, 0] - 1:bbox[1, 0], bbox[0, 1] - 1:bbox[1, 1],
                           bbox[0, 2] - 1:bbox[1, 2]]
            if vec_img is not None:
                vec_img = np.asarray(vec_img)[bbox[0, 0] - 1:bbox[1, 0],
                                              bbox[0, 1] - 1:bbox[1, 1],
                                              bbox[0, 2] - 1:bbox[1, 2], :]

        if pos is None:
            pos = np.round(np.asarray(img.shape) / 2).astype(int)
        pos = np.asarray(pos, dtype=int).ravel() - self.offset
        if np.any(pos <= 0):
            raise ValueError("Voxel selected falls outside of the bounding box.")
        self.pos = pos

        if vec_img is not None:
            vec_img = np.asarray(vec_img, dtype=float)
            if vec_img.shape[:3] != self.img.shape or vec_img.shape[3] != 3:
                raise ValueError("Vector field does not have correct size.")
        self.vec_img = vec_img

        if clim is None:
            if np.all(np.isnan(self.img)):
                raise ValueError("The image volume you provided does not have any "
                                 "meaningful values.")
            low, high = np.nanmin(self.img), np.nanmax(self.img)
            self.clim = None if low == high else (float(low), float(high))
        else:
            self.clim = tuple(float(v) for v in clim)

        if mri2mni is not None:
            mri2mni = np.asarray(mri2mni, dtype=float)
            if mri2mni.shape != (4, 4) or not np.allclose(np.round(mri2mni[3]),
                                                          [0, 0, 0, 1]):
                raise ValueError("Unrecognized format of the voxel-to-MNI mapping.")
        self.mri2mni = mri2mni
        self.cmap = _as_cmap(color)
        self.label = label or ""

        self.figure = plt.figure(fig_name, figsize=(8.5, 8.5 / 1.0187))
        self.figure.patch.set_facecolor("w")
        self.axes = [self.figure.add_subplot(2, 2, i + 1) for i in range(4)]
        self._boxes = {}
        self._draw()
        self.figure.canvas.mpl_connect("button_press_event", self._on_click)

    # -- coordinates -------------------------------------------------------
    @property
    def voxel_pos(self) -> np.ndarray:
        """Current position in the coordinates of the uncropped volume."""
        return self.pos + self.offset

    @property
    def mni_pos(self) -> np.ndarray:
        return np.round(self.mri2mni @ np.append(self.voxel_pos, 1.0)).astype(int)[:3]

    # -- drawing -----------------------------------------------------------
    def _draw(self):
        for axis in self.axes:
            axis.clear()
        limit = max(self.img.shape)
        planes = (self.img[:, self.pos[1] - 1, :], self.img[self.pos[0] - 1, :, :],
                  self.img[:, :, self.pos[2] - 1])

        for i, (axis, plane) in enumerate(zip(self.axes[:3], planes)):
            axis.imshow(plane.T, origin="lower", cmap=self.cmap,
                        vmin=None if self.clim is None else self.clim[0],
                        vmax=None if self.clim is None else self.clim[1],
                        extent=(0.5, plane.shape[0] + 0.5, 0.5, plane.shape[1] + 0.5))
            a, b = self._AXES[i]
            axis.plot(self.pos[a], self.pos[b], "o", color="m", linewidth=3,
                      markersize=12, fillstyle="none")
            axis.axvline(self.pos[a], color="k", linewidth=0.8)
            axis.axhline(self.pos[b], color="k", linewidth=0.8)
            axis.set_xlim(0, limit)
            axis.set_ylim(0, limit)
            axis.set_aspect("equal")
            axis.axis("off")
            if self.vec_img is not None:
                self._quiver(axis, i)

        self._info_panel()
        self.figure.canvas.draw_idle()

    def _quiver(self, axis, panel):
        step = _ARROW_STEP
        if panel == 0:
            rows = np.arange(0, self.img.shape[0], step)
            cols = np.arange(0, self.img.shape[2], step)
            u = self.vec_img[np.ix_(rows, [self.pos[1] - 1], cols)][:, 0, :, 0]
            v = self.vec_img[np.ix_(rows, [self.pos[1] - 1], cols)][:, 0, :, 2]
        elif panel == 1:
            rows = np.arange(0, self.img.shape[1], step)
            cols = np.arange(0, self.img.shape[2], step)
            u = self.vec_img[np.ix_([self.pos[0] - 1], rows, cols)][0, :, :, 1]
            v = self.vec_img[np.ix_([self.pos[0] - 1], rows, cols)][0, :, :, 2]
        else:
            rows = np.arange(0, self.img.shape[0], step)
            cols = np.arange(0, self.img.shape[1], step)
            u = self.vec_img[np.ix_(rows, cols, [self.pos[2] - 1])][:, :, 0, 0]
            v = self.vec_img[np.ix_(rows, cols, [self.pos[2] - 1])][:, :, 0, 1]
        grid_x, grid_y = np.meshgrid(rows + 1, cols + 1, indexing="ij")
        # Scale the arrows so the strongest one spans about one sampling step.
        magnitude = float(np.nanmax(np.hypot(u, v))) if u.size else 0.0
        scale = magnitude / (step * 0.9) if magnitude > 0 else None
        axis.quiver(grid_x, grid_y, u, v, color="k", angles="xy", scale_units="xy",
                    scale=scale, width=0.003)

    def _info_panel(self):
        panel = self.axes[3]
        panel.axis("off")
        value = self.img[self.pos[0] - 1, self.pos[1] - 1, self.pos[2] - 1]
        panel.text(0.5, 0.18, self.label, ha="center", fontsize=15, fontweight="bold")
        panel.text(0.5, 0.02, "%.2f" % value, ha="center", fontsize=20,
                   fontweight="bold")

        if not self._boxes:
            self._make_boxes()
        else:
            self._refresh_boxes()

        mappable = self.axes[1].images[0] if self.axes[1].images else None
        if mappable is not None and not getattr(self, "_colorbar", None):
            self._colorbar = self.figure.colorbar(mappable, ax=self.axes[1],
                                                  orientation="horizontal",
                                                  fraction=0.05, pad=0.02)
            self._colorbar.set_label(self.label, fontsize=12)

    def _make_boxes(self):
        labels = ("X", "Y", "Z")
        self.figure.text(0.545, 0.40, "Voxel", fontsize=11, fontweight="bold")
        for i, name in enumerate(labels):
            axis = self.figure.add_axes([0.62 + 0.11 * i, 0.395, 0.07, 0.035])
            box = TextBox(axis, name, initial=str(self.voxel_pos[i]))
            box.on_submit(lambda text, index=i: self._set_voxel(index, text))
            self._boxes[("voxel", i)] = box
        if self.mri2mni is not None:
            self.figure.text(0.555, 0.335, "MNI", fontsize=11, fontweight="bold")
            for i, name in enumerate(labels):
                axis = self.figure.add_axes([0.62 + 0.11 * i, 0.33, 0.07, 0.035])
                box = TextBox(axis, name, initial=str(self.mni_pos[i]))
                box.on_submit(lambda text, index=i: self._set_mni(index, text))
                self._boxes[("mni", i)] = box

    def _refresh_boxes(self):
        for i in range(3):
            self._boxes[("voxel", i)].set_val(str(self.voxel_pos[i]))
            if ("mni", i) in self._boxes:
                self._boxes[("mni", i)].set_val(str(self.mni_pos[i]))

    # -- interaction -------------------------------------------------------
    def _accept(self, pos) -> bool:
        return bool(np.all(pos >= 1) and np.all(pos <= np.asarray(self.img.shape)))

    def _on_click(self, event):
        if event.inaxes not in self.axes[:3] or event.xdata is None:
            return
        panel = self.axes.index(event.inaxes)
        a, b = self._AXES[panel]
        pos = self.pos.copy()
        pos[a] = int(round(event.xdata))
        pos[b] = int(round(event.ydata))
        if self._accept(pos):
            self.pos = pos
            self._draw()

    def _set_voxel(self, index, text):
        try:
            value = int(round(float(text)))
        except ValueError:
            return
        pos = self.pos.copy()
        pos[index] = value - self.offset[index]
        if self._accept(pos):
            self.pos = pos
            self._draw()

    def _set_mni(self, index, text):
        try:
            value = float(text)
        except ValueError:
            return
        mni = self.mni_pos.astype(float)
        mni[index] = value
        voxel = np.round(np.linalg.solve(self.mri2mni, np.append(mni, 1.0)))[:3]
        pos = voxel.astype(int) - self.offset
        if self._accept(pos):
            self.pos = pos
            self._draw()


def sliceshow(img, pos=None, color=None, clim=None, label=None, fig_name="",
              vec_img=None, mri2mni=None, bbox=None, show=False) -> SliceViewer:
    """Display a volume in three orthogonal, clickable views."""
    viewer = SliceViewer(img, pos, color, clim, label, fig_name, vec_img, mri2mni, bbox)
    if show:
        plt.show()
    return viewer
