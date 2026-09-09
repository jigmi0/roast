"""Interactive selection and inspection of the anatomical landmarks.

Used when the automatic registration to MNI space fails and the nasion, inion
and pre-auricular points have to be picked by hand.

The MATLAB GUI rotates a 3-D rendering to a fixed, axis-aligned camera for each
landmark and reads two coordinates off the click; the third is then found by
walking along the remaining axis.  The Python version presents that same view as
a projection of the head, so the click carries exactly the same information, and
the search along the third axis is identical.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy import ndimage

from ..config import lib_dir
from ..utils.logging import get_logger
from .surfaces import add_isosurface, _equalise

__all__ = ["get_landmarks_manual", "check_landmarks", "reorder_landmarks"]

logger = get_logger()

#: title, screenshot and pick mode of each step of the manual selection
_STEPS = (
    ("Select the Nasion: The Point in Between the Eyes", "Nasion_New.png", "sagittal"),
    ("Select the Right Ear: In the Front Middle", "Right_Ear_New.png", "coronal"),
    ("Select the Left Ear: In the Front Middle", "Left_Ear_New.png", "coronal"),
    ("Select the Inion: The Skull Begins to Slope Inwards", "Inion1_New.png", "coronal"),
    ("Select the Inion: In the Middle of the Skull", "Inion2_New.png", "sagittal"),
)


def reorder_landmarks(points) -> np.ndarray:
    """Reorder the five manual picks into nasion, inion, right ear, left ear."""
    points = np.asarray(points, dtype=float)
    return points[[0, 4, 1, 2], :]


def _pick_along_y(skin, skull, skin_smooth, x, z, phase):
    """Resolve the anterior/posterior coordinate of a sagittal pick."""
    columns = np.arange(1, skin.shape[1] + 1)
    skin_line = skin[x - 1, :, z - 1]
    skull_line = skull[x - 1, :, z - 1]
    smooth_line = skin_smooth[x - 1, :, z - 1]

    non_zero = columns[skin_line != 0]
    non_zero_smooth = columns[smooth_line != 0]
    non_zero_skull = columns[skull_line != 0]
    if non_zero.size == 0:
        return None, None

    if phase == 0:                                  # nasion: most anterior point
        return int(non_zero.max()), int(non_zero_smooth.max())

    # inion: the first background voxel behind the skull
    def behind_skull(line_zero):
        if non_zero_skull.size and line_zero.size:
            before = np.flatnonzero(line_zero < non_zero_skull.min())
            if before.size:
                return int(line_zero[before[-1]] + 1)
        return None

    zeros = columns[(skin_line == 0) & (skull_line == 0)]
    zeros_smooth = columns[(smooth_line == 0) & (skull_line == 0)]
    y = behind_skull(zeros) or int(non_zero.min())
    y_smooth = behind_skull(zeros_smooth) or int(non_zero_smooth.min())
    return y, y_smooth


def _pick_along_x(skin, skin_smooth, y, z, phase):
    """Resolve the left/right coordinate of a coronal pick."""
    rows = np.arange(1, skin.shape[0] + 1)
    skin_line = skin[:, y - 1, z - 1]
    smooth_line = skin_smooth[:, y - 1, z - 1]
    non_zero = rows[skin_line != 0]
    non_zero_smooth = rows[smooth_line != 0]
    if non_zero.size == 0:
        return None, None
    if phase == 1:                                  # right ear: largest x
        return int(non_zero.max()), int(non_zero_smooth.max())
    return int(non_zero.min()), int(non_zero_smooth.min())


def get_landmarks_manual(mask, show: bool = True):
    """Pick the five landmarks by hand.

    Returns ``(landmarks, smooth_landmarks)`` in the order nasion, right ear,
    left ear, inion (slope), inion (middle); each is a 1-based voxel coordinate.
    """
    logger.info("=============    LOADING GUI FOR MODIFY ...    =============")
    mask_img = np.asarray(mask.img)
    skin = (mask_img == 5).astype(np.float32)
    skull = (mask_img == 4).astype(np.float32)
    skin_smooth = ndimage.gaussian_filter(skin, 1)

    picks, smooth_picks = [], []
    state = {"phase": 0, "candidate": None, "smooth": None}

    figure = plt.figure("Selecting Landmarks ...", figsize=(12, 8))
    view_axis = figure.add_axes([0.05, 0.12, 0.62, 0.80])
    guide_axis = figure.add_axes([0.70, 0.35, 0.28, 0.50])
    guide_axis.set_axis_off()
    submit_axis = figure.add_axes([0.45, 0.02, 0.12, 0.06])
    submit = Button(submit_axis, "Submit")

    def projection(mode):
        # Depth of the outermost skin voxel, viewed along the axis the MATLAB
        # GUI looks down for this step.
        body = np.maximum(skin, skull)
        if mode == "sagittal":                       # looking along y: (x, z)
            return body.max(axis=1)
        return body.max(axis=0)                      # looking along x: (y, z)

    def redraw():
        phase = state["phase"]
        title, screenshot, mode = _STEPS[phase]
        view_axis.clear()
        view_axis.imshow(projection(mode).T, origin="lower", cmap="bone")
        view_axis.set_title(title, fontsize=14)
        view_axis.set_xlabel("x (R-L)" if mode == "sagittal" else "y (P-A)")
        view_axis.set_ylabel("z (I-S)")
        if state["candidate"] is not None:
            point = state["candidate"]
            u = point[0] if mode == "sagittal" else point[1]
            view_axis.plot(u - 1, point[2] - 1, "o", color="deepskyblue", markersize=10)
        guide_axis.clear()
        guide_axis.set_axis_off()
        guide_axis.set_title("Example Selection", fontsize=13, fontweight="bold")
        path = lib_dir() / "screenshots" / screenshot
        if path.exists():
            guide_axis.imshow(plt.imread(str(path)))
        figure.canvas.draw_idle()

    def on_click(event):
        if event.inaxes is not view_axis or event.xdata is None:
            return
        phase = state["phase"]
        mode = _STEPS[phase][2]
        u = int(round(event.xdata)) + 1
        z = int(round(event.ydata)) + 1
        if mode == "sagittal":
            if not (1 <= u <= skin.shape[0] and 1 <= z <= skin.shape[2]):
                return
            y, y_smooth = _pick_along_y(skin, skull, skin_smooth, u, z, phase)
            if y is None:
                logger.info("Click is out of bounds. Please click following the example.")
                return
            state["candidate"] = np.array([u, y, z])
            state["smooth"] = np.array([u, y_smooth, z])
        else:
            if not (1 <= u <= skin.shape[1] and 1 <= z <= skin.shape[2]):
                return
            x, x_smooth = _pick_along_x(skin, skin_smooth, u, z, phase)
            if x is None:
                logger.info("Click is out of bounds. Please click following the example.")
                return
            state["candidate"] = np.array([x, u, z])
            state["smooth"] = np.array([x_smooth, u, z])
        logger.info("Clicked at: %s", state["candidate"])
        redraw()

    def on_submit(_event):
        if state["candidate"] is None:
            return
        picks.append(state["candidate"])
        smooth_picks.append(state["smooth"])
        state["candidate"] = state["smooth"] = None
        state["phase"] += 1
        if state["phase"] >= len(_STEPS):
            plt.close(figure)
            return
        redraw()

    figure.canvas.mpl_connect("button_press_event", on_click)
    submit.on_clicked(on_submit)
    redraw()
    if show:
        plt.show()
    return np.asarray(picks, dtype=float), np.asarray(smooth_picks, dtype=float)


def check_landmarks(mask, landmarks, show: bool = True) -> np.ndarray:
    """Show the current landmarks in 3-D and let the user confirm or redo them.

    Returns the (possibly updated) four landmarks: nasion, inion, right ear and
    left ear.
    """
    logger.info("=============    OPENING MANUAL GUI ...    =============")
    landmarks = np.asarray(landmarks, dtype=float)
    mask_img = np.asarray(mask.img)
    skin = ndimage.gaussian_filter((mask_img == 5).astype(np.float32), 1)
    brain = (mask_img == 2).astype(np.float32)

    result = {"landmarks": landmarks, "modify": False}

    figure = plt.figure("3D Viewer. Please rotate and inspect the landmarks.",
                        figsize=(12, 8))
    axis = figure.add_axes([0.02, 0.10, 0.96, 0.85], projection="3d")
    add_isosurface(axis, skin, 0.5, (229 / 255, 181 / 255, 161 / 255), 0.3)
    add_isosurface(axis, brain, 0.5, (1.0, 0.6, 0.8), 1.0)
    axis.scatter(landmarks[:4, 0], landmarks[:4, 1], landmarks[:4, 2], s=150, c="red",
                 depthshade=False)
    for point, name in zip(landmarks[:4],
                           ("Nasion", "Inion", "Right Ear", "Left Ear")):
        axis.text(point[0], point[1], point[2], "  " + name, color="red", fontsize=12,
                  fontweight="bold")
    _equalise(axis, np.argwhere(mask_img > 0))
    axis.set_axis_off()
    axis.set_title("Selecting Landmarks ...")

    logger.info("Current voxel coordinates of landmarks "
                "(nasion, inion, right ear, left ear):\n%s", landmarks[:4])
    logger.info("3D interaction enabled. Rotate and inspect the landmarks.")

    confirm = Button(figure.add_axes([0.72, 0.02, 0.10, 0.05]), "Confirm")
    modify = Button(figure.add_axes([0.84, 0.02, 0.10, 0.05]), "Modify")
    confirm.on_clicked(lambda _event: plt.close(figure))

    def on_modify(_event):
        result["modify"] = True
        plt.close(figure)

    modify.on_clicked(on_modify)
    if show:
        plt.show()

    if result["modify"]:
        updated, _ = get_landmarks_manual(mask, show=show)
        result["landmarks"] = reorder_landmarks(updated)
        return check_landmarks(mask, result["landmarks"], show=show)
    return np.asarray(result["landmarks"], dtype=float)[:4]
