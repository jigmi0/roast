"""Validation and normalisation of the user-facing options.

This is the front half of the MATLAB ``roast``/``roast_target`` functions: it
turns the flexible argument shapes the toolbox accepts into the canonical
structures the rest of the pipeline works with, and rejects contradictory
combinations with the same messages as the original.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..config import (CAP_TYPES, DEFAULT_CONDUCTIVITIES, DEFAULT_ELEC_SIZE,
                      DEFAULT_MESH_OPTIONS, ELEC_TYPES)
from ..electrodes.preproc import ElecPara
from ..io.caps import sheet_for_cap_type
from ..utils.logging import get_logger

__all__ = ["RoastOptions", "TargetOptions", "parse_recipe", "build_elec_para",
           "validate_mesh_options", "validate_conductivities", "is_new_options"]

logger = get_logger()

_PAD_ORIENTATIONS = ("lr", "ap", "si")
_SIZE_HELP = ("Unrecognized electrode sizes. Please specify as [radius height] for disc, "
              "[length width height] for pad, and [innerRadius outterRadius height] for "
              "ring electrode.")
_ORI_HELP = ("Unrecognized pad orientation. Please enter 'lr', 'ap', or 'si' for pad "
             "orientation; or just enter the direction vector of the long axis of the pad")


# ---------------------------------------------------------------------------
# recipe
# ---------------------------------------------------------------------------
def parse_recipe(recipe):
    """Split a recipe into electrode names and injected currents.

    Returns ``(names, currents, is_lead_field)``; for the special ``leadField``
    recipe the names come from ``elec72.loc`` instead.
    """
    if isinstance(recipe, str):
        if recipe.lower() != "leadfield":
            raise ValueError("Unrecognized recipe. Use electrodeName-injectedCurrent "
                             "pairs, or the string 'leadField'.")
        return None, None, True
    if recipe is None:
        recipe = ["Fp1", 1, "P4", -1]
    if len(recipe) % 2 != 0:
        raise ValueError("Unrecognized format of your recipe. Please enter as "
                         "electrodeName-injectedCurrent pair.")
    names = [str(recipe[i]) for i in range(0, len(recipe), 2)]
    currents = np.array([float(recipe[i]) for i in range(1, len(recipe), 2)])
    if abs(currents.sum()) > np.finfo(float).eps:
        raise ValueError("Electric currents going in and out of the head not balanced. "
                         "Please make sure they sum to 0.")
    return names, currents, False


# ---------------------------------------------------------------------------
# electrodes
# ---------------------------------------------------------------------------
def _check_size_row(row, elec_type):
    row = np.asarray(row, dtype=float).ravel()
    if np.any(row <= 0):
        raise ValueError("Please enter non-negative values for electrode size.")
    if row.size not in (2, 3):
        raise ValueError(_SIZE_HELP)
    if elec_type == "disc" and row.size == 3:
        raise ValueError("Redundant size info for Disc electrodes. Please enter as "
                         "[radius height]")
    if elec_type in ("pad", "ring") and row.size == 2:
        raise ValueError("Insufficient size info for Pad or Ring electrodes. Please "
                         "specify as [length width height] for pad, and [innerRadius "
                         "outterRadius height] for ring electrode.")
    if elec_type == "pad":
        if row[0] < row[1]:
            raise ValueError("For Pad electrodes, the width of the pad should not be "
                             "bigger than its length. Please enter as "
                             "[length width height]")
        if row[2] < 3:
            raise ValueError("For Pad electrodes, the thickness should at least be 3 mm.")
        if np.any(row > 80):
            logger.warning("You're placing large pad electrodes (one of its dimensions "
                           "is bigger than 8 cm). For large pads, the size will not be "
                           "exact in the model because they will be bent to fit the "
                           "scalp surface.")
    if elec_type == "ring" and row[0] >= row[1]:
        raise ValueError("For Ring electrodes, the inner radius should be smaller than "
                         "outter radius. Please enter as [innerRadius outterRadius height]")
    return row


def _check_ori(value, elec_type):
    if elec_type != "pad":
        return None
    if value is None:
        return "lr"
    if isinstance(value, str):
        if value.lower() not in _PAD_ORIENTATIONS:
            raise ValueError(_ORI_HELP)
        return value.lower()
    array = np.atleast_2d(np.asarray(value, dtype=float))
    if array.shape[1] != 3:
        raise ValueError(_ORI_HELP)
    return array


def build_elec_para(elec_name: Sequence[str], cap_type="1010", elec_type="disc",
                    elec_size=None, elec_ori=None):
    """Normalise the electrode options into one :class:`ElecPara` per type.

    A single entry is returned when all electrodes share a type (the sizes may
    still differ row by row); otherwise one entry per electrode.
    """
    if cap_type is None:
        cap_type = "1010"
    if str(cap_type).lower() not in CAP_TYPES:
        raise ValueError("Supported cap types are: '1020', '1010', '1005', 'BioSemi' "
                         "and 'EGI'.")

    n_elec = len(elec_name)
    uniform = isinstance(elec_type, str)
    if uniform:
        if elec_type.lower() not in ELEC_TYPES:
            raise ValueError("Supported electrodes are: 'disc', 'pad' and 'ring'.")
        elec_type = elec_type.lower()
    else:
        elec_type = [str(t).lower() for t in elec_type]
        if len(elec_type) != n_elec:
            raise ValueError(
                "You want to place more than 1 type of electrodes, but did not tell "
                "ROAST which type for each electrode. Please provide the type for each "
                "electrode respectively, as the value for option 'elecType', in a list "
                "of length equal to the number of electrodes to be placed.")
        for one in elec_type:
            if one not in ELEC_TYPES:
                raise ValueError("Supported electrodes are: 'disc', 'pad' and 'ring'.")

    if uniform:
        if elec_size is None:
            sizes = np.atleast_2d(DEFAULT_ELEC_SIZE[elec_type])
        else:
            sizes = np.atleast_2d(np.asarray(elec_size, dtype=float))
            if sizes.shape[0] > 1 and sizes.shape[0] != n_elec:
                raise ValueError("You want different sizes for each electrode. Please "
                                 "tell ROAST the size for each electrode respectively, "
                                 "in an N-row matrix, where N is the number of "
                                 "electrodes to be placed.")
            sizes = np.vstack([_check_size_row(row, elec_type) for row in sizes])

        if elec_type != "pad" and elec_ori is not None:
            logger.warning("You're not placing pad electrodes; customized orientation "
                           "options will be ignored.")
            elec_ori = None
        orientation = _check_ori(elec_ori, elec_type)
        if isinstance(orientation, np.ndarray) and orientation.shape[0] > 1 \
                and orientation.shape[0] != n_elec:
            raise ValueError("You want different orientations for each pad electrode. "
                             "Please tell ROAST the orientation for each pad "
                             "respectively, in an N-by-3 matrix, where N is the number "
                             "of pads to be placed.")
        return [ElecPara(cap_type=cap_type, elec_type=elec_type, elec_size=sizes,
                         elec_ori=orientation)]

    # Mixed electrode types: every option must be given per electrode.
    if elec_size is None:
        elec_size = [None] * n_elec
    elif not isinstance(elec_size, (list, tuple)) or len(elec_size) != n_elec:
        raise ValueError(
            "You want to place more than 1 type of electrodes. Please tell ROAST the "
            "size for each electrode respectively, as the value for option 'elecSize', "
            "in a list of length equal to the number of electrodes to be placed.")
    if elec_ori is None or isinstance(elec_ori, str) or np.ndim(elec_ori) == 2:
        shared = elec_ori
        elec_ori = []
        pad_rows = np.atleast_2d(shared) if (shared is not None
                                             and not isinstance(shared, str)) else None
        pad_index = 0
        for one in elec_type:
            if one != "pad":
                elec_ori.append(None)
            elif pad_rows is None:
                elec_ori.append(shared)
            else:
                row = pad_rows[pad_index % pad_rows.shape[0]]
                elec_ori.append(row)
                pad_index += 1
    elif len(elec_ori) != n_elec:
        raise ValueError(
            "You want to place another type of electrodes aside from pad. Please tell "
            "ROAST the orientation for each electrode respectively, as the value for "
            "option 'elecOri', in a list of length equal to the number of electrodes to "
            "be placed (use None for non-pad electrodes).")

    paras = []
    for i, one_type in enumerate(elec_type):
        size = elec_size[i]
        size = (np.atleast_2d(DEFAULT_ELEC_SIZE[one_type]) if size is None
                else np.atleast_2d(_check_size_row(size, one_type)))
        if size.shape[0] > 1:
            raise ValueError("You're placing more than 1 type of electrodes. Please put "
                             "size info for each electrode as a single row.")
        paras.append(ElecPara(cap_type=cap_type, elec_type=one_type, elec_size=size,
                              elec_ori=_check_ori(elec_ori[i], one_type)))
    return paras


# ---------------------------------------------------------------------------
# mesh and conductivities
# ---------------------------------------------------------------------------
def validate_mesh_options(mesh_opt):
    """Fill in the defaults and reject unknown or non-positive mesh options."""
    if mesh_opt is None:
        return dict(DEFAULT_MESH_OPTIONS)
    if not isinstance(mesh_opt, dict):
        raise ValueError("Unrecognized format of mesh options. Please enter as a dict "
                         "with keys 'radbound', 'angbound', 'distbound', 'reratio' and "
                         "'maxvol'.")
    unknown = set(mesh_opt) - set(DEFAULT_MESH_OPTIONS)
    if unknown or not mesh_opt:
        raise ValueError("Unrecognized mesh options detected. Supported mesh options are "
                         "'radbound', 'angbound', 'distbound', 'reratio' and 'maxvol'.")
    out = dict(DEFAULT_MESH_OPTIONS)
    for key, value in mesh_opt.items():
        if not np.isscalar(value) or value <= 0:
            raise ValueError(f"Please enter a positive number for the mesh option "
                             f"'{key}'.")
        out[key] = value
    logger.warning("You're changing the advanced options of ROAST. Unless you know what "
                   "you're doing, please keep mesh options default.")
    return out


def validate_conductivities(conductivities, n_elec: int):
    """Fill in the defaults and expand the per-electrode conductivities."""
    out = dict(DEFAULT_CONDUCTIVITIES)
    if conductivities is not None:
        if not isinstance(conductivities, dict):
            raise ValueError("Unrecognized format of conductivity values. Please enter "
                             "as a dict with keys 'white', 'gray', 'csf', 'bone', "
                             "'skin', 'air', 'gel' and 'electrode'.")
        unknown = set(conductivities) - set(DEFAULT_CONDUCTIVITIES)
        if unknown or not conductivities:
            raise ValueError("Unrecognized tissue names detected. Supported tissue names "
                             "in the conductivity option are 'white', 'gray', 'csf', "
                             "'bone', 'skin', 'air', 'gel' and 'electrode'.")
        for key, value in conductivities.items():
            values = np.atleast_1d(np.asarray(value, dtype=float))
            if np.any(values <= 0):
                raise ValueError(f"Please enter a positive number for the {key} "
                                 "conductivity.")
            if key in ("gel", "electrode"):
                if values.size > 1 and values.size != n_elec:
                    raise ValueError(
                        f"You want to assign different conductivities to different "
                        f"{key}s, but didn't tell ROAST clearly which conductivity each "
                        "electrode should use. Please follow the order of electrodes you "
                        "put in 'recipe' to give each of them the corresponding "
                        "conductivity.")
                out[key] = values
            else:
                if values.size > 1:
                    raise ValueError("Tensor conductivity not supported by ROAST. Please "
                                     "enter a scalar value for conductivity.")
                out[key] = float(values[0])
        logger.warning("You're changing the advanced options of ROAST. Unless you know "
                       "what you're doing, please keep conductivity values default.")

    for key in ("gel", "electrode"):
        values = np.atleast_1d(np.asarray(out[key], dtype=float))
        if values.size == 1:
            values = np.repeat(values, n_elec)
        out[key] = values
    return out


# ---------------------------------------------------------------------------
# option records
# ---------------------------------------------------------------------------
@dataclass
class RoastOptions:
    """Everything that identifies one simulation run."""

    config_txt: str
    elec_para: list
    subj_ras_rspd: str
    t2: str | None
    multiaxial: bool
    affine: np.ndarray
    mri2mni: np.ndarray
    landmarks: np.ndarray
    mesh_opt: dict
    conductivities: dict
    unique_tag: str | None
    resamp: bool
    zero_pad: int
    is_non_ras: bool

    def signature(self) -> dict:
        """A canonical, comparable form of the options that define a simulation."""
        cap = str(self.elec_para[0].cap_type)
        cap_class = ("none" if cap.lower() == "none" else sheet_for_cap_type(cap))
        electrodes = []
        for para in self.elec_para:
            electrodes.append({
                "type": str(para.elec_type).lower(),
                "size": np.asarray(para.elec_size, dtype=float).round(6).tolist(),
                "ori": (para.elec_ori.round(6).tolist()
                        if isinstance(para.elec_ori, np.ndarray)
                        else (str(para.elec_ori).lower() if para.elec_ori else None)),
            })
        return {
            "config_txt": self.config_txt,
            "cap": cap_class,
            "electrodes": electrodes,
            "t2": self.t2 or "",
            "multiaxial": bool(self.multiaxial),
            "affine": np.asarray(self.affine, dtype=float).round(6).tolist(),
            "mesh": {k: float(v) for k, v in sorted(self.mesh_opt.items())},
            "conductivities": {k: np.atleast_1d(np.asarray(v, dtype=float)).tolist()
                               for k, v in sorted(self.conductivities.items())},
            "resamp": bool(self.resamp),
            "zero_pad": int(self.zero_pad),
        }


@dataclass
class TargetOptions:
    """Everything that identifies one targeting run."""

    roast_tag: str
    target_coord: np.ndarray
    target_coord_mni: np.ndarray | None
    target_coord_original: np.ndarray | None
    opt_type: str
    desired_intensity: float
    elec_num: int | None
    target_radius: float
    k: float | None
    u0: np.ndarray
    orient: list
    unique_tag: str | None = None

    def signature(self) -> dict:
        opt_type = self.opt_type.lower()
        signature = {
            "roast_tag": self.roast_tag,
            "targets": [
                {"coord": np.asarray(coord, dtype=float).tolist(),
                 "u0": np.asarray(self.u0[i], dtype=float).round(6).tolist(),
                 "orient": (self.orient[i].lower() if isinstance(self.orient[i], str)
                            else np.asarray(self.orient[i], dtype=float).round(6).tolist())}
                for i, coord in enumerate(np.atleast_2d(self.target_coord))],
            "opt_type": opt_type,
            "target_radius": float(self.target_radius),
        }
        if "wls" in opt_type or "lcmv" in opt_type:
            signature["desired_intensity"] = float(self.desired_intensity)
        if opt_type == "max-l1per":
            signature["elec_num"] = int(self.elec_num)
        if "wls" in opt_type:
            signature["k"] = float(self.k)
        return signature


def is_new_options(new, old) -> bool:
    """Whether two option records describe different runs.

    Targets are compared as a set, so re-running the same targets in a different
    order is recognised as the same run.
    """
    new_signature, old_signature = new.signature(), old.signature()
    if "targets" in new_signature:
        new_targets = new_signature.pop("targets")
        old_targets = old_signature.pop("targets")
        if len(new_targets) != len(old_targets):
            return True
        remaining = list(old_targets)
        for target in new_targets:
            if target in remaining:
                remaining.remove(target)
            else:
                return True
    return new_signature != old_signature
