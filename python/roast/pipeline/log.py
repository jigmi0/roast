"""Run bookkeeping: unique tags, option records and the human-readable logs.

Every run is identified by a tag.  ROAST keeps the full option set next to the
subject so that a repeated run with identical options re-uses the previous
results instead of recomputing them, and appends a readable summary to
``<subject>_roastLog`` / ``<subject>_targetLog``.

The MATLAB version stores the options in ``.mat`` files; here they are JSON, so
they can be inspected and diffed without a MATLAB session.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..electrodes.preproc import ElecPara
from ..utils.logging import get_logger
from ..utils.matlab import timestamp_tag
from .options import RoastOptions, TargetOptions, is_new_options

__all__ = ["options_path", "save_options", "load_options", "find_previous_run",
           "write_roast_log", "write_target_log", "write_target_results"]

logger = get_logger()

_ROAST_SUFFIX = "_roastOptions.json"
_TARGET_SUFFIX = "_targetOptions.json"


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _elec_para_to_dict(para: ElecPara) -> dict:
    return {"cap_type": para.cap_type, "elec_type": para.elec_type,
            "elec_size": _jsonable(para.elec_size), "elec_ori": _jsonable(para.elec_ori)}


def _elec_para_from_dict(data: dict) -> ElecPara:
    return ElecPara(cap_type=data["cap_type"], elec_type=data["elec_type"],
                    elec_size=np.asarray(data["elec_size"], dtype=float),
                    elec_ori=(data["elec_ori"] if isinstance(data["elec_ori"], str)
                              else (None if data["elec_ori"] is None
                                    else np.asarray(data["elec_ori"], dtype=float))))


def options_path(subj, tag: str, kind: str) -> Path:
    subj = Path(subj)
    suffix = _ROAST_SUFFIX if kind == "roast" else _TARGET_SUFFIX
    return subj.parent / f"{subj.stem}_{tag}{suffix}"


def save_options(subj, options, kind: str) -> Path:
    """Write the option record of a run."""
    if kind == "roast":
        payload = {
            "config_txt": options.config_txt,
            "elec_para": [_elec_para_to_dict(p) for p in options.elec_para],
            "subj_ras_rspd": str(options.subj_ras_rspd),
            "t2": str(options.t2) if options.t2 else None,
            "multiaxial": bool(options.multiaxial),
            "affine": _jsonable(options.affine),
            "mri2mni": _jsonable(options.mri2mni),
            "landmarks": _jsonable(options.landmarks),
            "mesh_opt": _jsonable(options.mesh_opt),
            "conductivities": _jsonable(options.conductivities),
            "unique_tag": options.unique_tag,
            "resamp": bool(options.resamp),
            "zero_pad": int(options.zero_pad),
            "is_non_ras": bool(options.is_non_ras),
        }
    else:
        payload = {
            "roast_tag": options.roast_tag,
            "target_coord": _jsonable(options.target_coord),
            "target_coord_mni": _jsonable(options.target_coord_mni),
            "target_coord_original": _jsonable(options.target_coord_original),
            "opt_type": options.opt_type,
            "desired_intensity": float(options.desired_intensity),
            "elec_num": options.elec_num,
            "target_radius": float(options.target_radius),
            "k": options.k,
            "u0": _jsonable(options.u0),
            "orient": [o if isinstance(o, str) else _jsonable(o) for o in options.orient],
            "unique_tag": options.unique_tag,
        }
    path = options_path(subj, options.unique_tag, kind)
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_options(path, kind: str):
    """Read back an option record."""
    data = json.loads(Path(path).read_text())
    if kind == "roast":
        return RoastOptions(
            config_txt=data["config_txt"],
            elec_para=[_elec_para_from_dict(p) for p in data["elec_para"]],
            subj_ras_rspd=data["subj_ras_rspd"], t2=data["t2"],
            multiaxial=data["multiaxial"],
            affine=np.asarray(data["affine"], dtype=float),
            mri2mni=np.asarray(data["mri2mni"], dtype=float),
            landmarks=np.asarray(data["landmarks"], dtype=float),
            mesh_opt=data["mesh_opt"],
            conductivities={k: (np.asarray(v, dtype=float) if isinstance(v, list)
                                else v) for k, v in data["conductivities"].items()},
            unique_tag=data["unique_tag"], resamp=data["resamp"],
            zero_pad=data["zero_pad"], is_non_ras=data["is_non_ras"])
    return TargetOptions(
        roast_tag=data["roast_tag"],
        target_coord=np.asarray(data["target_coord"], dtype=float),
        target_coord_mni=(None if data["target_coord_mni"] is None
                          else np.asarray(data["target_coord_mni"], dtype=float)),
        target_coord_original=(None if data["target_coord_original"] is None
                               else np.asarray(data["target_coord_original"], dtype=float)),
        opt_type=data["opt_type"], desired_intensity=data["desired_intensity"],
        elec_num=data["elec_num"], target_radius=data["target_radius"], k=data["k"],
        u0=np.asarray(data["u0"], dtype=float),
        orient=[o if isinstance(o, str) else np.asarray(o, dtype=float)
                for o in data["orient"]],
        unique_tag=data["unique_tag"])


def find_previous_run(subj, options, kind: str):
    """Return the tag of an earlier run with identical options, if there is one."""
    subj = Path(subj)
    suffix = _ROAST_SUFFIX if kind == "roast" else _TARGET_SUFFIX
    for path in sorted(subj.parent.glob(f"{subj.stem}_*{suffix}")):
        try:
            previous = load_options(path, kind)
        except (KeyError, ValueError, json.JSONDecodeError):
            continue
        if not is_new_options(options, previous):
            return previous.unique_tag
    return None


def _assign_tag(subj, options, kind: str) -> str:
    if not options.unique_tag:
        options.unique_tag = timestamp_tag()
    path = options_path(subj, options.unique_tag, kind)
    if path.exists():
        what = "simulation" if kind == "roast" else "targeting"
        raise FileExistsError(
            f"You're about to run a {what} using options that you never used before "
            "(especially if you manually chose different landmarks in the manual GUI), "
            f"but forgot to use a new tag for it. ROAST will get confused when managing "
            f"different {what}s with the same tag. Please use a new tag.")
    save_options(subj, options, kind)
    return options.unique_tag


def _fmt_matrix(matrix, fmt="%.1f") -> str:
    matrix = np.atleast_2d(np.asarray(matrix, dtype=float))
    rows = ["[" + ",".join(fmt % v for v in row) + "]" for row in matrix]
    return "; ".join(rows)


def write_roast_log(subj, options: RoastOptions) -> RoastOptions:
    """Record a simulation run and give it a tag."""
    subj = Path(subj)
    tag = _assign_tag(subj, options, "roast")
    lines = [f"{tag}:",
             f"recipe:\t{options.config_txt}",
             f"capType:\t{options.elec_para[0].cap_type}",
             "elecType:\t" + "\t".join(str(p.elec_type) for p in options.elec_para),
             "elecSize:\t" + "; ".join(_fmt_matrix(p.elec_size) for p in options.elec_para)]

    orientations = []
    for para in options.elec_para:
        if para.elec_ori is None:
            orientations.append("N/A")
        elif isinstance(para.elec_ori, str):
            orientations.append(para.elec_ori)
        else:
            orientations.append(_fmt_matrix(para.elec_ori, "%.4f"))
    lines.append("elecOri:\t" + "; ".join(orientations))

    conductivities = options.conductivities
    lines += [
        f"T2:\t{options.t2 if options.t2 else 'none'}",
        f"multiaxial:\t{'on' if options.multiaxial else 'off'}",
        "meshOpt:\tradbound: {radbound}; angbound: {angbound}; distbound: {distbound}; "
        "reratio: {reratio}; maxvol: {maxvol}".format(**options.mesh_opt),
        "conductivities:\twhite: %.3f; gray: %.3f; CSF: %.3f; bone: %.3f; skin: %.3f; "
        "air: %.1e; " % (conductivities["white"], conductivities["gray"],
                         conductivities["csf"], conductivities["bone"],
                         conductivities["skin"], conductivities["air"])
        + "gel: " + "".join("%.3f; " % v for v in np.atleast_1d(conductivities["gel"]))
        + "electrode: "
        + "".join("%.1e; " % v for v in np.atleast_1d(conductivities["electrode"])),
        f"reSampling:\t{'on' if options.resamp else 'off'}",
        f"zeroPadding:\t{options.zero_pad if options.zero_pad > 0 else 'none'}",
        "original MRI in RAS?:\t" + ("no, and is re-oriented into RAS"
                                     if options.is_non_ras else "yes"),
        f"MRI that was modeled by ROAST:\t{options.subj_ras_rspd}",
    ]
    if options.t2:
        lines.append(f"                         \t{options.t2}")
    lines.append("mri2mni matrix:\t [" + _fmt_matrix(options.mri2mni) + "]")

    with open(subj.parent / f"{subj.stem}_roastLog", "a") as handle:
        handle.write("\n".join(lines) + "\n\n")
    return options


def write_target_log(subj, options: TargetOptions) -> TargetOptions:
    """Record a targeting run and give it a tag."""
    subj = Path(subj)
    tag = _assign_tag(subj, options, "target")
    opt_type = options.opt_type.lower()

    orientations = []
    for one in options.orient:
        orientations.append(one if isinstance(one, str) else _fmt_matrix(one, "%.4f"))

    lines = [
        f"{tag}:",
        f"ROAST simulation tag:\t{options.roast_tag}",
        "targetCoord (in MNI space):\t"
        + (_fmt_matrix(options.target_coord_mni)
           if options.target_coord_mni is not None else "not provided"),
        "targetCoord (in original MRI voxel space):\t"
        + (_fmt_matrix(options.target_coord_original, "%d")
           if options.target_coord_original is not None else "not provided"),
        "targetCoord (in model voxel space):\t" + _fmt_matrix(options.target_coord, "%d"),
        "desired intensity at each target (in V/m, only for max-focality):\t"
        + ("%.3f" % options.desired_intensity
           if ("wls" in opt_type or "lcmv" in opt_type) else "N/A"),
        "desired orientation at each target:\t" + "; ".join(orientations),
        f"targeting algorithm:\t{options.opt_type}",
        "number of electrodes (only for max-l1per):\t"
        + (str(options.elec_num) if options.elec_num else "N/A"),
        "target radius (in mm):\t%d" % options.target_radius,
        "k (only for wls):\t" + ("%.3f" % options.k if options.k else "N/A"),
    ]
    with open(subj.parent / f"{subj.stem}_targetLog", "a") as handle:
        handle.write("\n".join(lines) + "\n")
    return options


def write_target_results(subj, results: dict) -> None:
    """Append the achieved intensities and focalities to the targeting log."""
    subj = Path(subj)
    lines = [
        f"optimal montage:\t{results['montage_txt']}",
        "E-field magnitude at each target (in V/m):\t"
        + ",".join("%.2f" % v for v in np.atleast_1d(results["target_mag"])),
        "E-field intensity along the desired orientation at each target (in V/m):\t"
        + ",".join("%.2f" % v for v in np.atleast_1d(results["target_int"])),
        "Focality of E-field magnitude at each target (in cm):\t"
        + ",".join("%.2f" % v for v in np.atleast_1d(results["target_mag_foc"])),
    ]
    with open(subj.parent / f"{subj.stem}_targetLog", "a") as handle:
        handle.write("\n".join(lines) + "\n\n")
