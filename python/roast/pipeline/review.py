"""``review_res`` - re-open the figures of a finished simulation or targeting.

A thin front-end over the visualisation code: instead of repeating all the
options, the user names the subject and the tag of the run.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ..config import example_dir
from ..electrodes.preproc import ElecPara, elec_preproc
from ..io.caps import read_elec_loc
from ..io.matfile import load_mat, load_spm_mapping
from ..io.nifti import NiftiVolume
from ..utils.logging import banner, get_logger
from .log import load_options, options_path
from .simulate import model_paths

__all__ = ["review_res", "TISSUE_VIEWS"]

logger = get_logger()

#: which surface to render in 3D, and which tissues to keep in the slice views
TISSUE_VIEWS = {
    "white": (1, (1,)),
    "gray": (2, (2,)),
    "csf": (3, (3,)),
    "bone": (4, (4,)),
    "skin": (5, (5,)),
    "air": (6, (6,)),
    "brain": (2, (1, 2)),
    "all": (5, (1, 2, 3, 4, 5, 6)),
}


def _currents_from_config(config_txt: str) -> np.ndarray:
    """Recover the injected currents from the recipe recorded in the log."""
    return np.array([float(value) for value in
                     re.findall(r"\(([-+0-9.eE]+)\s*mA\)", config_txt)])


def review_res(subj=None, sim_tag=None, tissue="brain", fast_render=True, tar_tag=None,
               show=True):
    """Visualise the results of an earlier ``roast`` or ``roast_target`` run."""
    banner("ROAST is an aggregated work by Yu (Andy) Huang licensed under",
           "General Public License version 3 or later. It's supported by",
           "both NIH grants and Soterix Medical Inc.", width=61)

    if subj is None:
        subj = example_dir() / "MNI152_T1_1mm.nii"
    if str(subj).lower() == "nyhead":
        subj = example_dir() / "nyhead.nii"
    subj = Path(subj)
    if not sim_tag:
        raise ValueError(f"Please provide a valid simulation tag for Subject {subj}")

    tissue = str(tissue).lower()
    if tissue not in TISSUE_VIEWS:
        raise ValueError("Supported tissues to be displayed are: "
                         + ", ".join(TISSUE_VIEWS))
    surface_index, slice_tissues = TISSUE_VIEWS[tissue]

    option_file = options_path(subj, sim_tag, "roast")
    if not option_file.exists():
        raise FileNotFoundError(f"Option file not found. Simulation {sim_tag} may never "
                                "be run. Please run it first.")
    opt_roast = load_options(option_file, "roast")
    is_roast = opt_roast.config_txt != "leadFieldGeneration"

    target_results = None
    if is_roast:
        if tar_tag:
            logger.warning("Simulation %s was not run for generating the lead field, so "
                           "targeting tag %s will be ignored.", sim_tag, tar_tag)
        logger.info("Showing results for Simulation %s ...", sim_tag)
        in_current = _currents_from_config(opt_roast.config_txt)
    else:
        if not tar_tag:
            raise ValueError(f"Simulation {sim_tag} was run for generating the lead "
                             "field. reviewRes() will visualize the results from "
                             "roast_target(), but no targeting tag was provided.")
        target_option_file = options_path(subj, tar_tag, "target")
        if not target_option_file.exists():
            raise FileNotFoundError(f"Option file not found. Targeting {tar_tag} may "
                                    "never be run. Please run it first.")
        opt_target = load_options(target_option_file, "target")
        logger.info("Showing results for Targeting %s ...", tar_tag)
        result_file = subj.parent / f"{subj.stem}_{tar_tag}_targetResult.mat"
        if not result_file.exists():
            raise FileNotFoundError(f"Result file {result_file} not found.")
        target_results = load_mat(result_file)
        montage = np.asarray(target_results["mon"], dtype=float).ravel()
        _, ind_in_user_input = elec_preproc(subj, read_elec_loc(),
                                            [ElecPara(cap_type="1010")])
        in_current = montage[ind_in_user_input]

    paths = model_paths(subj, opt_roast.subj_ras_rspd, opt_roast.t2,
                        opt_roast.multiaxial)
    if not paths.mapping.exists():
        raise FileNotFoundError(f"Mapping file {paths.mapping} not found.")
    image = load_spm_mapping(paths.mapping)[0]
    mri2mni = np.asarray(opt_roast.mri2mni, dtype=float)

    if not paths.masks.exists():
        raise FileNotFoundError(f"Segmentation masks {paths.masks} not found.")
    mask = NiftiVolume.load(paths.masks)

    if show and is_roast:
        from ..viz.surfaces import view_electrodes, view_mri, view_seg

        if subj.stem != "nyhead":
            logger.info("showing MRI...")
            if not Path(opt_roast.subj_ras_rspd).exists():
                raise FileNotFoundError(
                    f"The subject MRI {opt_roast.subj_ras_rspd} does not exist. Check if "
                    "you ran through resampling or zero-padding if you tried to do that.")
            view_mri(opt_roast.subj_ras_rspd, opt_roast.t2, mri2mni)
        else:
            logger.info("NEW YORK HEAD selected, there is NO MRI for it to show.")

        logger.info("showing segmentations...")
        view_seg(mask, mri2mni)

        gel_file = subj.parent / f"{subj.stem}_{sim_tag}_mask_gel.nii"
        elec_file = subj.parent / f"{subj.stem}_{sim_tag}_mask_elec.nii"
        for path in (gel_file, elec_file):
            if not path.exists():
                raise FileNotFoundError(f"{path} not found. Check if you ran through "
                                        "electrode placement.")
        logger.info("showing electrode placement...")
        view_electrodes(mask, NiftiVolume.load(elec_file), NiftiVolume.load(gel_file),
                        opt_roast.landmarks, image, sim_tag)
    elif show:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from ..viz.topoplot import topoplot

        colors = plt.get_cmap("jet")(np.linspace(0, 1, 64))
        if opt_target.opt_type.lower() in ("max-l1", "max-l1per"):
            colors[2:62, :3] = 1.0
        figure, axis = plt.subplots(num=f"Montage in Targeting: {tar_tag}",
                                    figsize=(6, 6))
        montage = np.asarray(target_results["mon"], dtype=float).ravel()
        handle = topoplot(montage, axis=axis, cmap=ListedColormap(colors),
                          clim=(montage.min(), montage.max()))
        bar = figure.colorbar(handle, ax=axis)
        bar.set_label("Injected current (mA)", fontsize=14)
        logger.info("Electrodes used are:\n%s",
                    str(target_results.get("montage_txt", "")).replace(", ", "\n"))

    mesh_file = subj.parent / f"{subj.stem}_{sim_tag}.mat"
    if not mesh_file.exists():
        raise FileNotFoundError(f"Mesh file {mesh_file} not found.")
    mesh = load_mat(mesh_file)
    node = np.asarray(mesh["node"], dtype=float)
    elem = np.asarray(mesh["elem"], dtype=np.int64)
    face = np.asarray(mesh["face"], dtype=np.int64)
    if not fast_render:
        logger.info("Smoothing the displayed surface is not implemented; rendering the "
                    "mesh as it is.")

    if is_roast:
        result_file = subj.parent / f"{subj.stem}_{sim_tag}_roastResult.mat"
        if not result_file.exists():
            raise FileNotFoundError(f"Result file {result_file} not found.")
        data = load_mat(result_file)
        payload = {"vol_all": np.asarray(data["vol_all"]),
                   "ef_mag": np.asarray(data["ef_mag"]),
                   "ef_all": np.asarray(data["ef_all"])}
    else:
        payload = {"ef_mag": np.asarray(target_results["ef_mag"]),
                   "ef_all": np.asarray(target_results["ef_all"]),
                   "xopt": np.asarray(target_results["xopt"]),
                   "target_coord": np.asarray(target_results["target_coord"])}

    if show:
        from ..viz.results import visualize_res
        visualize_res(subj, mask, mri2mni, node, elem, face, in_current, image,
                      sim_tag if is_roast else tar_tag, surface_index=surface_index,
                      **payload)
    return payload
