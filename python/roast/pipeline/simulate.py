"""``roast`` - the main entry point of the simulator.

Runs the six steps of the pipeline for one subject and one recipe:

1. segment the MRI (SPM12, or the Multiaxial CNN),
2. clean the segmentation up (SPM path) or register to MNI (Multiaxial path),
3. place the electrodes,
4. mesh the head,
5. solve the volume conductor model with getDP,
6. save and visualise the results.

Each step is skipped when its output is already on disk, so an interrupted run
can simply be repeated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import LANDMARKS_IN_TPM, example_dir
from ..electrodes.preproc import elec_preproc
from ..electrodes.placement import electrode_placement
from ..geometry.pointcloud import mask_to_edge_point_cloud
from ..io.caps import read_cap_info, read_elec_loc
from ..io.matfile import load_mat, load_spm_mapping, save_spm_mapping
from ..io.nifti import NiftiVolume
from ..mesh.build import mesh_by_iso2mesh
from ..preprocess.header import align_header_to_mni, rename_spm_res
from ..preprocess.orientation import convert_to_ras
from ..preprocess.padding import zero_padding
from ..preprocess.realign import realign_t2
from ..preprocess.resample import resamp_to_one_mm
from ..segment.multiaxial import run_multiaxial
from ..segment.niftyreg import run_nifty_reg
from ..segment.spm import start_seg
from ..segment.touchup import seg_touchup
from ..solver.getdp import solve_by_getdp
from ..solver.post import post_getdp
from ..solver.prepare import prepare_for_getdp
from ..utils.logging import banner, get_logger, step_banner
from .log import find_previous_run, write_roast_log
from .options import (RoastOptions, build_elec_para, parse_recipe,
                      validate_conductivities, validate_mesh_options)

__all__ = ["roast", "ModelPaths", "model_paths"]

logger = get_logger()

_LICENSE_BANNER = (
    "ROAST is an aggregated work by Yu (Andy) Huang licensed under",
    "General Public License version 3 or later. It's supported by",
    "both NIH grants and Soterix Medical Inc.")

_BAD_HEADER = ("The MRI has a bad header. SPM cannot generate the segmentation properly "
               "for MRI with bad header. You can manually align the MRI in SPM Display "
               "function to fix the header.")


@dataclass
class ModelPaths:
    """The chain of file names the pipeline derives from the input MRI."""

    subj: Path                 # the MRI as the user gave it
    model: Path                # after RAS conversion, resampling and padding
    spm: Path                  # name the SPM outputs are filed under
    seg: Path                  # name the segmentation is filed under
    t2: Path | None

    @property
    def masks(self) -> Path:
        return self.seg.parent / (self.seg.stem + "_masks.nii")

    @property
    def mapping(self) -> Path:
        """``_seg8.mat`` (SPM) or ``_niftyReg.mat`` (Multiaxial)."""
        return self._mapping

    @mapping.setter
    def mapping(self, value):
        self._mapping = value


def model_paths(subj, model, t2, multiaxial: bool) -> ModelPaths:
    """Derive the intermediate file names, mirroring the MATLAB naming scheme."""
    subj, model = Path(subj), Path(model)
    directory = model.parent
    suffix = model.suffix
    spm = directory / (model.stem + ("_T1andT2" if t2 else "_T1orT2") + suffix)
    if multiaxial:
        seg = directory / (model.stem + "_multiaxial" + suffix)
        mapping = directory / (model.stem + "_niftyReg.mat")
    else:
        seg = directory / (spm.stem + "_SPM" + suffix)
        mapping = directory / (spm.stem + "_seg8.mat")
    paths = ModelPaths(subj=subj, model=model, spm=spm, seg=seg,
                       t2=Path(t2) if t2 else None)
    paths.mapping = mapping
    return paths


def _preprocess_mri(subj, t2, do_resamp, padding_amount, multiaxial):
    """Re-orient, resample, pad and (if needed) align the T2."""
    volume = NiftiVolume.load(subj)
    if volume.has_bad_header():
        raise ValueError(_BAD_HEADER)

    if np.any(volume.resolution < 0.8) and not do_resamp:
        logger.warning("The MRI has higher resolution (<0.8mm) in at least one "
                       "direction. This will make the modeling process more "
                       "computationally expensive and thus slower. If you wish to run "
                       "faster using just 1-mm model, you can ask ROAST to re-sample the "
                       "MRI into 1 mm first, by turning on the 'resampling' option.")
    if len(np.unique(volume.resolution)) > 1 and not do_resamp:
        logger.warning("The MRI has anisotropic resolution. It is highly recommended "
                       "that you turn on the 'resampling' option, as the electrode size "
                       "will not be exact if the model is built from an MRI with "
                       "anisotropic resolution.")

    subj_ras, is_non_ras = convert_to_ras(subj)
    subj_ras_rs, do_resamp = resamp_to_one_mm(subj_ras, do_resamp)
    model = zero_padding(subj_ras_rs, padding_amount) if padding_amount > 0 else subj_ras_rs

    if t2:
        t2 = realign_t2(t2, model)
    return model, t2, is_non_ras, do_resamp


def _prepare_nyhead(subj, padding_amount, t2, multiaxial, manual_gui, do_resamp):
    """The New York head ships pre-segmented, so most of the pipeline is skipped."""
    masks = example_dir() / "nyhead_T1orT2_SPM_masks.nii"
    if not masks.exists():
        import zipfile
        with zipfile.ZipFile(str(masks) + ".zip") as archive:
            archive.extractall(example_dir())

    logger.warning("New York head selected. Note the New York head is a 0.5 mm model so "
                   "is more computationally expensive. Make sure you have a decent "
                   "machine (>50GB memory) to run ROAST with New York head.")
    if do_resamp:
        raise ValueError("The beauty of New York head is its 0.5 mm resolution. It's a "
                         "bad practice to resample it into 1 mm. Use another head "
                         "'example/MNI152_T1_1mm.nii' for 1 mm model.")

    if padding_amount > 0:
        zero_padding(masks, padding_amount)
        model = example_dir() / f"nyhead_padded{padding_amount}.nii"
        seg8 = example_dir() / f"nyhead_padded{padding_amount}_T1orT2_seg8.mat"
        if not seg8.exists():
            image, tpm, affine = load_spm_mapping(example_dir()
                                                  / "nyhead_T1orT2_seg8.mat")
            origin = np.linalg.solve(image.mat, [0.0, 0.0, 0.0, 1.0])[:3] + padding_amount
            image.mat[:3, 3] = -image.mat[:3, :3] @ origin
            image.dim = image.dim + padding_amount * 2
            save_spm_mapping(seg8, image, tpm, affine)
    else:
        model = Path(subj)

    if t2:
        logger.warning("New York head selected. Any specified T2 image will be ignored.")
        t2 = None
    if multiaxial:
        logger.warning("New York head selected. Multiaxial option will be ignored.")
        multiaxial = False
    if manual_gui:
        logger.warning("New York head selected. The manual GUI will be disabled.")
        manual_gui = False
    return model, t2, multiaxial, manual_gui


def _segment(paths: ModelPaths, t2, multiaxial: bool):
    """Run (or skip) the segmentation and the mapping to MNI space."""
    if not multiaxial:
        if not paths.mapping.exists():
            step_banner(1, 6, "SEGMENT THE MRI BY SPM ...")
            start_seg(paths.model, t2)
            rename_spm_res(paths.model, paths.spm)
        else:
            banner("MRI SEGMENTED BY SPM, SKIP STEP 1")
        if not paths.masks.exists():
            step_banner(2, 6, "SPM SEGMENTATION TOUCHUP ...")
            seg_touchup(paths.spm, paths.seg)
        else:
            banner("SEGMENTATION TOUCHUP DONE, SKIP STEP 2")
    else:
        if not paths.masks.exists():
            step_banner(1, 6, "MULTIAXIAL SEGMENTATION ...")
            run_multiaxial(paths.model)
        else:
            banner("MULTIAXIAL SEGMENTATION DONE, SKIP STEP 1")
        if not paths.mapping.exists():
            step_banner(2, 6, "NIFTYREG REGISTRATION ...")
            run_nifty_reg(paths.model)
        else:
            banner("NIFTYREG REGISTRATION DONE, SKIP STEP 2")
    return load_spm_mapping(paths.mapping)


def _landmarks_from_mapping(image, tpm, affine):
    """Map the template landmarks into the subject's voxel space."""
    tpm2mri = np.linalg.inv(image.mat) @ np.linalg.inv(affine) @ tpm.mat
    homogeneous = np.column_stack([LANDMARKS_IN_TPM, np.ones(len(LANDMARKS_IN_TPM))])
    landmarks = (tpm2mri @ homogeneous.T).T[:, :3]
    return np.round(landmarks)


def _refine_landmarks(paths, mask, image, tpm, affine, landmarks, multiaxial, show):
    """Let the user check the landmarks and, if they change, re-estimate the affine."""
    from ..electrodes.cap import fit_cap_to_individual
    from ..viz.landmarks import check_landmarks

    updated = np.array(landmarks, dtype=float, copy=True)
    updated[:4, :] = check_landmarks(mask, landmarks[:4, :], show=show)
    if np.array_equal(updated[:4, :], landmarks[:4, :]):
        return landmarks, affine, False

    banner("New landmarks detected. ROAST will use these new landmarks to re-run "
           "registration, overwrite the registration computed by SPM or niftyReg, and "
           "reset the headers in _MNI images.")
    logger.info("RE-RUNNING REGISTRATION ...")

    # Estimate the affine from the four clicked landmarks plus the scalp centre
    # and the fitted 10-10 electrodes on the central sagittal line.
    scalp = np.asarray(mask.img) > 0
    scalp_surface, _ = mask_to_edge_point_cloud(scalp, "erode", np.ones((3, 3, 3)))
    cap_info = read_cap_info("1010")
    fitted, center = fit_cap_to_individual(scalp, scalp_surface, updated[:4, :], image,
                                           cap_info, [], False, False)
    updated[7:, :] = fitted
    updated[6, :] = center

    ind = np.array([0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    source = np.column_stack([updated[ind, :], np.ones(len(ind))]).T
    target = np.column_stack([LANDMARKS_IN_TPM[ind, :], np.ones(len(ind))]).T
    mri2mni_vox = target @ np.linalg.pinv(source)
    affine = tpm.mat @ mri2mni_vox @ np.linalg.inv(image.mat)

    logger.info("OVERWRITING COMPUTED REGISTRATION ...")
    save_spm_mapping(paths.mapping, image, tpm, affine)

    # The neck landmarks are neither clicked nor fitted, so re-derive them.
    tpm2mri = np.linalg.inv(image.mat) @ np.linalg.inv(affine) @ tpm.mat
    neck = np.column_stack([LANDMARKS_IN_TPM[4:6, :], np.ones(2)])
    updated[4:6, :] = (tpm2mri @ neck.T).T[:, :3]
    return np.round(updated), affine, True


def roast(subj=None, recipe=None, *, cap_type="1010", elec_type="disc", elec_size=None,
          elec_ori=None, T2=None, multiaxial=False, manual_gui=False, mesh_options=None,
          conductivities=None, simulation_tag=None, resampling=False, zero_pad=0,
          show=True):
    """Run a transcranial electric stimulation simulation.

    Parameters
    ----------
    subj:
        Path of the subject's MRI (T1 or T2).  Defaults to the MNI152 head;
        pass ``'nyhead'`` for the New York head.
    recipe:
        ``[name, current, name, current, ...]`` with the currents in mA summing
        to zero, or the string ``'leadField'`` to generate a lead field for
        :func:`roast.roast_target`.
    cap_type, elec_type, elec_size, elec_ori:
        Electrode layout, shape, size (mm) and pad orientation.
    T2:
        Optional second image aiding the segmentation.
    multiaxial:
        Use the deep-learning segmentation instead of SPM12.
    manual_gui:
        Inspect and, if needed, correct the landmarks by hand.
    mesh_options, conductivities:
        Advanced options; see the documentation.
    simulation_tag:
        Name of the run.  Runs with identical options share a tag and re-use
        each other's intermediate files.
    resampling:
        Resample the MRI to 1 mm isotropic first.
    zero_pad:
        Add this many empty slices around the MRI, for electrodes that would
        otherwise fall outside the field of view.
    show:
        Display the figures (set to ``False`` for batch runs).
    """
    banner(*_LICENSE_BANNER, width=61)
    banner("CHECKING INPUTS...")

    if subj is None:
        subj = example_dir() / "MNI152_T1_1mm.nii"
    if str(subj).lower() == "nyhead":
        subj = example_dir() / "nyhead.nii"
    subj = Path(subj)
    is_nyhead = subj.stem == "nyhead"
    if not is_nyhead and not subj.exists():
        raise FileNotFoundError(f"The subject MRI you provided {subj} does not exist.")

    elec_name, inject_current, is_lead_field = parse_recipe(recipe)
    if is_lead_field:
        logger.warning("You specified the 'recipe' as the 'lead field generation'. Nice "
                       "choice! Note all customized options on electrodes are "
                       "overwritten by the defaults. Also this will usually take a long "
                       "time (>1 day) to generate the lead field for all the candidate "
                       "electrodes.")
        elec_name = read_elec_loc()
        cap_type, elec_type, elec_size, elec_ori = "1010", "disc", [6.0, 2.0], None
    elec_para = build_elec_para(elec_name, cap_type, elec_type, elec_size, elec_ori)

    if T2 is not None:
        T2 = Path(T2)
        if not T2.exists():
            raise FileNotFoundError(f"The T2 MRI you provided {T2} does not exist.")
        if NiftiVolume.load(T2).has_bad_header():
            raise ValueError(_BAD_HEADER)
    if multiaxial and T2 is not None:
        raise ValueError("Multiaxial cannot be run with both T1 and T2 images. If you "
                         "meant to use Multiaxial, please only provide T1 image with "
                         "option 'multiaxial' turned on.")

    mesh_opt = validate_mesh_options(mesh_options)
    conductivities = validate_conductivities(conductivities, len(elec_name))
    if zero_pad and (zero_pad <= 0 or int(zero_pad) != zero_pad):
        raise ValueError("Unrecognized option value. Please enter positive integer value "
                         "for option 'zero_pad'. A recommended value is 10.")
    zero_pad = int(zero_pad)

    # -- preprocessing -----------------------------------------------------
    if not is_nyhead:
        model, T2, is_non_ras, resampling = _preprocess_mri(
            subj, T2, bool(resampling), zero_pad, multiaxial)
    else:
        is_non_ras = False
        model, T2, multiaxial, manual_gui = _prepare_nyhead(
            subj, zero_pad, T2, multiaxial, manual_gui, bool(resampling))
    paths = model_paths(subj, model, T2, multiaxial)

    # -- electrodes --------------------------------------------------------
    elec_para, ind_in_user_input = elec_preproc(subj, elec_name, elec_para)
    elec_name = [elec_name[i] for i in ind_in_user_input]
    if not is_lead_field:
        inject_current = inject_current[ind_in_user_input]
        config_txt = ", ".join(f"{name} ({current:g} mA)"
                               for name, current in zip(elec_name, inject_current))
    else:
        config_txt = "leadFieldGeneration"
    conductivities["gel"] = conductivities["gel"][ind_in_user_input]
    conductivities["electrode"] = conductivities["electrode"][ind_in_user_input]
    if len(elec_para) == 1:
        if elec_para[0].elec_size.shape[0] > 1:
            elec_para[0].elec_size = elec_para[0].elec_size[ind_in_user_input, :]
        if isinstance(elec_para[0].elec_ori, np.ndarray) \
                and elec_para[0].elec_ori.shape[0] > 1:
            elec_para[0].elec_ori = elec_para[0].elec_ori[ind_in_user_input, :]
    elif len(elec_para) == len(elec_name):
        elec_para = [elec_para[i] for i in ind_in_user_input]
    else:
        raise RuntimeError("Something is wrong with the electrode options!")

    # -- segmentation and registration -------------------------------------
    if not is_nyhead:
        image, tpm, affine = _segment(paths, T2, multiaxial)
    else:
        banner("NEW YORK HEAD SELECTED, SKIPPING SEGMENTATION & REGISTRATION ...",
               "...AND NO MRI TO SHOW", width=66)
        seg8 = example_dir() / (paths.model.stem + "_T1orT2_seg8.mat")
        image, tpm, affine = load_spm_mapping(seg8)
    landmarks = _landmarks_from_mapping(image, tpm, affine)
    mri2mni = affine @ image.mat          # MRI voxel space to MNI space
    seg_mask = NiftiVolume.load(paths.masks)

    if manual_gui and not is_nyhead:
        landmarks, affine, changed = _refine_landmarks(
            paths, seg_mask, image, tpm, affine, landmarks, multiaxial, show)
        if changed:
            mri2mni = affine @ image.mat
            logger.info("RESETTING HEADERS IN _MNI IMAGES ...")
            align_header_to_mni(paths.model, T2, paths.seg, mri2mni)

    if show and not is_nyhead:
        from ..viz.surfaces import view_mri
        logger.info("VISUALIZING THE MRI... ")
        view_mri(paths.model, T2, mri2mni)
    if show:
        from ..viz.surfaces import view_seg
        logger.info("VISUALIZING THE SEGMENTATION... ")
        view_seg(seg_mask, mri2mni)

    if not (paths.seg.parent / (paths.seg.stem + "_masks_MNI.nii")).exists():
        logger.info("RESETTING HEADERS IN _MNI IMAGES ...")
        align_header_to_mni(paths.model, T2, paths.seg, mri2mni)

    # -- option bookkeeping ------------------------------------------------
    options = RoastOptions(config_txt=config_txt, elec_para=elec_para,
                           subj_ras_rspd=str(paths.model), t2=str(T2) if T2 else None,
                           multiaxial=bool(multiaxial), affine=affine, mri2mni=mri2mni,
                           landmarks=landmarks, mesh_opt=mesh_opt,
                           conductivities=conductivities, unique_tag=simulation_tag,
                           resamp=bool(resampling), zero_pad=zero_pad,
                           is_non_ras=bool(is_non_ras))
    previous = find_previous_run(subj, options, "roast")
    if previous is None:
        options = write_roast_log(subj, options)
    else:
        if options.unique_tag and options.unique_tag != previous:
            logger.warning("The simulation with the same options has been run before "
                           "under tag '%s'. The new tag you specified '%s' will be "
                           "ignored.", previous, options.unique_tag)
        options.unique_tag = previous
    tag = options.unique_tag

    banner(f"ROAST {subj}" if not is_nyhead else "ROAST New York head",
           "USING RECIPE:", config_txt,
           "...and simulation options saved in:",
           str(subj.parent / f"{subj.stem}_roastLog,"), f"under tag: {tag}")

    directory = subj.parent
    stem = subj.stem

    # -- STEP 3: electrode placement ---------------------------------------
    if not (directory / f"{stem}_{tag}_mask_elec.nii").exists():
        step_banner(3, 6, "ELECTRODE PLACEMENT...")
        elec, gel = electrode_placement(subj, seg_mask, image, landmarks, elec_name,
                                        options, tag)
    else:
        banner("ELECTRODE ALREADY PLACED, SKIP STEP 3")
        elec = NiftiVolume.load(directory / f"{stem}_{tag}_mask_elec.nii")
        gel = NiftiVolume.load(directory / f"{stem}_{tag}_mask_gel.nii")
    if show:
        from ..viz.surfaces import view_electrodes
        logger.info("VISUALIZING ELECTRODE PLACEMENT... ")
        view_electrodes(seg_mask, elec, gel, landmarks, image, tag)

    # -- STEP 4: meshing ---------------------------------------------------
    mesh_file = directory / f"{stem}_{tag}.mat"
    if not mesh_file.exists():
        step_banner(4, 6, "MESH GENERATION...")
        node, elem, face = mesh_by_iso2mesh(subj, seg_mask, elec, gel, mesh_opt,
                                            image, tag)
    else:
        banner("MESH ALREADY GENERATED, SKIP STEP 4")
        data = load_mat(mesh_file)
        node, elem, face = (np.asarray(data["node"], dtype=float),
                            np.asarray(data["elem"], dtype=np.int64),
                            np.asarray(data["face"], dtype=np.int64))

    result_file = directory / f"{stem}_{tag}_roastResult.mat"
    if not is_lead_field:
        # -- STEP 5: solve --------------------------------------------------
        if not (directory / f"{stem}_{tag}_v.pos").exists():
            step_banner(5, 6, "SOLVING THE MODEL...")
            prepare_for_getdp(subj, node, elem, elec_name, tag)
            solve_by_getdp(subj, inject_current, conductivities,
                           np.arange(len(elec_name)), tag, "")
        else:
            banner("MODEL ALREADY SOLVED, SKIP STEP 5")

        # -- STEP 6: results ------------------------------------------------
        if not result_file.exists():
            step_banner(6, 6, "SAVING AND VISUALIZING RESULTS...")
            vol_all, ef_mag, ef_all = post_getdp(subj, seg_mask, node, image, tag)
        else:
            banner("ALL STEPS DONE, LOADING RESULTS FOR VISUALIZATION")
            data = load_mat(result_file)
            vol_all, ef_mag, ef_all = (np.asarray(data["vol_all"]),
                                       np.asarray(data["ef_mag"]),
                                       np.asarray(data["ef_all"]))
        if show:
            from ..viz.results import visualize_res
            visualize_res(subj, seg_mask, mri2mni, node, elem, face, inject_current,
                          image, tag, vol_all=vol_all, ef_mag=ef_mag, ef_all=ef_all)
        outputs = (vol_all, ef_mag, ef_all)
    else:
        outputs = _generate_lead_field(subj, directory, stem, tag, elec_name, node, elem,
                                       conductivities, result_file)

    banner("ALL DONE ROAST")
    return {"tag": tag, "options": options, "node": node, "elem": elem, "face": face,
            "results": outputs}


def _generate_lead_field(subj, directory, stem, tag, elec_name, node, elem,
                         conductivities, result_file):
    """Solve once per candidate electrode and assemble the transfer matrix."""
    ind_ref = elec_name.index("Iz")
    ind_stim = np.setdiff1d(np.arange(len(elec_name)), [ind_ref])
    solved = np.array([(directory / f"{stem}_{tag}_e{index + 1}.pos").exists()
                       for index in ind_stim])

    if not solved.all() and not result_file.exists():
        step_banner(5, 6, "GENERATING THE LEAD FIELD...")
        logger.info("NOTE THIS WILL TAKE SOME TIME...")
        prepare_for_getdp(subj, node, elem, elec_name, tag)
        current = np.ones(len(elec_name))       # 1 mA at each candidate electrode
        current[ind_ref] = -1
        for i, index in enumerate(ind_stim):
            if solved[i]:
                logger.info("ELECTRODE %d HAS BEEN SOLVED, SKIPPING...", i + 1)
                continue
            banner(f"SOLVING FOR ELECTRODE {i + 1} OUT OF {len(ind_stim)} ...")
            solve_by_getdp(subj, current, conductivities, [index, ind_ref], tag,
                           str(index + 1))
    else:
        banner("LEAD FIELD ALREADY GENERATED, SKIP STEP 5")

    if not result_file.exists():
        step_banner(6, 6, "ASSEMBLING AND SAVING LEAD FIELD...")
        return post_getdp(subj, None, node, None, tag, ind_stim + 1)
    banner("ALL STEPS DONE, READY TO DO TARGETING",
           f"FOR SUBJECT {subj}", f"USING TAG {tag}")
    return None
