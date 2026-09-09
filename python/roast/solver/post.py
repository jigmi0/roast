"""Post-processing of the getDP solution.

The solver reports values at the mesh nodes; they are interpolated back onto
the regular MRI voxel grid so results can be saved as NIfTI volumes.  When a
lead field is being generated no interpolation happens - the per-electrode node
values are stacked into the transfer matrix that ``roast_target`` consumes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator

from ..io.matfile import save_mat
from ..io.meshfile import read_pos
from ..utils.logging import get_logger

__all__ = ["interpolate_to_grid", "post_getdp"]

logger = get_logger()

_NOT_CONVERGED = "getDP did not converge. Please check getDP before proceeding."


def interpolate_to_grid(points, values, dim):
    """Linear interpolation from scattered mesh nodes onto the voxel grid.

    This is MATLAB's ``TriScatteredInterp``: a Delaunay triangulation of the
    nodes, linear inside the hull and NaN outside it.
    """
    dim = np.asarray(dim, dtype=int)
    interpolator = LinearNDInterpolator(np.asarray(points, dtype=float),
                                        np.asarray(values, dtype=float))
    grid = np.meshgrid(*[np.arange(1, n + 1, dtype=float) for n in dim], indexing="ij")
    query = np.column_stack([g.ravel() for g in grid])
    return interpolator(query).reshape(dim)


def post_getdp(subj, template, node, geom, uni_tag, ind_solved=None):
    """Convert the solver output into volumes (``roast``) or into a lead field
    (``roast`` with the ``leadField`` recipe).

    Returns ``(vol_all, ef_mag, ef_all)`` for a simulation and the lead field
    array ``A_all`` for a lead-field run.
    """
    subj = Path(subj)
    directory = subj.parent
    subj_name = subj.stem
    node = np.array(node, dtype=float, copy=True)

    if template is not None:
        return _post_simulation(directory, subj_name, template, node, geom, uni_tag)
    return _post_lead_field(directory, subj_name, node, uni_tag, ind_solved)


def _post_simulation(directory, subj_name, template, node, geom, uni_tag):
    # Back from pseudo-world to voxel coordinates, so we can interpolate onto
    # the regular grid of the MRI.
    for i in range(3):
        node[:, i] = node[:, i] / geom.mat[i, i]

    logger.info("converting the results into Matlab format...")
    ids, values = read_pos(directory / f"{subj_name}_{uni_tag}_v.pos", 1)
    voltage = values[:, 0] - values[:, 0].min()          # re-reference the voltage
    vol_all = interpolate_to_grid(node[ids - 1, :3], voltage, geom.dim)
    if np.all(np.isnan(vol_all)):
        raise RuntimeError(_NOT_CONVERGED)

    ids, values = read_pos(directory / f"{subj_name}_{uni_tag}_e.pos", 3)
    points = node[ids - 1, :3]
    ef_all = np.zeros(tuple(geom.dim) + (3,))
    for component in range(3):
        ef_all[..., component] = interpolate_to_grid(points, values[:, component],
                                                     geom.dim)
    if np.all(np.isnan(ef_all)):
        raise RuntimeError(_NOT_CONVERGED)
    ef_mag = np.sqrt(np.sum(ef_all ** 2, axis=3))

    logger.info("saving the final results...")
    result_file = directory / f"{subj_name}_{uni_tag}_roastResult.mat"
    save_mat(result_file, {"vol_all": vol_all, "ef_all": ef_all, "ef_mag": ef_mag})

    for data, descrip, suffix, dims in ((vol_all, b"voltage", "v", 3),
                                        (ef_mag, b"EF mag", "emag", 3),
                                        (ef_all, b"EF", "e", 4)):
        out = template.copy()
        out.header.set_data_dtype(np.float32)
        out.set_scaling(1.0, 0.0)        # so that viewers do not rescale the data
        out.header["cal_max"] = 0
        out.header["cal_min"] = 0
        out.set_data(np.asarray(data, dtype=np.float32), descrip=descrip)
        out.save(directory / f"{subj_name}_{uni_tag}_{suffix}.nii")

    logger.info("=" * 54)
    logger.info("Results are saved as:\n%s", result_file)
    logger.info("...and also saved as NIFTI files:")
    logger.info("Voltage: %s", directory / f"{subj_name}_{uni_tag}_v.nii")
    logger.info("E-field: %s", directory / f"{subj_name}_{uni_tag}_e.nii")
    logger.info("E-field magnitude: %s", directory / f"{subj_name}_{uni_tag}_emag.nii")
    logger.info("=" * 54)
    logger.info("You can also find all the results in the following two text files:")
    logger.info("Voltage: %s", directory / f"{subj_name}_{uni_tag}_v.pos")
    logger.info("E-field: %s", directory / f"{subj_name}_{uni_tag}_e.pos")
    logger.info("=" * 54)
    logger.info("Look up the detailed info for this simulation in the log file:\n%s\n"
                "under the simulation tag \"%s\".", directory / f"{subj_name}_roastLog",
                uni_tag)
    logger.info("=" * 54)
    return vol_all, ef_mag, ef_all


def _post_lead_field(directory, subj_name, node, uni_tag, ind_solved):
    ind_solved = np.asarray(ind_solved, dtype=int).ravel()
    a_all = np.full((len(node), 3, len(ind_solved)), np.nan)

    logger.info("assembling the lead field...")
    for i, index in enumerate(ind_solved):
        logger.info("packing electrode %d out of %d ...", i + 1, len(ind_solved))
        pos_file = directory / f"{subj_name}_{uni_tag}_e{index}.pos"
        ids, values = read_pos(pos_file, 3)
        a_all[ids - 1, :, i] = values
        pos_file.unlink(missing_ok=True)      # to save disk space

    if np.all(np.isnan(a_all)):
        raise RuntimeError("getDP did not converge for all the electrodes. "
                           "Please check getDP before proceeding.")

    logger.info("saving the final results...")
    result_file = directory / f"{subj_name}_{uni_tag}_roastResult.mat"
    save_mat(result_file, {"A_all": a_all})

    logger.info("=" * 54)
    logger.info("The lead field matrix is saved as:\n%s", result_file)
    logger.info("Look up the detailed info for this simulation in the log file:\n%s\n"
                "under the simulation tag \"%s\".",
                directory / f"{subj_name}_roastLog", uni_tag)
    logger.info("=" * 54)
    logger.info("Now you can do targeting by calling roast_target(subj, sim_tag, "
                "target_coord, ...)")
    logger.info("=" * 54)
    return a_all
