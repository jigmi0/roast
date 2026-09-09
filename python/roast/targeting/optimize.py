"""Preparation, objective function and gateway of the targeting optimisation.

References:

* Dmochowski, Datta, Bikson, Su, Parra, "Optimized multi-electrode stimulation
  increases focality and intensity at target", J. Neural Eng. 8 (2011) 046011.
* Dmochowski et al., "Targeted transcranial direct current stimulation for
  rehabilitation after stroke", NeuroImage 75 (2013) 12-19.

Any number of target ROIs is supported.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..utils.logging import get_logger
from .currents import optimize_currents

__all__ = ["TargetingProblem", "optimize_prepare", "optimize", "optimize_anon"]

logger = get_logger()


@dataclass
class TargetingProblem:
    """Everything the optimiser needs about one targeting run."""

    target_coord: np.ndarray
    num_of_targets: int
    opt_type: str
    target_radius: float
    desired_intensity: float = 1.0
    elec_num: int | None = None
    k: float | None = None
    i_max: float = 2.0                      # maximum total current, in mA
    u: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    n_locs: int = 0
    target_nodes: list = field(default_factory=list)
    w: np.ndarray = field(default_factory=lambda: np.zeros(0))
    U: np.ndarray | None = None
    S: np.ndarray | None = None
    V: np.ndarray | None = None


def optimize_prepare(p: TargetingProblem, A, locs) -> TargetingProblem:
    """Find the target nodes, build the weights and factor the transfer matrix.

    The SVD is computed once here because every call of the objective function
    reuses it.
    """
    locs = np.asarray(locs, dtype=float)
    A = np.asarray(A, dtype=float)
    n_locs = p.n_locs

    target_nodes = []
    for n in range(p.num_of_targets):
        delta = locs - p.target_coord[n][None, :]
        distances = np.sqrt(np.sum(delta * delta, axis=1))
        nodes = np.flatnonzero(distances < p.target_radius)
        if nodes.size == 0:
            raise ValueError("No nodes found near target. Please increase the value "
                             "of 'targetRadius'.")
        target_nodes.append(nodes)
    p.target_nodes = target_nodes

    all_targets = np.concatenate(target_nodes)
    non_targets = np.setdiff1d(np.arange(n_locs), all_targets)

    logger.info("Preparing to optimize...this may take a moment...")

    if "wls" in p.opt_type:
        n_target = all_targets.size
        n_non_target = n_locs - n_target
        weight_target = n_locs / n_target * (p.k / (p.k + 1))
        weight_other = n_target / n_non_target * (weight_target / p.k)
    else:
        weight_target, weight_other = 1.0, 0.0

    w = np.zeros(n_locs)
    w[all_targets] = weight_target
    w[non_targets] = weight_other
    p.w = np.tile(w, 3)

    if p.opt_type in ("wls-l1", "wls-l1per"):
        p.U, singular, vt = np.linalg.svd(np.sqrt(p.w)[:, None] * A, full_matrices=False)
        p.S, p.V = np.diag(singular), vt.T
    elif p.opt_type in ("lcmv-l1", "lcmv-l1per"):
        p.U, singular, vt = np.linalg.svd(A, full_matrices=False)
        p.S, p.V = np.diag(singular), vt.T
    else:
        p.U = p.S = p.V = None
    return p


def _desired_field(p: TargetingProblem, orientations) -> np.ndarray:
    """Assemble the desired electric field vector over all target ROIs."""
    xd = np.zeros(3 * p.n_locs)
    for n in range(p.num_of_targets):
        nodes = p.target_nodes[n]
        xd[nodes] = p.desired_intensity * orientations[n, 0]
        xd[nodes + p.n_locs] = p.desired_intensity * orientations[n, 1]
        xd[nodes + 2 * p.n_locs] = p.desired_intensity * orientations[n, 2]
    return xd


def optimize(p: TargetingProblem, A) -> np.ndarray:
    """Run the optimisation and return the optimal electrode currents.

    :func:`optimize_prepare` must have been called first.
    """
    xd = _desired_field(p, p.u)
    logger.info("=" * 28)
    logger.info("Performing optimization...")
    logger.info("=" * 28)
    _, s_opt, status = optimize_currents(A, xd, p.i_max, p.w, p.target_nodes,
                                         p.opt_type, p.U, p.S, p.V, p.elec_num, False)
    if status == "Failed":
        logger.warning("Optimization FAILED!!\n Program will continue but results may "
                       "be INACCURATE!")
    return s_opt


def optimize_anon(p: TargetingProblem, t, A) -> float:
    """Objective function for the search over the field orientation at the target.

    ``t`` is an ``n x 2`` array of azimuth/elevation pairs, one per target ROI.
    Lower is better, so the intensity based criteria are negated.
    """
    t = np.atleast_2d(np.asarray(t, dtype=float))
    if t.shape[1] != 2:
        raise ValueError("Orientation variable not in correct format. Please provide "
                         "the orientation in n-by-2 matrix, where n is the number of "
                         "target ROIs, and 2 columns representing azimuth and elevation.")

    orientations = np.zeros((p.num_of_targets, 3))
    for n in range(p.num_of_targets):
        azimuth, elevation = t[n, 0], t[n, 1]
        orientations[n] = [np.cos(azimuth) * np.sin(elevation),
                           np.sin(azimuth) * np.sin(elevation),
                           np.cos(elevation)]

    xd = _desired_field(p, orientations)
    x_opt, _, status = optimize_currents(A, xd, p.i_max, p.w, p.target_nodes,
                                         p.opt_type, p.U, p.S, p.V, p.elec_num, False)
    if status == "Failed":
        logger.warning("Inner optimization FAILED!!\n Program will continue but results "
                       "may be INACCURATE!")

    if "wls" in p.opt_type:
        return float(np.linalg.norm(np.sqrt(p.w) * (xd - x_opt)))
    if "lcmv" in p.opt_type:
        return float(np.linalg.norm(x_opt))
    if "max-l1" in p.opt_type:
        intensity = 0.0
        for n in range(p.num_of_targets):
            nodes = p.target_nodes[n]
            mean_field = np.array([x_opt[nodes].mean(),
                                   x_opt[nodes + p.n_locs].mean(),
                                   x_opt[nodes + 2 * p.n_locs].mean()])
            intensity += float(np.dot(orientations[n], mean_field))
        return -intensity
    raise ValueError(f"Unknown optimization type: {p.opt_type}")
