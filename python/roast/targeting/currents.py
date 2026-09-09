"""Core of the tDCS targeting technique.

Refer to Dmochowski et al., "Optimized multi-electrode stimulation increases
focality and intensity at target", J. Neural Eng. 8 (2011) 046011.
"""

from __future__ import annotations

import numpy as np

from ..utils.logging import get_logger
from .convex import lcmv_l1, lcmv_l1per, ls_l1, ls_l1per, max_l1, max_l1per

__all__ = ["optimize_currents"]

logger = get_logger()

# The LCMV variants relax the target intensity until the program becomes
# feasible; this caps that search so it cannot spin forever.
_MAX_RELAXATIONS = 200


def _target_constraints(A, d, tar_nodes, n_locs, n_elec):
    """Mean lead field and mean desired field over each target ROI."""
    n_roi = len(tar_nodes)
    C = np.zeros((3 * n_roi, n_elec))
    f = np.zeros(3 * n_roi)
    for n, nodes in enumerate(tar_nodes):
        C[3 * n:3 * n + 3, :] = np.vstack([A[nodes, :].mean(axis=0),
                                           A[nodes + n_locs, :].mean(axis=0),
                                           A[nodes + 2 * n_locs, :].mean(axis=0)])
        f[3 * n:3 * n + 3] = [d[nodes].mean(), d[nodes + n_locs].mean(),
                              d[nodes + 2 * n_locs].mean()]
    return C, f


def optimize_currents(A, d, s_max, w, tar_nodes, method, U, S, V,
                      sol_elec_num=None, verbose: bool = False):
    """Solve for the optimal electrode currents.

    Parameters mirror the MATLAB signature:

    ``A``     transfer matrix, ``3N x M`` (N mesh nodes, M electrodes without
              the reference)
    ``d``     desired electric field, a ``3N`` vector
    ``s_max`` maximum total current
    ``w``     weights of the weighted least squares, also flagging the targets
    ``tar_nodes``  node indices of each target ROI (0-based)
    ``U, S, V``    SVD of the (weighted) transfer matrix, precomputed once

    Returns ``(x_opt, s_opt, status)``: the achieved field, the electrode
    currents and whether the program was solved.
    """
    A = np.asarray(A, dtype=float)
    d = np.asarray(d, dtype=float).ravel()
    n_elec = A.shape[1]
    n_locs = A.shape[0] // 3
    tar_nodes = [np.asarray(nodes, dtype=int).ravel() for nodes in tar_nodes]

    if method == "unconstrained-wls":
        sqrt_w = np.sqrt(w)
        s_opt = np.linalg.lstsq(sqrt_w[:, None] * A, sqrt_w * d, rcond=None)[0]
        return A @ s_opt, s_opt, "Solved"

    if method in ("wls-l1", "wls-l1per"):
        sqrt_w = np.sqrt(w)
        solver = ls_l1 if method == "wls-l1" else ls_l1per
        s_opt, status = solver(S @ V.T, U.T @ (sqrt_w * d), s_max, verbose)
        return A @ s_opt, s_opt, status

    if method == "unconstrained-lcmv":
        C, f = _target_constraints(A, d, tar_nodes, n_locs, n_elec)
        ata_inv = np.linalg.inv(A.T @ A)
        s_opt = ata_inv @ C.T @ np.linalg.inv(C @ ata_inv @ C.T) @ f
        return A @ s_opt, s_opt, "Solved"

    if method in ("lcmv-l1", "lcmv-l1per"):
        C, f = _target_constraints(A, d, tar_nodes, n_locs, n_elec)
        solver = lcmv_l1 if method == "lcmv-l1" else lcmv_l1per
        status = ""
        s_opt = np.full(n_elec, np.nan)
        for _ in range(_MAX_RELAXATIONS):
            s_opt, status = solver(S @ V.T, C, f, s_max, verbose)
            if status == "Solved":
                break
            f = f / 1.25          # infeasible: ask for a weaker field at the target
        if np.all(np.abs(f) < 1e-5):
            # Nothing can be achieved at this target; flag it rather than
            # returning a meaningless montage.
            s_opt = np.full(n_elec, np.nan)
        return A @ s_opt, s_opt, status

    if method in ("max-l1", "max-l1per"):
        n_roi = len(tar_nodes)
        Cf = np.zeros((n_elec, n_roi))
        for n, nodes in enumerate(tar_nodes):
            C = np.vstack([A[nodes, :].mean(axis=0),
                           A[nodes + n_locs, :].mean(axis=0),
                           A[nodes + 2 * n_locs, :].mean(axis=0)])
            f = np.array([d[nodes].mean(), d[nodes + n_locs].mean(),
                          d[nodes + 2 * n_locs].mean()])
            Cf[:, n] = C.T @ f
        if method == "max-l1":
            s_opt, status = max_l1(Cf, s_max, verbose)
        else:
            s_opt, status = max_l1per(Cf, s_max, sol_elec_num, verbose)
        return A @ s_opt, s_opt, status

    raise ValueError(f"Unknown optimization type: {method}")
