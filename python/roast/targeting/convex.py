"""Convex programs behind the targeting algorithms.

These mirror the CVX formulations of the MATLAB implementation one to one; the
constraint ``norm([x; -sum(x)], 1) <= 2 * ub`` bounds the total injected
current, the additional infinity-norm constraint bounds the current through any
single electrode (the ``...per`` variants).  The reference electrode current is
``-sum(x)``, which is why it is appended to the vector.
"""

from __future__ import annotations

import numpy as np

__all__ = ["ls_l1", "ls_l1per", "max_l1", "max_l1per", "lcmv_l1", "lcmv_l1per",
           "CVXPY_HINT"]

CVXPY_HINT = ("The targeting optimisation needs cvxpy. Install it with "
              "`pip install cvxpy` (or `pip install roast[targeting]`).")


def _cvxpy():
    try:
        import cvxpy
    except ImportError as error:                     # pragma: no cover
        raise ImportError(CVXPY_HINT) from error
    return cvxpy


def _solve(problem, variable, verbose: bool):
    cp = _cvxpy()
    try:
        problem.solve(verbose=verbose)
    except cp.error.SolverError:
        return np.full(variable.shape, np.nan), "Failed"
    if problem.status not in ("optimal", "optimal_inaccurate") or variable.value is None:
        return np.full(variable.shape, np.nan), "Failed"
    return np.asarray(variable.value).ravel(), "Solved"


def _with_reference(cp, x):
    """``[x; -sum(x)]`` - the currents including the reference electrode."""
    return cp.hstack([x, -cp.sum(x)])


def ls_l1(A, d, ub, verbose: bool = False):
    """Least squares with a bound on the total current."""
    cp = _cvxpy()
    x = cp.Variable(np.shape(A)[1])
    constraints = [cp.norm(_with_reference(cp, x), 1) <= 2 * ub]
    problem = cp.Problem(cp.Minimize(cp.norm(A @ x - np.asarray(d).ravel(), 2)),
                         constraints)
    return _solve(problem, x, verbose)


def ls_l1per(A, d, ub, verbose: bool = False):
    """Least squares, also bounding the current through each electrode."""
    cp = _cvxpy()
    x = cp.Variable(np.shape(A)[1])
    constraints = [cp.norm(_with_reference(cp, x), 1) <= 2 * ub,
                   cp.norm(_with_reference(cp, x), "inf") <= ub / 2]
    problem = cp.Problem(cp.Minimize(cp.norm(A @ x - np.asarray(d).ravel(), 2)),
                         constraints)
    return _solve(problem, x, verbose)


def max_l1(f, ub, verbose: bool = False):
    """Maximum intensity along the desired orientation."""
    cp = _cvxpy()
    f = np.atleast_2d(np.asarray(f, dtype=float))
    x = cp.Variable(f.shape[0])
    constraints = [cp.norm(_with_reference(cp, x), 1) <= 2 * ub]
    problem = cp.Problem(cp.Maximize(cp.sum(f.T @ x)), constraints)
    return _solve(problem, x, verbose)


def max_l1per(f, ub, sol_elec_num, verbose: bool = False):
    """Maximum intensity with a fixed number of active electrodes.

    ``sol_elec_num`` is the desired number of anodes plus cathodes: bounding
    each electrode by ``2*ub/sol_elec_num`` drives the solution towards exactly
    that many active electrodes.
    """
    cp = _cvxpy()
    f = np.atleast_2d(np.asarray(f, dtype=float))
    x = cp.Variable(f.shape[0])
    constraints = [cp.norm(_with_reference(cp, x), 1) <= 2 * ub,
                   cp.norm(_with_reference(cp, x), "inf") <= 2 * ub / sol_elec_num]
    problem = cp.Problem(cp.Maximize(cp.sum(f.T @ x)), constraints)
    return _solve(problem, x, verbose)


def lcmv_l1(A, C, f, ub, verbose: bool = False):
    """Linearly constrained minimum variance: minimise the field elsewhere while
    holding the field at the target fixed."""
    cp = _cvxpy()
    x = cp.Variable(np.shape(A)[1])
    constraints = [cp.norm(_with_reference(cp, x), 1) <= 2 * ub,
                   np.asarray(C) @ x == np.asarray(f).ravel()]
    problem = cp.Problem(cp.Minimize(cp.norm(A @ x, 2)), constraints)
    return _solve(problem, x, verbose)


def lcmv_l1per(A, C, f, ub, verbose: bool = False):
    """LCMV, also bounding the current through each electrode."""
    cp = _cvxpy()
    x = cp.Variable(np.shape(A)[1])
    constraints = [cp.norm(_with_reference(cp, x), 1) <= 2 * ub,
                   cp.norm(_with_reference(cp, x), "inf") <= ub / 2,
                   np.asarray(C) @ x == np.asarray(f).ravel()]
    problem = cp.Problem(cp.Minimize(cp.norm(A @ x, 2)), constraints)
    return _solve(problem, x, verbose)
