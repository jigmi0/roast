"""The convex programs and the targeting objective."""

import numpy as np
import pytest

cvxpy = pytest.importorskip("cvxpy")

from roast.targeting import (TargetingProblem, lcmv_l1, ls_l1, max_l1,
                             optimize, optimize_anon, optimize_currents,
                             optimize_prepare)


@pytest.fixture
def problem():
    rng = np.random.default_rng(0)
    n_locs, n_elec = 40, 6
    A = rng.normal(size=(3 * n_locs, n_elec))
    locs = rng.normal(size=(n_locs, 3)) * 10
    p = TargetingProblem(target_coord=np.zeros((1, 3)), num_of_targets=1,
                         opt_type="max-l1", target_radius=8.0, n_locs=n_locs,
                         u=np.array([[0.0, 0.0, 1.0]]))
    return p, A, locs


def test_currents_sum_to_zero_with_the_reference(problem):
    p, A, locs = problem
    p = optimize_prepare(p, A, locs)
    currents = optimize(p, A)
    total = np.append(currents, -currents.sum())
    assert total.sum() == pytest.approx(0.0, abs=1e-6)


def test_max_l1_saturates_the_total_current_budget(problem):
    p, A, locs = problem
    p = optimize_prepare(p, A, locs)
    currents = optimize(p, A)
    l1 = np.abs(np.append(currents, -currents.sum())).sum()
    assert l1 == pytest.approx(2 * p.i_max, rel=1e-3)


def test_max_l1per_bounds_every_electrode(problem):
    p, A, locs = problem
    p.opt_type, p.elec_num = "max-l1per", 4
    p = optimize_prepare(p, A, locs)
    currents = optimize(p, A)
    assert np.max(np.abs(currents)) <= 2 * p.i_max / p.elec_num + 1e-6


def test_wls_weights_split_by_k(problem):
    # k is the ratio between the total weight on the targets and the total
    # weight everywhere else, so a small k buys focality at the cost of intensity.
    p, A, locs = problem
    p.opt_type, p.k = "wls-l1", 0.2
    p = optimize_prepare(p, A, locs)
    targets = p.target_nodes[0]
    others = np.setdiff1d(np.arange(p.n_locs), targets)
    weights = p.w[:p.n_locs]
    assert weights[targets].sum() / weights[others].sum() == pytest.approx(p.k)
    assert p.U is not None and p.S.shape[0] == p.V.shape[0]


def test_objective_is_finite_for_every_algorithm(problem):
    p, A, locs = problem
    for opt_type in ("max-l1", "wls-l1", "lcmv-l1"):
        p.opt_type = opt_type
        p.k = 0.2
        prepared = optimize_prepare(p, A, locs)
        value = optimize_anon(prepared, np.array([[0.3, 0.7]]), A)
        assert np.isfinite(value)


def test_unconstrained_wls_is_a_least_squares_fit():
    rng = np.random.default_rng(3)
    A = rng.normal(size=(30, 4))
    d = rng.normal(size=30)
    w = np.ones(30)
    x, s, status = optimize_currents(A, d, 2.0, w, [np.arange(5)],
                                     "unconstrained-wls", None, None, None)
    assert status == "Solved"
    assert np.allclose(s, np.linalg.lstsq(A, d, rcond=None)[0])


def test_convex_helpers_report_their_status():
    A = np.eye(3)
    x, status = ls_l1(A, np.array([1.0, 0, 0]), 1.0)
    assert status == "Solved" and np.abs(np.append(x, -x.sum())).sum() <= 2 + 1e-6
    x, status = max_l1(np.eye(3), 1.0)
    assert status == "Solved"
    x, status = lcmv_l1(A, np.eye(3), np.zeros(3), 1.0)
    assert status == "Solved"
