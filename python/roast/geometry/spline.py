"""Approximation of 2-D data by a natural cubic spline.

Port of ``ncs2dapprox`` by Murtaza Khan (bundled under ``lib/ncs2daprox``),
used to measure the nasion-to-inion distance along the scalp surface.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

__all__ = ["ncs2dapprox", "spline_curve"]


def _merge_break(bx, by, breaks, x_new, y_new, index_new):
    """Insert a break point, keeping the list sorted and free of duplicates."""
    if index_new in breaks:
        return bx, by, breaks
    position = int(np.searchsorted(breaks, index_new))
    return (np.insert(bx, position, x_new),
            np.insert(by, position, y_new),
            np.insert(breaks, position, index_new))


def spline_curve(breaks, bx, by, query):
    """Evaluate the not-a-knot cubic spline through ``(bx, by)`` at ``query``.

    This is MATLAB's ``ppval(spline(t, [bx, by]'), query)``.
    """
    breaks = np.asarray(breaks, dtype=float)
    values = np.column_stack([np.asarray(bx, dtype=float),
                              np.asarray(by, dtype=float)])
    if breaks.size < 2:
        raise ValueError("at least two break points are required")
    if breaks.size == 2:                      # CubicSpline needs 3+ for not-a-knot
        spline = CubicSpline(breaks, values, axis=0, bc_type="natural")
    else:
        spline = CubicSpline(breaks, values, axis=0, bc_type="not-a-knot")
    out = spline(np.asarray(query, dtype=float))
    return out[:, 0], out[:, 1]


def ncs2dapprox(x, y, max_allowed_sq_dist: float = 1.0, initial_breaks=None):
    """Fit a parametric cubic spline through a subset of the samples.

    Break points are added greedily at the worst-fitting sample until every
    sample is within ``max_allowed_sq_dist`` of the curve.

    Returns ``(bx, by, breaks)`` where ``breaks`` are 1-based indices into the
    input samples, matching the MATLAB original.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError("first two arguments must have equal number of values")
    n = x.size
    if n < 2:
        raise ValueError("At least two values are required in data")

    breaks = (np.asarray(initial_breaks, dtype=np.int64).ravel()
              if initial_breaks is not None else np.array([1, n], dtype=np.int64))
    breaks = np.unique(breaks)
    bx, by = x[breaks - 1], y[breaks - 1]
    bx, by, breaks = _merge_break(bx, by, breaks, x[0], y[0], 1)
    bx, by, breaks = _merge_break(bx, by, breaks, x[-1], y[-1], n)

    query = np.linspace(1, n, n)
    while True:
        xi, yi = spline_curve(breaks, bx, by, query)
        sq_dist = (x - xi) ** 2 + (y - yi) ** 2
        worst = int(np.argmax(sq_dist))
        if sq_dist[worst] <= max_allowed_sq_dist:
            break
        bx, by, breaks = _merge_break(bx, by, breaks, x[worst], y[worst], worst + 1)
    return bx, by, breaks
