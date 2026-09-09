"""Small re-implementations of the MATLAB / Image Processing Toolbox routines
that ROAST relies on.

Two conventions are kept from the original MATLAB code because the algorithms
depend on them:

* **Column-major (Fortran) ordering.**  Wherever the original used
  ``find``/``ind2sub``/``sub2ind`` the ordering of the resulting point clouds
  matters (it decides ties when a "first"/"last" element is picked), so the
  helpers here operate in Fortran order just like MATLAB.
* **1-based voxel coordinates.**  Point clouds, landmarks and mesh nodes are
  kept 1-based, which is also what the SPM voxel-to-world matrices assume.
  Convert with :func:`to_index` right before indexing a NumPy array.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

__all__ = [
    "find", "ind2sub", "sub2ind", "to_index", "at",
    "fspecial_gaussian", "imfilter", "imfilter_uint8",
    "imerode", "imdilate", "imopen", "imclose", "imfill_holes",
    "bwconncomp_sizes", "size_of_object", "bwareaopen", "connectivity_structure",
    "edge_sobel", "prctile", "sortrows", "unique_rows", "intersect_rows",
    "ismember_rows", "cart2sph", "sph2cart", "interp1", "nanmean",
    "timestamp_tag",
]


# ---------------------------------------------------------------------------
# indexing helpers
# ---------------------------------------------------------------------------
def find(mask: np.ndarray) -> np.ndarray:
    """MATLAB ``find``: linear (column-major) indices of the non-zero entries."""
    return np.flatnonzero(np.asarray(mask).ravel(order="F"))


def ind2sub(shape, indices) -> np.ndarray:
    """MATLAB ``ind2sub`` returning an ``(n, ndim)`` array of 1-based subscripts."""
    subs = np.unravel_index(np.asarray(indices, dtype=np.int64), shape, order="F")
    return np.stack(subs, axis=-1) + 1


def sub2ind(shape, subs) -> np.ndarray:
    """MATLAB ``sub2ind`` for an ``(n, ndim)`` array of 1-based subscripts."""
    subs = np.asarray(subs, dtype=np.int64) - 1
    return np.ravel_multi_index(tuple(subs.T), shape, order="F")


def to_index(coords) -> tuple:
    """Turn 1-based coordinates into a tuple of 0-based NumPy index arrays."""
    coords = np.asarray(coords, dtype=np.int64) - 1
    if coords.ndim == 1:
        return tuple(int(c) for c in coords)
    return tuple(coords[:, i] for i in range(coords.shape[1]))


def at(volume: np.ndarray, coords):
    """Sample ``volume`` at 1-based ``coords`` (a point or an ``(n, 3)`` array)."""
    return volume[to_index(coords)]


# ---------------------------------------------------------------------------
# filtering
# ---------------------------------------------------------------------------
def fspecial_gaussian(hsize: int = 5, sigma: float = 0.5) -> np.ndarray:
    """MATLAB ``fspecial('gaussian', hsize, sigma)``."""
    n = (hsize - 1) / 2.0
    y, x = np.mgrid[-n:n + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    h[h < np.finfo(float).eps * h.max()] = 0.0
    total = h.sum()
    if total != 0:
        h /= total
    return h


def imfilter(img: np.ndarray, kernel: np.ndarray, mode: str = "constant") -> np.ndarray:
    """MATLAB ``imfilter``: correlation (not convolution) with zero padding."""
    return ndimage.correlate(np.asarray(img, dtype=float), kernel,
                             mode=mode, cval=0.0)


def imfilter_uint8(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """``imfilter`` on integer data: filter in double, then round and saturate.

    MATLAB keeps the class of the input, so the result of filtering a ``uint8``
    image is rounded and clipped back into ``[0, 255]``.
    """
    out = imfilter(img, kernel)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def filter_slices(volume: np.ndarray, kernel: np.ndarray,
                  keep_uint8: bool = True) -> np.ndarray:
    """Apply a 2-D kernel slice by slice along the third dimension."""
    out = np.empty_like(volume)
    for i in range(volume.shape[2]):
        if keep_uint8:
            out[:, :, i] = imfilter_uint8(volume[:, :, i], kernel)
        else:
            out[:, :, i] = imfilter(volume[:, :, i], kernel)
    return out


# ---------------------------------------------------------------------------
# morphology
# ---------------------------------------------------------------------------
def _binary(img):
    return np.asarray(img).astype(bool)


def imerode(img, selem):
    return ndimage.binary_erosion(_binary(img), structure=_binary(selem),
                                  border_value=1)


def imdilate(img, selem):
    return ndimage.binary_dilation(_binary(img), structure=_binary(selem),
                                   border_value=0)


def imopen(img, selem):
    return imdilate(imerode(img, selem), selem)


def imclose(img, selem):
    return imerode(imdilate(img, selem), selem)


def imfill_holes(img):
    """MATLAB ``imfill(img, 'holes')`` for binary images."""
    return ndimage.binary_fill_holes(_binary(img))


def connectivity_structure(conn: int, ndim: int) -> np.ndarray:
    """Structuring element for a MATLAB connectivity value (4/8 or 6/18/26)."""
    if ndim == 2:
        rank = {4: 1, 8: 2}[conn]
    elif ndim == 3:
        rank = {6: 1, 18: 2, 26: 3}[conn]
    else:
        raise ValueError("only 2-D and 3-D connectivities are supported")
    return ndimage.generate_binary_structure(ndim, rank)


def bwconncomp_sizes(img, conn: int | None = None):
    """Labelled image plus the size of every connected component."""
    img = _binary(img)
    if conn is None:
        conn = 8 if img.ndim == 2 else 26
    labels, num = ndimage.label(img, structure=connectivity_structure(conn, img.ndim))
    if num == 0:
        return labels, np.zeros(0, dtype=np.int64)
    sizes = np.bincount(labels.ravel())[1:]
    return labels, sizes


def size_of_object(img, conn: int | None = None):
    """Port of ``sizeOfObject``: component sizes sorted in descending order."""
    labels, sizes = bwconncomp_sizes(img, conn)
    order = np.argsort(-sizes, kind="stable")
    return sizes[order], order + 1


def bwareaopen(img, thresh: int, conn: int | None = None) -> np.ndarray:
    """MATLAB ``bwareaopen``: drop components with fewer than ``thresh`` voxels."""
    labels, sizes = bwconncomp_sizes(img, conn)
    if sizes.size == 0:
        return _binary(img)
    keep = np.concatenate([[False], sizes >= thresh])
    return keep[labels]


# ---------------------------------------------------------------------------
# edges
# ---------------------------------------------------------------------------
def edge_sobel(img: np.ndarray) -> np.ndarray:
    """Re-implementation of MATLAB ``edge(img, 'sobel')`` with an automatic
    threshold and directional thinning."""
    a = np.asarray(img, dtype=float)
    op = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]) / 8.0
    bx = ndimage.correlate(a, op.T, mode="nearest")
    by = ndimage.correlate(a, op, mode="nearest")
    b = bx * bx + by * by
    cutoff = 4.0 * b.mean()
    strong = b > cutoff

    def shifted(arr, axis, offset):
        out = np.full_like(arr, -np.inf)
        if offset > 0:
            slicer_dst = [slice(None)] * arr.ndim
            slicer_src = [slice(None)] * arr.ndim
            slicer_dst[axis] = slice(offset, None)
            slicer_src[axis] = slice(None, -offset)
        else:
            slicer_dst = [slice(None)] * arr.ndim
            slicer_src = [slice(None)] * arr.ndim
            slicer_dst[axis] = slice(None, offset)
            slicer_src[axis] = slice(-offset, None)
        out[tuple(slicer_dst)] = arr[tuple(slicer_src)]
        return out

    bxs, bys = bx * bx, by * by
    along_cols = (bxs >= bys) & (b >= shifted(b, 1, 1)) & (b > shifted(b, 1, -1))
    along_rows = (bys >= bxs) & (b >= shifted(b, 0, 1)) & (b > shifted(b, 0, -1))
    return strong & (along_cols | along_rows)


# ---------------------------------------------------------------------------
# statistics / sorting
# ---------------------------------------------------------------------------
def prctile(x, p):
    """MATLAB ``prctile`` (linear interpolation on the ``(i-0.5)/n`` grid)."""
    x = np.asarray(x, dtype=float).ravel()
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan
    xs = np.sort(x)
    n = xs.size
    q = 100.0 * (np.arange(1, n + 1) - 0.5) / n
    return np.interp(np.asarray(p, dtype=float), q, xs)


def sortrows(a: np.ndarray, columns=None) -> np.ndarray:
    """MATLAB ``sortrows``: returns the permutation that sorts the rows."""
    a = np.asarray(a)
    if columns is None:
        columns = range(a.shape[1])
    keys = [a[:, c] for c in columns]
    return np.lexsort(tuple(reversed(keys)))


def unique_rows(a: np.ndarray) -> np.ndarray:
    """MATLAB ``unique(a, 'rows')`` (sorted, duplicates removed)."""
    a = np.asarray(a)
    if a.size == 0:
        return a.reshape(0, a.shape[1] if a.ndim == 2 else 0)
    return np.unique(a, axis=0)


def _rows_view(a: np.ndarray):
    a = np.ascontiguousarray(a)
    return a.view([("", a.dtype)] * a.shape[1]).ravel()


def intersect_rows(a: np.ndarray, b: np.ndarray):
    """MATLAB ``[c, ia, ib] = intersect(a, b, 'rows')`` (0-based ia/ib)."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.size == 0 or b.size == 0:
        empty = np.zeros((0, a.shape[1]), dtype=a.dtype)
        return empty, np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    if a.dtype != b.dtype:
        b = b.astype(a.dtype)
    common, ia, ib = np.intersect1d(_rows_view(a), _rows_view(b),
                                    return_indices=True)
    return a[ia], ia, ib


def ismember_rows(a: np.ndarray, b: np.ndarray):
    """MATLAB ``[tf, loc] = ismember(a, b, 'rows')``; ``loc`` is 0-based, -1 if absent."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.size == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=int)
    if b.size == 0:
        return np.zeros(len(a), dtype=bool), np.full(len(a), -1, dtype=int)
    if a.dtype != b.dtype:
        b = b.astype(a.dtype)
    av, bv = _rows_view(a), _rows_view(b)
    order = np.argsort(bv, kind="stable")
    bs = bv[order]
    pos = np.searchsorted(bs, av)
    pos_clipped = np.clip(pos, 0, len(bs) - 1)
    tf = bs[pos_clipped] == av
    loc = np.where(tf, order[pos_clipped], -1)
    return tf, loc


# ---------------------------------------------------------------------------
# coordinate conversions & interpolation
# ---------------------------------------------------------------------------
def cart2sph(x, y, z):
    """MATLAB ``cart2sph``: azimuth, elevation (from the xy-plane), radius."""
    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
    hypotxy = np.hypot(x, y)
    r = np.hypot(hypotxy, z)
    elev = np.arctan2(z, hypotxy)
    az = np.arctan2(y, x)
    return az, elev, r


def sph2cart(az, elev, r):
    """MATLAB ``sph2cart``."""
    az, elev, r = np.asarray(az), np.asarray(elev), np.asarray(r)
    rcoselev = r * np.cos(elev)
    return rcoselev * np.cos(az), rcoselev * np.sin(az), r * np.sin(elev)


def interp1(x, y, xq):
    """MATLAB ``interp1`` (linear, NaN outside the sample range)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xq = np.asarray(xq, dtype=float)
    order = np.argsort(x)
    out = np.interp(xq, x[order], y[order], left=np.nan, right=np.nan)
    return out


def nanmean(x):
    x = np.asarray(x, dtype=float)
    if np.all(np.isnan(x)):
        return np.nan
    return float(np.nanmean(x))


def timestamp_tag() -> str:
    """MATLAB ``datestr(now, 30)``, used to build unique simulation tags."""
    from datetime import datetime
    return datetime.now().strftime("%Y%m%dT%H%M%S")
