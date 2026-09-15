"""Backend for the Octave TriScatteredInterp shim.

MATLAB's TriScatteredInterp(P,V) with the default 'linear' method is a
Delaunay triangulation of the scattered points followed by barycentric
linear interpolation, returning NaN outside the convex hull.  That is
exactly scipy's Delaunay + LinearNDInterpolator.

The Delaunay triangulation is cached on disk and keyed by the point data,
because ROAST builds four interpolants over the same node set.
"""
import hashlib, os, pickle, sys
import numpy as np
import scipy.io as sio
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay

in_path, out_path, cache_dir = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(cache_dir, exist_ok=True)

d = sio.loadmat(in_path)
P = np.ascontiguousarray(d["P"], dtype=np.float64)
V = np.asarray(d["V"], dtype=np.float64).ravel()

key = hashlib.md5(P.tobytes()).hexdigest()
cache = os.path.join(cache_dir, key + ".tri.pkl")
if os.path.exists(cache):
    with open(cache, "rb") as fh:
        tri = pickle.load(fh)
else:
    tri = Delaunay(P)
    with open(cache, "wb") as fh:
        pickle.dump(tri, fh, protocol=4)

if "dims" in d:
    dims = np.asarray(d["dims"]).ravel().astype(int)
    ax = [np.arange(1, n + 1, dtype=np.float64) for n in dims]
    # ndgrid order: first axis varies fastest
    g = np.meshgrid(*ax, indexing="ij")
    Q = np.column_stack([a.ravel(order="F") for a in g])
    out_shape = tuple(dims)
    order = "F"
else:
    Q = np.ascontiguousarray(d["Q"], dtype=np.float64)
    out_shape = tuple(np.asarray(d["qshape"]).ravel().astype(int))
    order = "F"

R = LinearNDInterpolator(tri, V)(Q)
sio.savemat(out_path, {"R": np.asarray(R, dtype=np.float64).reshape(out_shape, order=order)},
            do_compression=False)
