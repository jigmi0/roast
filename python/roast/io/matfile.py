"""Reading and writing the ``.mat`` files that ROAST exchanges with SPM.

``scipy.io`` covers everything up to MATLAB v7; the lead field and the
simulation results can be far larger than the 2 GB limit of that format, so
those are written through HDF5 (the container MATLAB itself uses for ``-v7.3``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import io as sio

from .nifti import ImageGeom

__all__ = ["load_mat", "save_mat", "load_spm_mapping", "save_spm_mapping"]

_V7_LIMIT = 1.5 * 1024 ** 3


def _struct_get(struct, field):
    """Read one field from whatever scipy hands back for a MATLAB struct."""
    if isinstance(struct, np.ndarray) and struct.dtype.names:
        value = struct[field]
        while isinstance(value, np.ndarray) and value.dtype == object and value.size == 1:
            value = value.item()
        return value
    return getattr(struct, field)


def _first(struct):
    """SPM stores struct *arrays* (``image(1)``); take the first element."""
    if isinstance(struct, np.ndarray) and struct.dtype.names and struct.size > 1:
        return struct.ravel()[0]
    if isinstance(struct, np.ndarray) and struct.dtype == object and struct.size >= 1:
        return struct.ravel()[0]
    if isinstance(struct, (list, tuple)):
        return struct[0]
    return struct


def load_mat(path) -> dict:
    """Load a ``.mat`` file written by either MATLAB or :func:`save_mat`."""
    path = Path(path)
    try:
        return sio.loadmat(str(path), struct_as_record=False, squeeze_me=True)
    except NotImplementedError:
        import h5py

        out = {}
        with h5py.File(str(path), "r") as handle:
            for key, value in handle.items():
                if key.startswith("#"):
                    continue
                data = np.asarray(value)
                # HDF5-backed .mat files store arrays transposed.
                out[key] = data.T if data.ndim > 1 else data
        return out


def save_mat(path, variables: dict) -> Path:
    """Save ``variables``, transparently switching to HDF5 for large data."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    total = sum(np.asarray(v).nbytes for v in variables.values()
                if isinstance(v, (np.ndarray, np.generic)))
    if total < _V7_LIMIT:
        sio.savemat(str(path), variables, do_compression=True)
        return path

    import h5py

    with h5py.File(str(path), "w") as handle:
        handle.attrs["MATLAB_class"] = np.bytes_(b"struct")
        for key, value in variables.items():
            data = np.asarray(value)
            handle.create_dataset(key, data=data.T if data.ndim > 1 else data,
                                  compression="gzip", compression_opts=4)
    return path


def load_spm_mapping(path):
    """Read an SPM ``_seg8.mat`` (or our ``_niftyReg.mat``) mapping file.

    Returns ``(image_geom, tpm_geom, affine)`` where the two geometries follow
    the :class:`~roast.io.nifti.ImageGeom` convention and ``affine`` is the
    MRI-to-MNI affine estimated during segmentation/registration.
    """
    data = load_mat(path)
    image = _first(data["image"])
    tpm = _first(data["tpm"])
    affine = np.asarray(data["Affine"], dtype=float).reshape(4, 4)
    image_geom = ImageGeom(np.asarray(_struct_get(image, "mat"), dtype=float),
                           np.asarray(_struct_get(image, "dim"), dtype=int).ravel())
    tpm_geom = ImageGeom(np.asarray(_struct_get(tpm, "mat"), dtype=float),
                         np.asarray(_struct_get(tpm, "dim"), dtype=int).ravel())
    return image_geom, tpm_geom, affine


def save_spm_mapping(path, image_geom: ImageGeom, tpm_geom: ImageGeom,
                     affine: np.ndarray) -> Path:
    """Write a mapping file in the same layout SPM12 uses for ``_seg8.mat``."""
    image = np.zeros((1,), dtype=[("mat", "O"), ("dim", "O")])
    image["mat"][0] = np.asarray(image_geom.mat, dtype=float)
    image["dim"][0] = np.asarray(image_geom.dim, dtype=float).reshape(1, 3)
    tpm = np.zeros((1,), dtype=[("mat", "O"), ("dim", "O")])
    tpm["mat"][0] = np.asarray(tpm_geom.mat, dtype=float)
    tpm["dim"][0] = np.asarray(tpm_geom.dim, dtype=float).reshape(1, 3)
    return save_mat(path, {"image": image, "tpm": tpm,
                           "Affine": np.asarray(affine, dtype=float)})
