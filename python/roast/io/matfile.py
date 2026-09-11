"""Reading and writing the ``.mat`` files that ROAST exchanges with SPM.

``scipy.io`` covers everything up to MATLAB v7; the lead field and the
simulation results can be far larger than the 2 GB limit of that format, so
those are written through HDF5 in the layout MATLAB itself uses for ``-v7.3``
(the 512-byte MAT-file header, one dataset per variable with reversed axes and
a ``MATLAB_class`` attribute), so MATLAB can ``load`` them as well.
"""

from __future__ import annotations

import platform
import time
from pathlib import Path

import numpy as np
from scipy import io as sio

from .nifti import ImageGeom

__all__ = ["load_mat", "save_mat", "load_spm_mapping", "save_spm_mapping"]

_V7_LIMIT = 1.5 * 1024 ** 3
_USERBLOCK = 512                       # size of the MAT-file header block

_MATLAB_CLASS = {
    np.dtype(np.float64): b"double", np.dtype(np.float32): b"single",
    np.dtype(np.int8): b"int8", np.dtype(np.uint8): b"uint8",
    np.dtype(np.int16): b"int16", np.dtype(np.uint16): b"uint16",
    np.dtype(np.int32): b"int32", np.dtype(np.uint32): b"uint32",
    np.dtype(np.int64): b"int64", np.dtype(np.uint64): b"uint64",
}
_NUMPY_DTYPE = {name: dtype for dtype, name in _MATLAB_CLASS.items()}


def _struct_get(struct, field):
    """Read one field from whatever scipy hands back for a MATLAB struct."""
    if isinstance(struct, np.ndarray) and struct.dtype.names:
        value = struct[field]
        while isinstance(value, np.ndarray) and value.dtype == object and value.size == 1:
            value = value.item()
        return value
    if isinstance(struct, dict):
        return struct[field]
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


# ---------------------------------------------------------------------------
# HDF5 (MATLAB v7.3) layout
# ---------------------------------------------------------------------------
def _squeeze(data: np.ndarray):
    """Drop singleton axes the way ``scipy.io.loadmat(squeeze_me=True)`` does."""
    if data.ndim > 1 and 1 in data.shape:
        data = np.squeeze(data)
    if data.ndim == 0:
        return data[()]
    return data


def _read_hdf5_variable(node):
    import h5py

    if isinstance(node, h5py.Group):                 # a MATLAB struct
        return {name: _read_hdf5_variable(child) for name, child in node.items()}

    matlab_class = node.attrs.get("MATLAB_class", b"")
    if isinstance(matlab_class, str):
        matlab_class = matlab_class.encode()
    data = np.asarray(node)

    if node.attrs.get("MATLAB_empty", 0):
        if matlab_class == b"char":
            return ""
        shape = tuple(int(v) for v in data.ravel())
        return np.zeros(shape, dtype=_NUMPY_DTYPE.get(matlab_class, np.float64))

    # HDF5 is row-major and MATLAB column-major: the axes are reversed on disk.
    data = data.T if data.ndim > 1 else data
    if matlab_class == b"char":
        return data.astype("<u2").tobytes().decode("utf-16-le")
    if matlab_class == b"logical":
        data = data.astype(bool)
    return _squeeze(data)


def _write_hdf5_variable(handle, key: str, value) -> None:
    """Store one variable the way MATLAB's ``-v7.3`` does."""
    if isinstance(value, (str, bytes)):
        text = value.decode() if isinstance(value, bytes) else value
        codes = np.frombuffer(text.encode("utf-16-le"), dtype=np.uint16)
        if codes.size == 0:
            dataset = handle.create_dataset(key, data=np.array([0, 0], dtype=np.uint64))
            dataset.attrs["MATLAB_empty"] = np.uint8(1)
        else:
            dataset = handle.create_dataset(key, data=codes.reshape(1, -1).T)
        dataset.attrs["MATLAB_class"] = np.bytes_(b"char")
        dataset.attrs["MATLAB_int_decode"] = np.int32(2)
        return

    if isinstance(value, dict):                      # a MATLAB struct
        group = handle.create_group(key)
        group.attrs["MATLAB_class"] = np.bytes_(b"struct")
        for name, member in value.items():
            _write_hdf5_variable(group, str(name), member)
        return

    data = np.asarray(value)
    if data.dtype == bool:
        data, matlab_class = data.astype(np.uint8), b"logical"
    elif data.dtype in _MATLAB_CLASS:
        matlab_class = _MATLAB_CLASS[data.dtype]
    else:
        data, matlab_class = data.astype(np.float64), b"double"
    if data.ndim == 0:
        data = data.reshape(1, 1)
    elif data.ndim == 1:
        data = data.reshape(1, -1)                   # MATLAB has no 1-D arrays

    if data.size == 0:
        dataset = handle.create_dataset(key, data=np.asarray(data.shape, dtype=np.uint64))
        dataset.attrs["MATLAB_empty"] = np.uint8(1)
    else:
        dataset = handle.create_dataset(key, data=data.T, compression="gzip",
                                        compression_opts=4)
    dataset.attrs["MATLAB_class"] = np.bytes_(matlab_class)
    if matlab_class == b"logical":
        dataset.attrs["MATLAB_int_decode"] = np.int32(1)


def _write_matlab_header(path: Path) -> None:
    """Stamp the 128-byte header MATLAB looks for at the start of a v7.3 file."""
    text = ("MATLAB 7.3 MAT-file, Platform: Python %s, Created on: %s HDF5 schema 1.00 ."
            % (platform.python_version(), time.strftime("%a %b %d %H:%M:%S %Y")))
    header = text.encode("ascii")[:116].ljust(116)   # padded with spaces
    header += b"\x00" * 8                            # subsystem data offset
    header += b"\x00\x02" + b"IM"                    # version 0x0200, little endian
    with open(path, "r+b") as handle:
        handle.write(header.ljust(_USERBLOCK, b"\x00"))


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def load_mat(path) -> dict:
    """Load a ``.mat`` file written by either MATLAB or :func:`save_mat`.

    MATLAB structs come back as dicts and char arrays as strings; singleton
    axes are squeezed as with ``scipy.io.loadmat(squeeze_me=True)``.
    """
    import h5py

    path = Path(path)
    if not h5py.is_hdf5(str(path)):
        return sio.loadmat(str(path), struct_as_record=False, squeeze_me=True)

    out = {}
    with h5py.File(str(path), "r") as handle:
        for key, value in handle.items():
            if key.startswith("#"):                  # MATLAB's reference table
                continue
            out[key] = _read_hdf5_variable(value)
    return out


def save_mat(path, variables: dict) -> Path:
    """Save ``variables``, transparently switching to HDF5 (``-v7.3``) for
    large data.  Strings and dicts become MATLAB char arrays and structs."""
    import h5py

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    total = sum(np.asarray(v).nbytes for v in variables.values()
                if isinstance(v, (np.ndarray, np.generic)))
    if total < _V7_LIMIT:
        sio.savemat(str(path), variables, do_compression=True)
        return path

    with h5py.File(str(path), "w", userblock_size=_USERBLOCK) as handle:
        for key, value in variables.items():
            _write_hdf5_variable(handle, str(key), value)
    _write_matlab_header(path)
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
