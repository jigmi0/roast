"""NIfTI input/output.

The MATLAB implementation uses Jimmy Shen's ``load_untouch_nii`` /
``save_untouch_nii`` pair, which deliberately keeps the raw voxel data and the
header exactly as stored on disk.  :class:`NiftiVolume` gives the same
behaviour on top of nibabel: ``volume.img`` is the *unscaled* array and the
header fields (``srow_*``, ``qoffset_*``, ``sform_code``, ...) can be edited
directly before saving.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

__all__ = ["NiftiVolume", "load_untouch", "ImageGeom", "geom_from_nifti",
           "srow_to_spm_mat", "spm_mat_to_srow"]


def srow_to_spm_mat(srow: np.ndarray) -> np.ndarray:
    """Convert a NIfTI ``srow`` (0-based voxel to world) into an SPM ``mat``
    (1-based voxel to world)."""
    shift = np.eye(4)
    shift[:3, 3] = -1.0
    return np.asarray(srow, dtype=float) @ shift


def spm_mat_to_srow(mat: np.ndarray) -> np.ndarray:
    """Inverse of :func:`srow_to_spm_mat`."""
    shift = np.eye(4)
    shift[:3, 3] = 1.0
    return np.asarray(mat, dtype=float) @ shift


@dataclass
class ImageGeom:
    """The geometry of a volume, in the SPM ``spm_vol`` convention.

    ``mat`` maps **1-based** voxel coordinates to world coordinates and ``dim``
    holds the volume size, exactly like ``image(1).mat`` / ``image(1).dim`` in
    the MATLAB code.
    """

    mat: np.ndarray
    dim: np.ndarray

    def __post_init__(self):
        self.mat = np.asarray(self.mat, dtype=float).reshape(4, 4)
        self.dim = np.asarray(self.dim, dtype=int).reshape(3)

    @property
    def voxel_size(self) -> np.ndarray:
        """Diagonal of ``mat``, i.e. the resolution along each axis."""
        return np.array([self.mat[0, 0], self.mat[1, 1], self.mat[2, 2]], dtype=float)

    @property
    def mean_resolution(self) -> float:
        """Mean voxel size; used wherever ROAST handles anisotropic MRIs."""
        return float(np.mean(self.voxel_size))


class NiftiVolume:
    """A NIfTI volume with untouched voxel data and an editable header."""

    def __init__(self, img: np.ndarray, header: nib.Nifti1Header,
                 filename: str | Path | None = None):
        self.img = img
        self.header = header
        self.filename = str(filename) if filename is not None else None

    # -- construction ------------------------------------------------------
    @classmethod
    def load(cls, path) -> "NiftiVolume":
        path = Path(path)
        image = nib.load(str(path))
        header = image.header.copy()
        try:
            data = image.dataobj.get_unscaled()
        except AttributeError:          # pragma: no cover - non-scaled formats
            data = np.asanyarray(image.dataobj)
        return cls(np.asarray(data), header, path)

    def copy(self) -> "NiftiVolume":
        return NiftiVolume(np.array(self.img, copy=True),
                           copy.deepcopy(self.header), self.filename)

    # -- header access -----------------------------------------------------
    @property
    def shape(self):
        return self.img.shape

    @property
    def pixdim(self) -> np.ndarray:
        return np.asarray(self.header["pixdim"], dtype=float)

    @property
    def resolution(self) -> np.ndarray:
        """Voxel size along the three axes (``pixdim(2:4)`` in MATLAB)."""
        return self.pixdim[1:4]

    @property
    def sform_code(self) -> int:
        return int(self.header["sform_code"])

    @property
    def qform_code(self) -> int:
        return int(self.header["qform_code"])

    def srow(self) -> np.ndarray:
        """The 4x4 ``[srow_x; srow_y; srow_z; 0 0 0 1]`` matrix."""
        return np.vstack([
            np.asarray(self.header["srow_x"], dtype=float),
            np.asarray(self.header["srow_y"], dtype=float),
            np.asarray(self.header["srow_z"], dtype=float),
            [0.0, 0.0, 0.0, 1.0],
        ])

    def set_srow(self, matrix: np.ndarray, force_sform: bool = False) -> None:
        matrix = np.asarray(matrix, dtype=float)
        self.header["srow_x"] = matrix[0, :]
        self.header["srow_y"] = matrix[1, :]
        self.header["srow_z"] = matrix[2, :]
        self.header["qoffset_x"] = matrix[0, 3]
        self.header["qoffset_y"] = matrix[1, 3]
        self.header["qoffset_z"] = matrix[2, 3]
        if force_sform:
            self.header["qform_code"] = 0
            self.header["sform_code"] = 1

    def qform_matrix(self) -> np.ndarray:
        """METHOD 2 of the NIfTI standard: build the affine from the quaternion."""
        h = self.header
        b = float(h["quatern_b"])
        c = float(h["quatern_c"])
        d = float(h["quatern_d"])
        a = np.sqrt(max(0.0, 1.0 - b * b - c * c - d * d))
        pixdim = self.pixdim
        qfac = pixdim[0] if pixdim[0] != 0 else 1.0
        dx, dy, dz = pixdim[1], pixdim[2], pixdim[3] * qfac
        return np.array([
            [(a * a + b * b - c * c - d * d) * dx, (2 * b * c - 2 * a * d) * dy,
             (2 * b * d + 2 * a * c) * dz, float(h["qoffset_x"])],
            [(2 * b * c + 2 * a * d) * dx, (a * a + c * c - b * b - d * d) * dy,
             (2 * c * d - 2 * a * b) * dz, float(h["qoffset_y"])],
            [(2 * b * d - 2 * a * c) * dx, (2 * c * d + 2 * a * b) * dy,
             (a * a + d * d - c * c - b * b) * dz, float(h["qoffset_z"])],
            [0.0, 0.0, 0.0, 1.0],
        ])

    def orientation_matrix(self) -> np.ndarray:
        """The voxel-to-world mapping, following the NIfTI-1 method priority.

        ``sform`` (method 3) wins when its code is set, otherwise the quaternion
        based ``qform`` (method 2) is used and promoted to an ``sform``.
        """
        if self.sform_code > 0:
            return self.srow()
        if self.qform_code > 0:
            matrix = self.qform_matrix()
            self.header["sform_code"] = self.qform_code
            self.header["qform_code"] = 0
            return matrix
        raise ValueError("Unknown MRI header: neither sform_code nor qform_code is set.")

    def has_bad_header(self) -> bool:
        """ROAST refuses MRIs whose header carries no usable origin."""
        return (float(self.header["qoffset_x"]) == 0
                and float(np.asarray(self.header["srow_x"], dtype=float)[3]) == 0)

    def geom(self) -> ImageGeom:
        return ImageGeom(srow_to_spm_mat(self.orientation_matrix()), self.img.shape[:3])

    # -- writing -----------------------------------------------------------
    def set_scaling(self, slope=1.0, inter=0.0) -> None:
        """Set ``scl_slope``/``scl_inter`` together.

        A slope without an intercept is an invalid combination, and some MRIs
        carry a NaN intercept, so the two are always written as a pair.
        """
        self.header["scl_slope"] = slope
        self.header["scl_inter"] = inter

    def set_data(self, img: np.ndarray, descrip: str | None = None,
                 datatype: str | None = None) -> None:
        """Replace the voxel data, updating the header the way ROAST does."""
        self.img = img
        dims = np.asarray(self.header["dim"]).copy()
        dims[0] = img.ndim
        dims[1:img.ndim + 1] = img.shape
        dims[img.ndim + 1:] = 1
        self.header["dim"] = dims
        if datatype is not None:
            self.header.set_data_dtype(np.dtype(datatype))
            if np.dtype(datatype) != np.uint8:
                self.set_scaling(1.0, 0.0)
                self.header["cal_max"] = 0
                self.header["cal_min"] = 0
        self.header["glmax"] = np.nanmax(img) if img.size else 0
        self.header["glmin"] = np.nanmin(img) if img.size else 0
        if descrip is not None:
            self.header["descrip"] = descrip

    def save(self, path) -> Path:
        """Write the volume, keeping the header fields untouched."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = copy.deepcopy(self.header)
        slope, inter = header["scl_slope"], header["scl_inter"]
        if np.isfinite(slope) and not np.isfinite(inter):
            header["scl_inter"] = 0.0        # a slope without an intercept is invalid
        dtype = np.dtype(header.get_data_dtype())
        data = np.asarray(self.img)
        if data.dtype != dtype:
            data = data.astype(dtype)
        image = nib.Nifti1Image(data, None, header=header)
        # nibabel recomputes the s/q-form from the affine, so restore the codes
        # and rows we were asked to keep.
        image.header["srow_x"] = header["srow_x"]
        image.header["srow_y"] = header["srow_y"]
        image.header["srow_z"] = header["srow_z"]
        image.header["qoffset_x"] = header["qoffset_x"]
        image.header["qoffset_y"] = header["qoffset_y"]
        image.header["qoffset_z"] = header["qoffset_z"]
        image.header["quatern_b"] = header["quatern_b"]
        image.header["quatern_c"] = header["quatern_c"]
        image.header["quatern_d"] = header["quatern_d"]
        image.header["sform_code"] = header["sform_code"]
        image.header["qform_code"] = header["qform_code"]
        image.header["pixdim"] = header["pixdim"]
        image.header["scl_slope"] = header["scl_slope"]
        image.header["scl_inter"] = header["scl_inter"]
        image.header["descrip"] = header["descrip"]
        nib.save(image, str(path))
        self.filename = str(path)
        return path


def load_untouch(path) -> NiftiVolume:
    """Shorthand for :meth:`NiftiVolume.load` (mirrors ``load_untouch_nii``)."""
    return NiftiVolume.load(path)


def geom_from_nifti(path) -> ImageGeom:
    """Read only the geometry of a volume (the Python side of ``spm_vol``)."""
    return NiftiVolume.load(path).geom()
