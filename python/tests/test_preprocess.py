"""Re-orientation, padding and resampling of the MRI."""

import shutil

import numpy as np
import pytest

from roast.config import example_dir
from roast.io import NiftiVolume
from roast.preprocess import (convert_to_ras, convert_to_ras_point_cloud,
                              orientation_of, resamp_to_one_mm, reslice_to_grid,
                              bounding_box, zero_padding, rigid_matrix)

MNI = example_dir() / "MNI152_T1_1mm.nii"


@pytest.fixture
def subject(tmp_path):
    path = tmp_path / "subj.nii"
    shutil.copy(MNI, path)
    return path


def test_mni152_is_detected_as_non_ras():
    _, perm, flip = orientation_of(NiftiVolume.load(MNI))
    assert perm.tolist() == [0, 1, 2]
    assert flip[0] < 0                      # the MNI152 template is stored LAS


def test_convert_to_ras_flips_the_volume(subject):
    path, is_non_ras = convert_to_ras(subject)
    assert is_non_ras and path.name == "subj_ras.nii"
    volume = NiftiVolume.load(path)
    assert np.allclose(np.diag(volume.srow()[:3, :3]), [1, 1, 1])
    original = NiftiVolume.load(subject)
    assert np.array_equal(volume.img, np.flip(original.img, axis=0))
    # an already-RAS volume is left alone
    assert convert_to_ras(path) == (path, False)


def test_convert_to_ras_point_cloud_mirrors_the_volume(subject):
    dims = np.asarray(NiftiVolume.load(subject).header["dim"], dtype=float)
    points = np.array([[10.0, 20.0, 30.0]])
    moved, perm = convert_to_ras_point_cloud(subject, points)
    assert perm.tolist() == [0, 1, 2]
    assert moved[0].tolist() == [dims[1] - 10 + 1, 20.0, 30.0]


def test_zero_padding_grows_the_volume_and_shifts_the_origin(subject):
    path = zero_padding(subject, 5)
    assert path.name == "subj_padded5.nii"
    original, padded = NiftiVolume.load(subject), NiftiVolume.load(path)
    assert padded.img.shape == tuple(np.array(original.img.shape) + 10)
    assert padded.img[5:-5, 5:-5, 5:-5].sum() == original.img.sum()
    # the world coordinates of the anatomy are unchanged
    corner_before = original.srow() @ [0, 0, 0, 1]
    corner_after = padded.srow() @ [5, 5, 5, 1]
    assert np.allclose(corner_before, corner_after)
    # re-padding an already padded file is a no-op
    assert zero_padding(path, 5) == path


def test_resampling_a_1mm_volume_is_a_no_op(subject):
    path, do_resamp = resamp_to_one_mm(subject, True)
    assert path == subject and do_resamp is False


def test_resampling_changes_the_grid(tmp_path):
    source = NiftiVolume.load(MNI)
    small = source.copy()
    small.img = np.ascontiguousarray(source.img[::2, ::2, ::2])
    dims = np.asarray(small.header["dim"]).copy()
    dims[1:4] = small.img.shape
    small.header["dim"] = dims
    pixdim = small.pixdim.copy()
    pixdim[1:4] = 2.0
    small.header["pixdim"] = pixdim
    srow = small.srow()
    srow[:3, :3] = srow[:3, :3] * 2
    small.set_srow(srow, force_sform=True)
    path = tmp_path / "coarse.nii"
    small.save(path)

    resampled, do_resamp = resamp_to_one_mm(path, True)
    assert do_resamp and resampled.name == "coarse_1mm.nii"
    out = NiftiVolume.load(resampled)
    assert np.allclose(out.resolution, [1, 1, 1])
    assert out.img.shape[0] > small.img.shape[0]


def test_bounding_box_spans_the_volume():
    geom = NiftiVolume.load(MNI).geom()
    bbox = bounding_box(geom.mat, geom.dim)
    assert bbox.shape == (2, 3)
    assert np.all(bbox[1] > bbox[0])


def test_reslice_to_grid_recovers_the_same_image():
    volume = NiftiVolume.load(MNI)
    geom = volume.geom()
    same = reslice_to_grid(volume, geom.mat, geom.mat, geom.dim)
    assert np.allclose(same, volume.img, atol=1e-3)


def test_rigid_matrix_is_orthonormal():
    matrix = rigid_matrix([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
    rotation = matrix[:3, :3]
    assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)
    assert matrix[:3, 3].tolist() == [1.0, 2.0, 3.0]
