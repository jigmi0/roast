"""The MATLAB primitives the port relies on."""

import numpy as np
import pytest

from roast.utils import matlab as m


def test_find_and_ind2sub_are_column_major():
    volume = np.zeros((3, 4), dtype=bool)
    volume[2, 0] = volume[0, 1] = True
    # MATLAB's find walks down the columns first.
    assert m.find(volume).tolist() == [2, 3]
    assert m.ind2sub(volume.shape, m.find(volume)).tolist() == [[3, 1], [1, 2]]


def test_sub2ind_round_trips():
    shape = (5, 7, 3)
    subs = np.array([[1, 1, 1], [5, 7, 3], [2, 4, 2]])
    assert m.ind2sub(shape, m.sub2ind(shape, subs)).tolist() == subs.tolist()


def test_fspecial_gaussian_is_normalised_and_symmetric():
    kernel = m.fspecial_gaussian(5, 0.5)
    assert kernel.shape == (5, 5)
    assert kernel.sum() == pytest.approx(1.0)
    assert np.allclose(kernel, kernel.T)
    assert kernel.argmax() == 12          # the centre of a 5x5 kernel


def test_imfilter_uint8_rounds_and_saturates():
    image = np.zeros((5, 5), dtype=np.uint8)
    image[2, 2] = 255
    out = m.imfilter_uint8(image, m.fspecial_gaussian(5, 1))
    assert out.dtype == np.uint8
    assert out.max() <= 255 and out[2, 2] > 0


def test_prctile_matches_matlab_convention():
    # MATLAB evaluates the percentiles on the (i-0.5)/n grid and clamps outside.
    assert m.prctile([1, 2, 3, 4], 50) == pytest.approx(2.5)
    assert m.prctile([1, 2, 3, 4], 12.5) == pytest.approx(1.0)
    assert m.prctile([1, 2, 3, 4], 95) == pytest.approx(4.0)


def test_bwareaopen_drops_small_components():
    volume = np.zeros((10, 10, 10), dtype=bool)
    volume[1:5, 1:5, 1:5] = True          # 64 voxels
    volume[8, 8, 8] = True                # 1 voxel
    cleaned = m.bwareaopen(volume, 10)
    assert cleaned.sum() == 64
    assert not cleaned[8, 8, 8]


def test_size_of_object_sorts_descending():
    volume = np.zeros((10, 10), dtype=bool)
    volume[0:2, 0:2] = True               # 4 pixels
    volume[5, 5] = True                   # 1 pixel
    sizes, _ = m.size_of_object(volume, 8)
    assert sizes.tolist() == [4, 1]


def test_ismember_rows():
    a = np.array([[3, 4], [9, 9], [1, 2]])
    b = np.array([[1, 2], [3, 4]])
    found, where = m.ismember_rows(a, b)
    assert found.tolist() == [True, False, True]
    assert where.tolist() == [1, -1, 0]


def test_intersect_rows():
    a = np.array([[1, 2], [3, 4], [5, 6]])
    b = np.array([[3, 4], [5, 6], [7, 8]])
    common, ia, ib = m.intersect_rows(a, b)
    assert common.tolist() == [[3, 4], [5, 6]]
    assert ia.tolist() == [1, 2] and ib.tolist() == [0, 1]


def test_cart2sph_round_trip():
    azimuth, elevation, radius = m.cart2sph(1.0, 2.0, 3.0)
    x, y, z = m.sph2cart(azimuth, elevation, radius)
    assert (x, y, z) == pytest.approx((1.0, 2.0, 3.0))


def test_edge_sobel_outlines_a_square():
    image = np.zeros((20, 20))
    image[5:15, 5:15] = 1.0
    edges = m.edge_sobel(image)
    assert edges.any()
    # the outline hugs the border of the square, nothing in the middle
    assert not edges[9:11, 9:11].any()
    assert edges[4:16, 4:16].sum() == edges.sum()
