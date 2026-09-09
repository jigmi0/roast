"""Shape sampling, point clouds and the spline fit."""

import numpy as np
import pytest

from roast.geometry import (draw_line, draw_cuboid, draw_cylinder, map_to_points,
                            mask_to_edge_point_cloud, ncs2dapprox,
                            project_to_closest_surface_points, get_data_around_target)


def test_draw_line_samples_at_the_requested_density():
    coords = draw_line([0, 0, 0], [1, 0, 0], 10, 2)
    assert coords.shape == (21, 3)
    assert coords[:, 0].max() == pytest.approx(10.0)
    assert np.allclose(coords[:, 1:], 0)


def test_draw_cuboid_fills_the_slab():
    coords = draw_cuboid([50, 50, 50], [20, 10, 4], [1, 0, 0], [0, 1, 0], [0, 0, 1], 2)
    assert coords.shape[0] == 41 * 21 * 9
    assert coords[:, 0].min() == pytest.approx(40.0)
    assert coords[:, 0].max() == pytest.approx(60.0)


def test_draw_cylinder_stays_within_its_radius():
    coords = draw_cylinder(0, 6, [10, 10, 12], [10, 10, 10], 2)
    radius = np.hypot(coords[:, 0] - 10, coords[:, 1] - 10)
    assert radius.max() <= 6.001
    assert coords[:, 2].min() >= 9.99 and coords[:, 2].max() <= 12.01


def test_draw_cylinder_is_deterministic():
    first = draw_cylinder(2, 5, [0, 0, 4], [0, 0, 0], 2)
    second = draw_cylinder(2, 5, [0, 0, 4], [0, 0, 0], 2)
    assert np.array_equal(first, second)


def test_mask_to_edge_point_cloud_returns_one_based_shell():
    volume = np.zeros((20, 20, 20), dtype=bool)
    volume[5:15, 5:15, 5:15] = True
    edge, eroded = mask_to_edge_point_cloud(volume, "erode", np.ones((3, 3, 3)))
    assert len(edge) == 10 ** 3 - 8 ** 3
    assert edge.min() == 6 and edge.max() == 15      # 1-based coordinates
    assert eroded.sum() == 8 ** 3


def test_map_to_points_finds_the_closest_and_farthest():
    goal = np.array([[0.0, 0, 0], [10, 0, 0], [3, 0, 0]])
    distance, index = map_to_points([[2.0, 0, 0]], goal, "closest")
    assert index[0] == 2 and distance[0] == pytest.approx(1.0)
    distance, index = map_to_points([[2.0, 0, 0]], goal, "farthest")
    assert index[0] == 1 and distance[0] == pytest.approx(8.0)


def test_project_to_closest_surface_points_prefers_aligned_directions():
    surface = np.array([[0.0, 0, 10], [0, 0, -10], [10, 0, 0]])
    cosine, order = project_to_closest_surface_points([[0.0, 0, 5]], surface,
                                                      [0.0, 0, 0])
    assert order[0, 0] == 0
    assert cosine[0, 0] == pytest.approx(1.0, abs=1e-6)


def test_get_data_around_target_averages_a_sphere():
    data = np.zeros((21, 21, 21))
    data[9:12, 9:12, 9:12] = 4.0
    assert get_data_around_target(data, [11, 11, 11], 2.0) == pytest.approx(4.0)


def test_ncs2dapprox_reproduces_the_curve():
    t = np.arange(1, 41, dtype=float)
    x, y = t, np.sin(t / 6.0) * 20
    bx, by, breaks = ncs2dapprox(x, y, 1.0)
    assert breaks[0] == 1 and breaks[-1] == len(t)
    assert len(breaks) < len(t)                      # fewer knots than samples
    assert np.allclose(bx, x[breaks - 1])
