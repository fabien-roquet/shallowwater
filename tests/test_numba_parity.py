import numpy as np
import pytest


numba = pytest.importorskip("numba")

from shallowwater import operators_numba as nbops  # noqa: E402


def test_numba_average_and_divergence_match_numpy_formulas():
    rng = np.random.default_rng(42)
    centers = rng.normal(size=(5, 7))
    fx = rng.normal(size=(5, 8))
    fy = rng.normal(size=(6, 7))

    expected_u = np.zeros((5, 8))
    expected_u[:, 1:7] = 0.5 * (centers[:, :-1] + centers[:, 1:])
    expected_u[:, 0] = centers[:, 0]
    expected_u[:, -1] = centers[:, -1]
    expected_div = (fx[:, 1:] - fx[:, :-1]) / 2.0 + (fy[1:, :] - fy[:-1, :]) / 3.0

    np.testing.assert_allclose(nbops.avg_center_to_u_nb(centers), expected_u)
    np.testing.assert_allclose(nbops.divergence_nb(fx, fy, 2.0, 3.0), expected_div)


def test_numba_staggered_velocity_interpolation_matches_reference():
    rng = np.random.default_rng(7)
    u = rng.normal(size=(5, 8))
    v = rng.normal(size=(6, 7))

    expected_v_on_u = np.zeros((5, 8))
    expected_v_on_u[:, 1:7] = 0.25 * (
        v[:-1, :-1] + v[1:, :-1] + v[:-1, 1:] + v[1:, 1:]
    )
    expected_u_on_v = np.zeros((6, 7))
    expected_u_on_v[1:5, :] = 0.25 * (
        u[:-1, :-1] + u[:-1, 1:] + u[1:, :-1] + u[1:, 1:]
    )

    np.testing.assert_allclose(nbops.v_on_u_nb(v), expected_v_on_u)
    np.testing.assert_allclose(nbops.u_on_v_nb(u), expected_u_on_v)



def test_numba_closed_curl_matches_corner_stencil_including_walls():
    rng = np.random.default_rng(19)
    for ny, nx in ((2, 3), (5, 7)):
        u = rng.normal(size=(ny, nx+1))
        v = rng.normal(size=(ny+1, nx))
        u[:, [0, -1]] = 0
        v[[0, -1], :] = 0
        corner = np.zeros((ny+1, nx+1))
        corner[:, 1:-1] = np.diff(v, axis=1)/2.
        corner[1:-1] -= np.diff(u, axis=0)/3.
        expected = .25*(corner[:-1, :-1]+corner[1:, :-1]+corner[:-1, 1:]+corner[1:, 1:])
        np.testing.assert_allclose(nbops.curl_on_center_nb(u, v, 2., 3.), expected, atol=1e-14)
