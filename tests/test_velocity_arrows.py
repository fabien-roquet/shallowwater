"""Velocity overlays must follow saved C-grid fields and survive contour redraws."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.quiver import Quiver
import numpy as np
import pytest

from shallowwater import make_grid
from shallowwater.visualize import animate_eta


def test_gravity_wave_direction_arrows_preserve_direction_and_hide_rest():
    from pathlib import Path
    import runpy
    import sys
    scripts = Path(__file__).resolve().parents[1]/'scripts'
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    normalize = runpy.run_path(str(scripts/'02_gravity_waves.py'))['direction_vectors']
    u = np.array([[[3., -30., 0., 1e-14]], [[0., 0., 0., 0.]]])
    v = np.array([[[4., 40., 0., 0.]], [[0., 0., 0., 0.]]])
    east, north = normalize(u, v)
    np.testing.assert_allclose(east[0, 0, :2], [.6, -.6])
    np.testing.assert_allclose(north[0, 0, :2], [.8, .8])
    np.testing.assert_allclose(np.hypot(east, north).compressed(), 1.)
    assert east.mask[0, 0, 2:].all()
    assert east.mask[1].all()
    assert np.isfinite(east.data).all() and np.isfinite(north.data).all()


def test_arrows_center_c_grid_velocities_select_frames_and_survive_updates():
    grid = make_grid(4, 3, 40e3, 30e3)
    u = np.broadcast_to(2.*np.arange(5)[None, :], (3, 5)).copy()
    v = np.broadcast_to(4.*np.arange(4)[:, None], (4, 4)).copy()
    out = dict(eta=np.zeros((2, 3, 4)), time=np.array([0., 1.]),
               u=np.stack([u, -u]), v=np.stack([v, 2*v]))
    movie = animate_eta(out, grid, velocity_arrows=True, arrow_stride=2,
                        arrow_reference=2., arrow_scale=40., frames=[1, 0])
    try:
        movie.figure.canvas.draw()
        ax = movie.figure.axes[0]
        arrows = next(a for a in ax.collections if isinstance(a, Quiver))
        np.testing.assert_allclose(arrows.U, [-1., -5., -1., -5.])
        np.testing.assert_allclose(arrows.V, [4., 4., 20., 20.])
        np.testing.assert_allclose(arrows.get_offsets(), [[5e3, 5e3], [25e3, 5e3],
                                                         [5e3, 25e3], [25e3, 25e3]])
        movie.animation._func(1)
        np.testing.assert_allclose(arrows.U, [1., 5., 1., 5.])
        np.testing.assert_allclose(arrows.V, [2., 2., 10., 10.])
        assert arrows.scale == 40.
        assert arrows in ax.collections
        assert sum(isinstance(a, Quiver) for a in ax.collections) == 1
    finally:
        plt.close(movie.figure)


def test_eta_only_remains_supported_and_missing_velocity_is_reported():
    out = dict(eta=np.zeros((2, 3, 4)), time=np.array([0., 1.]))
    movie = animate_eta(out)
    try:
        movie.figure.canvas.draw()
        assert not any(isinstance(a, Quiver) for a in movie.figure.axes[0].collections)
    finally:
        plt.close(movie.figure)
    with pytest.raises(ValueError, match="requires saved 'u' and 'v'"):
        animate_eta(out, velocity_arrows=True)
