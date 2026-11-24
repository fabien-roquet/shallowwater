"""Analytic checks for the channel streamfunction plotted in experiment 13."""
from pathlib import Path
import runpy

import numpy as np

from shallowwater import make_grid

TOOLS = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts' / '_experiment_tools.py'))
streamfunction = TOOLS['channel_streamfunction']


def test_streamfunction_retains_uniform_zonal_flow_transport():
    grid = make_grid(32, 16, 800e3, 400e3, periodic_x=True)
    speeds = np.array([0.3, -0.7])
    zeta = np.zeros((2, grid.Ny, grid.Nx))
    mean_u = np.broadcast_to(speeds[:, None], (2, grid.Ny))
    psi = streamfunction(zeta, mean_u, grid)
    expected = -speeds[:, None, None]*(grid.y_c[None, :, None]-grid.Ly/2)
    np.testing.assert_allclose(psi, np.broadcast_to(expected, psi.shape), atol=1e-8)


def test_streamfunction_inverts_periodic_mode_with_impermeable_walls():
    grid = make_grid(40, 24, 800e3, 400e3, periodic_x=True)
    X, Y = np.meshgrid(grid.x_c, grid.y_c)
    expected = 1e5*np.cos(4*np.pi*X/grid.Lx)*np.sin(np.pi*Y/grid.Ly)
    eigenvalue = (-4*np.sin(2*np.pi/grid.Nx)**2/grid.dx**2
                  -4*np.sin(np.pi/(2*grid.Ny))**2/grid.dy**2)
    psi = streamfunction(eigenvalue*expected, np.zeros(grid.Ny), grid)
    np.testing.assert_allclose(psi, expected, atol=1e-8)


def test_gyre_spinup_cache_invalidates_when_physics_changes(tmp_path):
    """A visualization rerun may reuse a state; a new drag rate must not."""
    import sys
    from dataclasses import asdict
    from shallowwater import ModelParams, uniform_wind_forcing
    from functools import partial
    scripts = str(Path(__file__).resolve().parents[1] / 'scripts')
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    experiment = runpy.run_path(str(Path(scripts) / '14_double_gyre_adjustment.py'))
    grid = make_grid(8, 6, 80e3, 60e3)
    params = ModelParams(H=200., r=1e-6)
    forcing = partial(uniform_wind_forcing, tau_x=0.03)
    path = tmp_path / 'spinup.npz'
    kwargs = dict(days=0.005, frames=3, path=path)
    first = experiment['spin_up'](grid, params, forcing,
                                 configuration=asdict(params), **kwargs)
    cached = experiment['spin_up'](grid, params, forcing,
                                  configuration=asdict(params), **kwargs)
    assert not first[-1] and cached[-1]
    for a, b in zip(first[0], cached[0]):
        np.testing.assert_array_equal(a, b)
    params.r *= 2
    changed = experiment['spin_up'](grid, params, forcing,
                                   configuration=asdict(params), **kwargs)
    assert not changed[-1]


def test_wind_ramp_starts_at_zero_and_builds_opposite_gyre_curls():
    import sys
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    ramp = runpy.run_path(str(scripts / '_gyre_wind_ramp.py'))
    grid = make_grid(48, 64, 6e6, 1.6e6)
    stress = ramp['wind_stress'](grid, .08, 100e3, seed_fraction=0.)
    np.testing.assert_allclose(stress.mean(axis=0), 0., atol=1e-17)
    np.testing.assert_allclose(stress.max(), .08*np.cos(np.pi/grid.Ny))
    jet = ramp['wind_stress'](grid, .08, 100e3, seed_fraction=0., profile='jet')
    np.testing.assert_allclose(jet.max(), .08)
    curl = -np.gradient(stress, grid.dy, axis=0)
    assert np.all(curl[:32] < 0.) and np.all(curl[32:] > 0.)
    times = np.array([-1., 0., 90., 180., 300.])*86400
    np.testing.assert_array_equal(ramp['ramp_fraction'](times, 180.), [0., 0., .5, 1., 1.])
    seeded = ramp['wind_stress'](grid, .08, 100e3)
    assert np.max(np.abs(seeded-stress)) < .002*.08


def test_spatial_viscosity_dissipates_energy_and_reduces_to_laplacian():
    import sys
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    ramp = runpy.run_path(str(scripts / '_gyre_wind_ramp.py'))
    from shallowwater import operators
    rng = np.random.default_rng(46)
    a = rng.normal(size=(9, 13))
    nu = rng.uniform(10., 1000., size=a.shape)
    tendency = ramp['variable_viscosity_tendency'](a, nu, 2., 3.)
    work = np.sum(a*tendency)
    expected = (-np.sum(.5*(nu[:, 1:]+nu[:, :-1])*(np.diff(a, axis=1)/2.)**2)
                -np.sum(.5*(nu[1:]+nu[:-1])*(np.diff(a, axis=0)/3.)**2))
    np.testing.assert_allclose(work, expected, rtol=1e-14)
    assert work < 0
    uniform = ramp['variable_viscosity_tendency'](a, np.full_like(a, 50.), 2., 3.)
    np.testing.assert_allclose(uniform, 50.*operators.laplacian_u(a, 2., 3.), atol=1e-13)


def test_wind_continuation_has_no_stress_jump_and_joins_clocks():
    import sys
    scripts = str(Path(__file__).resolve().parents[1] / 'scripts')
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from _gyre_wind_sweep import stage_fraction, join_histories
    stress = .08*stage_fraction(np.array([0., 30., 60., 90.])*86400, .04, .08, 60.)
    np.testing.assert_allclose(stress, [.04, .06, .08, .08])
    chunks = [dict(time_days=np.array([0., 1., 2.]), x=np.arange(4), y=np.arange(3),
                   wind_stress_peak=np.array(values), temporal_mean_window_days=1.)
              for values in ([0., .02, .04], [.04, .06, .08])]
    joined = join_histories(chunks)
    np.testing.assert_array_equal(joined['time_days'], [0., 1., 2., 3., 4.])
    np.testing.assert_allclose(joined['wind_stress_peak'], [0., .02, .04, .06, .08])
    np.testing.assert_array_equal(joined['x'], np.arange(4))
