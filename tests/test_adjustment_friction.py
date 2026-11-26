"""Coastal momentum friction removes kinetic energy without damping mass."""
from pathlib import Path
import runpy
import sys

import numpy as np

from shallowwater import ModelParams, make_grid

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
EXP = runpy.run_path(str(SCRIPTS/'10_geostrophic_adjustment.py'))


def test_coastal_friction_damps_energy_and_leaves_the_centre_undamped():
    grid = make_grid(32, 24, 4e6, 4e6)
    hook, mask = EXP['momentum_sponge'](grid, width=800e3, tau=3600.)
    rng = np.random.default_rng(10)
    state = dict(eta=rng.normal(size=(grid.Ny, grid.Nx)),
                 u=rng.normal(size=(grid.Ny, grid.Nx+1)),
                 v=rng.normal(size=(grid.Ny+1, grid.Nx)))
    de, du, dv = hook(state, 0., grid, ModelParams())
    assert de is None
    assert np.sum(state['u']*du)+np.sum(state['v']*dv) < 0
    assert 0 <= mask.min() <= mask.max() <= 1
    assert mask[grid.Ny//2, grid.Nx//2] == 0
    assert du[grid.Ny//2, grid.Nx//2] == 0
    assert dv[grid.Ny//2, grid.Nx//2] == 0
    assert mask[0, 0] > 0


def test_finely_sampled_adjustment_matches_uninterrupted_run(tmp_path):
    from types import SimpleNamespace
    from _adjustment_3d import run_3d_case
    from shallowwater import run_model, zero_forcing

    args = SimpleNamespace(hours=4., smoke=False, sponge_width_km=800.,
                           sponge_hours=1., output_dir=tmp_path, replot=False)
    def setup(args):
        grid = make_grid(16, 16, 4e6, 4e6)
        params = ModelParams(H=1000., f0=1e-4, beta=0., r=0., linear=True)
        return grid, params, 14400., 2*np.pi/params.f0, 30., 1
    grid, params, data = run_3d_case(args, setup, EXP['ic_balanced_fraction'], EXP['momentum_sponge'])
    sponge, _ = EXP['momentum_sponge'](grid, width=800e3, tau=3600.)
    reference = run_model(14400., 30., grid, params, zero_forcing,
                          EXP['ic_balanced_fraction'], save_every=480, hooks=[sponge])
    np.testing.assert_allclose(data['eta'][-1], reference['eta'][-1], rtol=0., atol=1e-14)
    assert data['time'][0] == 0. and data['time'][-1] == 14400.
    spacing = np.diff(data['time'])
    np.testing.assert_allclose(spacing, 120.)
    _, no_rotation, other = run_3d_case(args, setup, EXP['ic_balanced_fraction'],
                                        EXP['momentum_sponge'], f0=0.)
    assert no_rotation.f0 == 0.
    np.testing.assert_array_equal(data['time'], other['time'])
    np.testing.assert_array_equal(data['eta'][0], other['eta'][0])
    assert not np.allclose(data['eta'][-1], other['eta'][-1])
