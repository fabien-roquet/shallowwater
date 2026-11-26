"""Analytic C-grid Poincaré modes and hemisphere symmetry for experiment 15."""
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest

from shallowwater import ModelParams, make_grid

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
EXP = runpy.run_path(str(SCRIPTS/'15_inertial_waves.py'))


@pytest.mark.parametrize('a', [.2, 1., 5.])
def test_standing_mode_matches_exact_discrete_frequency_and_energy(a):
    ld = np.sqrt(9.81*100)/EXP['FREF']
    ly = np.pi*ld/a
    grid = make_grid(6, 40, ly, ly, periodic_x=True)
    params = ModelParams(H=100., f0=EXP['FREF'], beta=0., r=0., linear=True)
    period = 2*np.pi/EXP['standing_frequency'](grid, params, discrete=True)
    data = EXP['integrate_case'](grid, params, EXP['standing_initial'](grid, params),
                                 1.37*period, 51)
    exact = EXP['standing_solution'](grid, params, data['time'][-1])
    for key, field in zip(('eta', 'u', 'v'), exact):
        # SSPRK3 introduces a small time-discretization error.
        scale = max(float(np.max(np.abs(field))), .01)
        np.testing.assert_allclose(data[key][-1], field, atol=4e-4*scale, rtol=0.)
    assert np.max(np.abs(data['energy']/data['energy'][0]-1)) < 4e-4
    assert len(data['time']) == 51
    assert np.max(np.abs(data['eta'].mean(axis=(1, 2)))) < 1e-13


def test_reversing_hemisphere_reverses_zonal_rotation_only():
    grid = make_grid(4, 24, 2e6, 2e6, periodic_x=True)
    params = [ModelParams(H=100., f0=f, beta=0., r=0., linear=True)
              for f in (1e-4, -1e-4)]
    data = [EXP['integrate_case'](grid, p, EXP['standing_initial'](grid, p), .6*EXP['TF'], 25)
            for p in params]
    np.testing.assert_allclose(data[0]['eta'], data[1]['eta'], atol=1e-13)
    np.testing.assert_allclose(data[0]['v'], data[1]['v'], atol=1e-13)
    np.testing.assert_allclose(data[0]['u'], -data[1]['u'], atol=1e-13)
    # Initially northward flow turns eastward in the northern hemisphere.
    assert data[0]['u'][1, grid.Ny//2, 0] > 0
    assert data[1]['u'][1, grid.Ny//2, 0] < 0
