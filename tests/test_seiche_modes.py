"""The uniform seiche mode has zero frequency and remains a nonzero offset."""
from pathlib import Path
import runpy
import sys

import numpy as np

from shallowwater import ModelParams, make_grid, run_model, zero_forcing

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
EXP = runpy.run_path(str(SCRIPTS/'06_seiche.py'))


def test_zero_mode_is_uniform_stationary_and_mass_preserving():
    grid = make_grid(8, 6, 2e6, 2e6)
    params = ModelParams(H=1000., f0=0., beta=0., r=0., linear=True)
    assert EXP['mode_frequency'](grid, params, 0, 0) == 0.
    initial = EXP['seiche_ic'](grid, params, A=.1, m=0, n=0)
    out = run_model(3600., 100., grid, params, zero_forcing, lambda g, p: initial, save_every=6)
    np.testing.assert_allclose(out['eta'], .1, rtol=0., atol=1e-15)
    np.testing.assert_array_equal(np.asarray(out['u']), 0.)
    np.testing.assert_array_equal(np.asarray(out['v']), 0.)


def test_first_oscillating_modes_have_the_correct_frequencies():
    grid = make_grid(32, 32, 2e6, 2e6)
    params = ModelParams(H=1000., f0=0., beta=0., linear=True)
    frequency = EXP['mode_frequency']
    fundamental = np.pi*np.sqrt(params.g*params.H)/grid.Lx
    np.testing.assert_allclose(frequency(grid, params, 1, 0), fundamental)
    np.testing.assert_allclose(frequency(grid, params, 0, 1), fundamental)
    np.testing.assert_allclose(frequency(grid, params, 1, 1), np.sqrt(2)*fundamental)
    assert EXP['make_parser']().parse_args([]).modes == [(1, 0), (1, 1)]
