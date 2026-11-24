"""Check that the short-wave initial condition actually excludes long modes."""
from pathlib import Path
import runpy
import sys

import numpy as np

from shallowwater import make_grid

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from _rossby_small_scales import initial_height


def test_filtered_disturbance_has_no_long_sine_modes():
    grid = make_grid(128, 96, 800e3, 600e3)
    ld = np.sqrt(9.81*100)/1e-4
    eta = initial_height(grid, ld, filtered=True)
    X, Y = np.meshgrid(grid.x_c, grid.y_c)
    for n in range(1, 8):
        for m in range(1, 6):
            if np.hypot(n*np.pi/grid.Lx, m*np.pi/grid.Ly)*ld < 8.:
                mode = np.sin(n*np.pi*X/grid.Lx)*np.sin(m*np.pi*Y/grid.Ly)
                assert abs(np.mean(eta*mode)) < 1e-14
    assert np.isclose(eta.max(), .15)
    assert eta.min() < 0.  # Removing long modes requires compensating rings.


def test_experiment_11_import_does_not_start_integrations(monkeypatch):
    import shallowwater
    def forbidden(*args, **kwargs):
        raise AssertionError('Importing an experiment must not start a simulation.')
    monkeypatch.setattr(shallowwater, 'run_model', forbidden)
    module = runpy.run_path(str(SCRIPTS/'11_Rossby_wave_propagation.py'))
    assert callable(module['run_existing_cases'])
    assert module['make_parser']().parse_args(['--case', 'tiny']).case == 'tiny'
