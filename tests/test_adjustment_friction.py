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
