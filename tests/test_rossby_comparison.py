"""Check identical initial conditions, sign reversal, and saved-time continuations."""
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import numpy as np

from shallowwater import ModelParams, make_grid, run_model, zero_forcing

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from _rossby_comparison import case_setup, sampled_integration


def test_cyclonic_tiny_setup_is_exactly_the_opposite_initial_state():
    args = SimpleNamespace(smoke=False, quick=True, days=4., frames=12)
    anti = case_setup('tiny', args)
    cyclone = case_setup('tiny-cyclonic', args)
    for a, c in zip(anti[2], cyclone[2]):
        np.testing.assert_array_equal(a, -c)
    assert anti[3] == cyclone[3]
    np.testing.assert_array_equal(anti[4], cyclone[4])
    assert anti[1] == cyclone[1]


def test_nonuniform_saved_times_preserve_the_same_nonlinear_trajectory():
    grid = make_grid(10, 8, 160e3, 120e3)
    params = ModelParams(H=100., f0=1e-4, beta=2e-11, y0=60e3, linear=False)
    rng = np.random.default_rng(15)
    initial = (rng.normal(0, .001, (8, 10)), rng.normal(0, .001, (8, 11)), rng.normal(0, .001, (9, 10)))
    initial[1][:, 0] = initial[1][:, -1] = 0.
    initial[2][0] = initial[2][-1] = 0.
    dt = 10.
    indices = np.array([0, 1, 3, 7, 12])
    sampled, _ = sampled_integration(grid, params, initial, dt, indices)
    full = run_model(120., dt, grid, params, zero_forcing, lambda g, p: initial, save_every=1)
    np.testing.assert_array_equal(sampled['time'], np.asarray(full['time'])[indices])
    np.testing.assert_allclose(sampled['eta'], np.asarray(full['eta'])[indices], rtol=0., atol=1e-15)


def test_partial_linear_cache_adds_early_times_without_interpolating():
    grid = make_grid(8, 6, 160e3, 120e3)
    params = ModelParams(H=100., f0=1e-4, beta=0., linear=True)
    initial = (np.zeros((6, 8)), np.zeros((6, 9)), np.zeros((7, 8)))
    initial[0][2:4, 3:5] = .01
    dt = 10.
    full = run_model(100., dt, grid, params, zero_forcing, lambda g, p: initial, save_every=1)
    indices = np.array([0, 1, 2, 4, 8, 10])
    previous = dict(time=np.array([0., 40., 80., 100.]), eta=np.asarray(full['eta'])[[0, 4, 8, 10]])
    sampled, stats = sampled_integration(grid, params, initial, dt, indices, existing=previous)
    np.testing.assert_allclose(sampled['eta'], np.asarray(full['eta'])[indices], rtol=0., atol=1e-15)
    assert stats['reused_linear_frames'] == 3


def test_cyclone_reuses_only_a_matching_sign_reversed_linear_reference(tmp_path):
    from dataclasses import asdict
    import json
    from _rossby_comparison import previous_linear
    from _experiment_tools import model_fingerprint
    args = SimpleNamespace(smoke=True, quick=False, days=.05, frames=4)
    grid, params, initial, dt, indices, _ = case_setup('tiny-cyclonic', args)
    path = tmp_path/'experiment_11_tiny_eddy_linear.npz'
    anti = [-a for a in initial]
    np.savez_compressed(path, eta=np.stack([anti[0], .9*anti[0]]), time=[0., dt],
                        initial_u=anti[1], initial_v=anti[2], x=grid.x_c, y=grid.y_c)
    path.with_suffix('.json').write_text(json.dumps(dict(configuration=dict(
        solver_fingerprint=model_fingerprint(), parameters=asdict(params),
        experiment_fingerprint='matching', dt=dt))))
    result = previous_linear(tmp_path, 'tiny_eddy_cyclonic', grid, params, initial, dt, 'matching')
    np.testing.assert_array_equal(result['eta'][0], initial[0])
    np.testing.assert_array_equal(result['eta'][1], .9*initial[0])
    assert previous_linear(tmp_path, 'tiny_eddy_cyclonic', grid, params, initial, dt, 'changed') is None
