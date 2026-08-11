import numpy as np

from shallowwater import (
    ModelParams,
    center_depth,
    compute_dt_cfl,
    enforce_bcs,
    make_grid,
    run_model,
    setup_initial_state,
    zero_forcing,
)


def test_grid_shapes_and_spacing():
    grid = make_grid(12, 7, 120_000.0, 35_000.0)

    assert grid.dx == 10_000.0
    assert grid.dy == 5_000.0
    assert grid.x_c.shape == (12,)
    assert grid.y_c.shape == (7,)
    assert grid.x_u.shape == (13,)
    assert grid.y_v.shape == (8,)


def test_initial_state_shapes_and_rest():
    grid = make_grid(10, 6, 100_000.0, 60_000.0)
    params = ModelParams()

    eta, u, v = setup_initial_state(grid, params, mode="rest")

    assert eta.shape == (6, 10)
    assert u.shape == (6, 11)
    assert v.shape == (7, 10)
    assert np.count_nonzero(eta) == 0
    assert np.count_nonzero(u) == 0
    assert np.count_nonzero(v) == 0


def test_center_depth_accepts_supported_inputs():
    grid = make_grid(5, 3, 50_000.0, 30_000.0)

    scalar = center_depth(grid, 400.0)
    profile = center_depth(grid, np.linspace(100.0, 500.0, grid.Nx))
    field = np.arange(grid.Nx * grid.Ny, dtype=float).reshape(grid.Ny, grid.Nx) + 1.0

    assert scalar.shape == (grid.Ny, grid.Nx)
    assert np.all(scalar == 400.0)
    assert np.all(profile[0] == profile[-1])
    np.testing.assert_array_equal(center_depth(grid, field), field)


def test_center_depth_rejects_bad_values_and_shape():
    grid = make_grid(5, 3, 50_000.0, 30_000.0)

    for bad in (0.0, -1.0, np.full((3, 5), np.nan), np.ones((2, 5))):
        try:
            center_depth(grid, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid depth to fail: {bad!r}")


def test_cfl_uses_deepest_water():
    grid = make_grid(8, 4, 80_000.0, 40_000.0)
    depth = np.full((grid.Ny, grid.Nx), 100.0)
    depth[:, -1] = 400.0
    params = ModelParams(H=depth, g=10.0)

    dt = compute_dt_cfl(grid, params, cfl=0.5)

    expected = 0.5 * min(grid.dx, grid.dy) / np.sqrt(10.0 * 400.0)
    assert np.isclose(dt, expected)


def test_boundary_enforcement_sets_normal_flow_to_zero():
    u = np.ones((4, 7))
    v = np.ones((5, 6))

    enforce_bcs(u, v)

    assert np.all(u[:, (0, -1)] == 0.0)
    assert np.all(v[(0, -1), :] == 0.0)
    assert np.all(u[:, 1:-1] == 1.0)
    assert np.all(v[1:-1, :] == 1.0)


def test_short_rest_integration_remains_at_rest():
    grid = make_grid(12, 8, 120_000.0, 80_000.0)
    params = ModelParams(H=500.0, f0=0.0, beta=0.0, r=0.0, linear=True)
    dt = compute_dt_cfl(grid, params, cfl=0.4)

    out = run_model(
        tmax=5 * dt,
        dt=dt,
        grid=grid,
        params=params,
        forcing_fn=zero_forcing,
        ic_fn=lambda g, p: setup_initial_state(g, p, mode="rest"),
        save_every=1,
    )

    assert len(out["time"]) == 6
    assert np.max(np.abs(out["eta"][-1])) == 0.0
    assert np.max(np.abs(out["u"][-1])) == 0.0
    assert np.max(np.abs(out["v"][-1])) == 0.0

