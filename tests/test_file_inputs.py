import numpy as np
import pytest

from shallowwater import (
    ModelParams,
    compute_dt_cfl,
    load_bathymetry,
    make_grid,
    make_wind_forcing,
    make_wind_forcing_from_file,
    run_model,
    setup_initial_state,
)


@pytest.fixture
def grid():
    return make_grid(6, 4, 60_000.0, 40_000.0)


def test_load_bathymetry_supported_formats(tmp_path, grid):
    depth = 100.0 + np.arange(grid.Ny * grid.Nx).reshape(grid.Ny, grid.Nx)
    np.save(tmp_path / "depth.npy", depth)
    np.savez(tmp_path / "depth.npz", H=depth, x=grid.x_c, y=grid.y_c)
    np.savetxt(tmp_path / "depth.csv", depth, delimiter=",")

    for name in ("depth.npy", "depth.npz", "depth.csv"):
        np.testing.assert_allclose(load_bathymetry(tmp_path / name, grid), depth)


@pytest.mark.parametrize(
    "name,values",
    [
        ("wrong.npy", np.ones((3, 6))),
        ("nan.npy", np.full((4, 6), np.nan)),
        ("zero.npy", np.zeros((4, 6))),
    ],
)
def test_load_bathymetry_rejects_invalid_maps(tmp_path, grid, name, values):
    np.save(tmp_path / name, values)
    with pytest.raises(ValueError):
        load_bathymetry(tmp_path / name, grid)


def test_load_bathymetry_rejects_missing_key_and_bad_coordinates(tmp_path, grid):
    depth = np.full((grid.Ny, grid.Nx), 200.0)
    np.savez(tmp_path / "missing.npz", depth=depth)
    np.savez(tmp_path / "coords.npz", H=depth, x=grid.x_c + 1.0, y=grid.y_c)

    with pytest.raises(ValueError, match="does not contain"):
        load_bathymetry(tmp_path / "missing.npz", grid)
    with pytest.raises(ValueError, match="coordinates"):
        load_bathymetry(tmp_path / "coords.npz", grid)


def test_loaded_bathymetry_matches_in_memory_short_run(tmp_path, grid):
    depth = np.repeat(np.linspace(400.0, 100.0, grid.Nx)[None, :], grid.Ny, axis=0)
    np.save(tmp_path / "depth.npy", depth)

    def integrate(H):
        params = ModelParams(H=H, f0=0.0, beta=0.0, linear=True)
        dt = compute_dt_cfl(grid, params, cfl=0.3)
        return run_model(
            5 * dt,
            dt,
            grid,
            params,
            lambda t, g, p: (
                np.zeros((g.Ny, g.Nx + 1)),
                np.zeros((g.Ny + 1, g.Nx)),
                np.zeros((g.Ny, g.Nx)),
            ),
            lambda g, p: setup_initial_state(g, p, mode="gaussian_bump", amp=0.01),
            save_every=5,
            out_vars=("eta",),
        )

    direct = integrate(depth)
    loaded = integrate(load_bathymetry(tmp_path / "depth.npy", grid))
    np.testing.assert_allclose(direct["eta"][-1], loaded["eta"][-1])


def test_static_wind_map_and_envelope(tmp_path, grid):
    tau_x = np.full((grid.Ny, grid.Nx), 0.12)
    tau_y = np.full((grid.Ny, grid.Nx), -0.03)
    np.savez(tmp_path / "wind.npz", tau_x=tau_x, tau_y=tau_y)

    forcing = make_wind_forcing_from_file(
        tmp_path / "wind.npz", grid, envelope=lambda t: t / 10.0
    )
    taux, tauy, source = forcing(5.0, grid, ModelParams())

    assert taux.shape == (grid.Ny, grid.Nx + 1)
    assert tauy.shape == (grid.Ny + 1, grid.Nx)
    np.testing.assert_allclose(taux, 0.06)
    np.testing.assert_allclose(tauy, -0.015)
    assert np.count_nonzero(source) == 0


def test_wind_file_is_loaded_only_when_callable_is_created(tmp_path, grid, monkeypatch):
    tau_x = np.full((grid.Ny, grid.Nx), 0.1)
    tau_y = np.zeros_like(tau_x)
    path = tmp_path / "wind.npz"
    np.savez(path, tau_x=tau_x, tau_y=tau_y)

    import shallowwater.forcing as forcing_module

    real_load = forcing_module.np.load
    calls = []

    def counted_load(*args, **kwargs):
        calls.append(args[0])
        return real_load(*args, **kwargs)

    monkeypatch.setattr(forcing_module.np, "load", counted_load)
    forcing = make_wind_forcing_from_file(path, grid)
    forcing(0.0, grid, ModelParams())
    forcing(1.0, grid, ModelParams())

    assert len(calls) == 1


def test_wind_map_validation(grid):
    with pytest.raises(ValueError, match="shape"):
        make_wind_forcing(np.ones((3, 6)), np.ones((4, 6)), grid)
    bad = np.ones((grid.Ny, grid.Nx))
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        make_wind_forcing(bad, np.zeros_like(bad), grid)


def test_wind_file_rejects_unsupported_format_and_missing_keys(tmp_path, grid):
    text_path = tmp_path / "wind.csv"
    text_path.write_text("0,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match=".npz"):
        make_wind_forcing_from_file(text_path, grid)

    np.savez(tmp_path / "missing.npz", tau_x=np.zeros((grid.Ny, grid.Nx)))
    with pytest.raises(ValueError, match="missing key"):
        make_wind_forcing_from_file(tmp_path / "missing.npz", grid)


def test_uniform_wind_ramp_and_shutoff(grid):
    from shallowwater import uniform_wind_forcing

    taux, tauy, _ = uniform_wind_forcing(
        5.0, grid, ModelParams(), tau_x=0.2, tau_y=-0.1, t_ramp=10.0
    )
    np.testing.assert_allclose(taux, 0.1)
    np.testing.assert_allclose(tauy, -0.05)

    taux, tauy, _ = uniform_wind_forcing(
        12.0, grid, ModelParams(), tau_x=0.2, tau_y=-0.1, t_off=10.0
    )
    np.testing.assert_allclose(taux, 0.0)
    np.testing.assert_allclose(tauy, 0.0)
