import numpy as np

from shallowwater import (
    ModelParams,
    compute_dt_cfl,
    make_grid,
    run_model,
    shelf_bathymetry,
    setup_initial_state,
    wind_gyre_forcing,
    zero_forcing,
)


def _right_going_pulse(grid, params, *, x0, radius, amplitude=0.05):
    """Linear one-dimensional eta/u pulse travelling toward increasing x."""
    eta_line = amplitude * np.exp(-((grid.x_c - x0) / radius) ** 2)
    eta = np.repeat(eta_line[None, :], grid.Ny, axis=0)
    u_line = amplitude * np.exp(-((grid.x_u - x0) / radius) ** 2)
    u = np.repeat((np.sqrt(params.g * float(params.H)) / float(params.H) * u_line)[None, :], grid.Ny, axis=0)
    v = np.zeros((grid.Ny + 1, grid.Nx))
    return eta, u, v


def test_uniform_depth_wave_speed_matches_theory():
    grid = make_grid(240, 4, 1_200_000.0, 20_000.0)
    params = ModelParams(H=400.0, g=9.81, f0=0.0, beta=0.0, r=0.0, linear=True)
    dt = compute_dt_cfl(grid, params, cfl=0.4)
    duration = 2.0 * 3600.0
    x0 = 250_000.0

    out = run_model(
        tmax=duration,
        dt=dt,
        grid=grid,
        params=params,
        forcing_fn=zero_forcing,
        ic_fn=lambda g, p: _right_going_pulse(g, p, x0=x0, radius=45_000.0),
        save_every=10_000,
        out_vars=("eta",),
    )

    final_eta = np.asarray(out["eta"][-1]).mean(axis=0)
    measured = (grid.x_c[np.argmax(final_eta)] - x0) / out["time"][-1]
    theory = np.sqrt(params.g * params.H)
    assert np.isclose(measured, theory, rtol=0.035)


def test_variable_depth_case_stays_finite():
    grid = make_grid(48, 12, 480_000.0, 120_000.0)
    depth = shelf_bathymetry(
        grid,
        H_deep=800.0,
        H_coast=100.0,
        shelf_width=180_000.0,
        coast="east",
    )
    params = ModelParams(H=depth, f0=0.0, beta=0.0, r=0.0, linear=True)
    dt = compute_dt_cfl(grid, params, cfl=0.35)

    out = run_model(
        tmax=20 * dt,
        dt=dt,
        grid=grid,
        params=params,
        forcing_fn=zero_forcing,
        ic_fn=lambda g, p: setup_initial_state(
            g, p, mode="gaussian_bump", amp=0.05, R=40_000.0, x0=0.25 * g.Lx
        ),
        save_every=5,
    )

    for name in ("eta", "u", "v"):
        values = np.asarray(out[name])
        assert np.isfinite(values).all()


def test_existing_wind_forcing_short_run():
    grid = make_grid(24, 16, 240_000.0, 160_000.0)
    params = ModelParams(H=200.0, f0=1e-4, beta=0.0, r=1 / 86_400.0, linear=True)
    dt = compute_dt_cfl(grid, params, cfl=0.35)

    out = run_model(
        tmax=10 * dt,
        dt=dt,
        grid=grid,
        params=params,
        forcing_fn=wind_gyre_forcing,
        ic_fn=lambda g, p: setup_initial_state(g, p, mode="rest"),
        save_every=5,
    )

    assert np.isfinite(np.asarray(out["eta"])).all()
    assert np.max(np.abs(out["u"][-1])) > 0.0



def test_closed_basin_curl_uses_collocated_corner_derivatives():
    """A discrete streamfunction mode has a known corner-Laplacian curl."""
    from shallowwater import operators
    from shallowwater.dynamics import _curl_on_center_local
    grid = make_grid(32, 24, 800e3, 600e3)
    X, Y = np.meshgrid(grid.x_u, grid.y_v)
    psi = 1e5*np.sin(np.pi*X/grid.Lx)*np.sin(2*np.pi*Y/grid.Ly)
    u = -np.diff(psi, axis=0)/grid.dy
    v = np.diff(psi, axis=1)/grid.dx
    eigenvalue = (-4*np.sin(np.pi/(2*grid.Nx))**2/grid.dx**2
                  -4*np.sin(np.pi/grid.Ny)**2/grid.dy**2)
    expected = .25*eigenvalue*(psi[:-1, :-1]+psi[1:, :-1]+psi[:-1, 1:]+psi[1:, 1:])
    np.testing.assert_allclose(operators.curl_on_center(u, v, grid.dx, grid.dy), expected, atol=1e-18)
    np.testing.assert_allclose(_curl_on_center_local(u, v, grid.dx, grid.dy), expected, atol=1e-18)


def test_beta_generates_the_correct_vorticity_tendency_in_both_models():
    """Northward flow on beta>0 must have zeta_t=-beta*v in the interior."""
    from shallowwater import operators, tendencies
    grid = make_grid(12, 10, 120e3, 100e3)
    state = dict(eta=np.zeros((10, 12)), u=np.zeros((10, 13)), v=np.ones((11, 12)))
    state['v'][[0, -1]] = 0.
    for linear in (True, False):
        params = ModelParams(H=200., f0=1e-4, beta=2e-11, y0=grid.Ly/2,
                             r=0., Ah=0., linear=linear)
        _, du, dv = tendencies(state, 0., grid, params, zero_forcing)
        curl_t = operators.curl_on_center(du, dv, grid.dx, grid.dy)
        np.testing.assert_allclose(curl_t[2:-2, 2:-2], -params.beta, atol=1e-23)
