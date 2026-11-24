"""Physics and boundary regressions for the periodic-channel extension."""
import numpy as np

from shallowwater import ModelParams, make_grid, run_model, zero_forcing, tendencies
from shallowwater import channel
from shallowwater.bathymetry import depth_on_u
from shallowwater.dynamics import enforce_bcs


def test_periodic_face_operators_use_neighbors_across_seam():
    c = np.arange(24, dtype=float).reshape(4, 6)
    avg = channel.avg_center_to_u(c)
    grad = channel.grad_x_on_u(c, 2.0)
    np.testing.assert_allclose(avg[:, 0], 0.5*(c[:, -1]+c[:, 0]))
    np.testing.assert_allclose(grad[:, 0], (c[:, 0]-c[:, -1])/2.0)
    np.testing.assert_array_equal(avg[:, 0], avg[:, -1])
    np.testing.assert_array_equal(grad[:, 0], grad[:, -1])
    grid = make_grid(6, 4, 60., 40., periodic_x=True)
    np.testing.assert_allclose(depth_on_u(grid, c+1), channel.avg_center_to_u(c+1))


def test_periodic_viscosity_has_correct_fourier_eigenvalue():
    n = 32
    x = 2*np.pi*np.arange(n)/n
    u = np.broadcast_to(np.cos(np.r_[x, 2*np.pi]), (5, n+1)).copy()
    v = np.broadcast_to(np.cos(x), (6, n)).copy()
    eigenvalue = -4*np.sin(np.pi/n)**2
    np.testing.assert_allclose(channel.laplacian_u(u, 1., 1.), eigenvalue*u, atol=1e-14)
    np.testing.assert_allclose(channel.laplacian_v(v, 1., 1.), eigenvalue*v, atol=1e-14)


def test_channel_curl_of_zonal_shear_and_periodic_wave():
    grid = make_grid(64, 16, 2*np.pi, 2.)
    u = np.broadcast_to(0.3*grid.y_u[:, None], (16, 65)).copy()
    v = np.broadcast_to(np.sin(grid.x_v)[None, :], (17, 64)).copy()
    curl = channel.curl_on_center(u, v, grid.dx, grid.dy)
    expected = np.broadcast_to(np.cos(grid.x_c)*np.sin(grid.dx)/grid.dx-0.3, (14, 64))
    np.testing.assert_allclose(curl[1:-1], expected, atol=1e-14)


def test_nonlinear_channel_conserves_mass_and_stores_boundary_conditions():
    grid = make_grid(12, 8, 120e3, 80e3, periodic_x=True)
    rng = np.random.default_rng(21)
    initial = (rng.normal(0, 0.01, (8, 12)), rng.normal(0, 0.01, (8, 13)),
               rng.normal(0, 0.01, (9, 12)))
    params = ModelParams(H=np.broadcast_to(np.linspace(100, 200, 12), (8, 12)), linear=False)
    out = run_model(100., 5., grid, params, zero_forcing, lambda g, p: initial, save_every=1)
    np.testing.assert_allclose(np.sum(out["eta"], axis=(1, 2)), np.sum(initial[0]), atol=1e-12)
    for u, v in zip(out["u"], out["v"]):
        np.testing.assert_array_equal(u[:, 0], u[:, -1])
        assert not np.any(v[[0, -1]])


def test_gravity_wave_propagates_through_periodic_seam():
    grid = make_grid(80, 4, 800e3, 40e3, periodic_x=True)
    params = ModelParams(H=100., g=10., f0=0., beta=0., linear=True)
    c = np.sqrt(params.g*params.H)
    k = 2*np.pi/grid.Lx
    eta = np.broadcast_to(0.01*np.sin(k*grid.x_c), (4, 80)).copy()
    u = np.broadcast_to(0.01*c/params.H*np.sin(k*grid.x_u), (4, 81)).copy()
    v = np.zeros((5, 80))
    duration = 1.25*grid.Lx/c
    out = run_model(duration, duration/1000, grid, params, zero_forcing,
                    lambda g, p: (eta, u, v), save_every=1000)
    expected = 0.01*np.sin(k*(grid.x_c-c*out["time"][-1]))
    np.testing.assert_allclose(out["eta"][-1], np.broadcast_to(expected, eta.shape), atol=3e-5)


def test_advection_switch_only_changes_momentum_advection():
    grid = make_grid(12, 8, 120e3, 80e3, periodic_x=True)
    rng = np.random.default_rng(9)
    state = dict(eta=rng.normal(0, 1, (8, 12)), u=rng.normal(0, 1, (8, 13)),
                 v=rng.normal(0, 1, (9, 12)))
    enforce_bcs(state["u"], state["v"], True)
    params = ModelParams(H=100., linear=False, momentum_advection=False)
    off = tendencies(state, 0., grid, params, zero_forcing)
    params.momentum_advection = True
    on = tendencies(state, 0., grid, params, zero_forcing)
    np.testing.assert_array_equal(on[0], off[0])
    zeta = channel.curl_on_center(state["u"], state["v"], grid.dx, grid.dy)
    K = 0.5*(channel.avg_u_to_center(state["u"])**2 + channel.avg_v_to_center(state["v"])**2)
    du = channel.avg_center_to_u(zeta)*channel.v_on_u(state["v"]) - channel.grad_x_on_u(K, grid.dx)
    dv = -channel.avg_center_to_v(zeta)*channel.u_on_v(state["u"]) - channel.grad_y_on_v(K, grid.dy)
    enforce_bcs(du, dv, True)
    np.testing.assert_allclose(on[1]-off[1], du, atol=1e-16)
    np.testing.assert_allclose(on[2]-off[2], dv, atol=1e-16)


def test_periodic_wind_and_balanced_initialization_have_no_seam_jump():
    from shallowwater import center_wind_to_staggered, geostrophic_velocities_from_eta
    grid = make_grid(40, 16, 400e3, 160e3, periodic_x=True)
    params = ModelParams(beta=0.)
    X, Y = np.meshgrid(grid.x_c, grid.y_c)
    eta = 0.01*np.sin(2*np.pi*X/grid.Lx)*np.sin(np.pi*Y/grid.Ly)
    u, v = geostrophic_velocities_from_eta(eta, grid, params, alpha=1., sponge=2)
    np.testing.assert_array_equal(u[:, 0], u[:, -1])
    stress = np.broadcast_to(np.arange(40), (16, 40)).copy()
    tx, ty = center_wind_to_staggered(stress, np.zeros_like(stress), grid)
    np.testing.assert_allclose(tx[:, 0], 19.5)
    np.testing.assert_array_equal(tx[:, 0], tx[:, -1])


def test_unseeded_forced_channel_preserves_zonal_symmetry():
    from shallowwater import make_wind_forcing, setup_initial_state
    grid = make_grid(16, 12, 160e3, 120e3, periodic_x=True)
    params = ModelParams(H=200., linear=False, Ah=100.)
    tx = np.broadcast_to(0.05/np.cosh((grid.y_c[:, None]-grid.Ly/2)/20e3)**2, (12, 16))
    forcing = make_wind_forcing(tx, np.zeros_like(tx), grid)
    out = run_model(3600., 20., grid, params, forcing,
                    lambda g, p: setup_initial_state(g, p), save_every=30)
    for key in ('eta', 'u', 'v'):
        fields = np.asarray(out[key])
        np.testing.assert_allclose(fields, np.broadcast_to(fields[:, :, :1], fields.shape), atol=1e-16)


def test_viscosity_matches_copy_edge_stencil_on_rectangular_grid():
    from shallowwater import operators
    rng = np.random.default_rng(15)
    for function, shape in ((operators.laplacian_u, (4, 8)), (operators.laplacian_v, (5, 7))):
        field = rng.normal(size=shape)
        expected = np.zeros_like(field)
        for j in range(shape[0]):
            for i in range(shape[1]):
                expected[j, i] = (field[j, min(i+1, shape[1]-1)]-2*field[j, i]
                                  +field[j, max(i-1, 0)])/4.
                expected[j, i] += (field[min(j+1, shape[0]-1), i]-2*field[j, i]
                                   +field[max(j-1, 0), i])/9.
        np.testing.assert_allclose(function(field, 2., 3.), expected, atol=1e-14)
