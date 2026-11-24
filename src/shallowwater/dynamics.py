import numpy as np
from . import operators, channel
from .bathymetry import center_depth


def _curl_on_center_local(u: np.ndarray, v: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Use collocated corner derivatives for closed-basin nonlinear vorticity."""
    return operators.curl_on_center(u, v, dx, dy)


def enforce_bcs(u, v, periodic_x=False):
    if periodic_x:
        u[:, -1] = u[:, 0]
    else:
        u[:, 0] = 0.0
        u[:, -1] = 0.0
    v[0, :] = 0.0
    v[-1, :] = 0.0


def coriolis_on_u(grid, params):
    return params.f0 + params.beta * (grid.y_u[:, None] - params.y0)


def coriolis_on_v(grid, params):
    return params.f0 + params.beta * (grid.y_v[:, None] - params.y0)


def tendencies(state, t, grid, params, forcing_fn, hooks=None):
    # Nonlinear debug switches.  You can set these as attributes on params.
    nl_mass = getattr(params, "nl_mass", True)
    nl_H_denom = getattr(params, "nl_H_denom", True)

    eta = state["eta"]
    u = state["u"].copy()
    v = state["v"].copy()

    enforce_bcs(u, v, grid.periodic_x)
    op = channel if grid.periodic_x else operators

    f_out = forcing_fn(t, grid, params)
    if isinstance(f_out, tuple) and len(f_out) == 4:
        taux_u, tauy_v, Q_eta, phi_eta = f_out
    else:
        taux_u, tauy_v, Q_eta = f_out
        phi_eta = None

    eta_total = eta + phi_eta / params.g if phi_eta is not None else eta

    # Background depth on eta-, u-, and v-points. params.H can be scalar,
    # a length-Nx x-profile, or a full (Ny, Nx) array.
    Hc = center_depth(grid, params.H)
    H_u0 = op.avg_center_to_u(Hc)
    H_v0 = op.avg_center_to_v(Hc)

    eta_u = op.avg_center_to_u(eta)
    eta_v = op.avg_center_to_v(eta)
    H_u_full = H_u0 + eta_u
    H_v_full = H_v0 + eta_v

    Hmin_frac = float(getattr(params, "Hmin_frac", 0.0))
    if Hmin_frac > 0.0:
        H_u_full = np.maximum(H_u_full, Hmin_frac * H_u0)
        H_v_full = np.maximum(H_v_full, Hmin_frac * H_v0)

    if params.linear or not nl_mass:
        H_u_flux = H_u0
        H_v_flux = H_v0
    else:
        H_u_flux = H_u_full
        H_v_flux = H_v_full

    if params.linear or not nl_H_denom:
        H_u_mom = H_u0
        H_v_mom = H_v0
    else:
        H_u_mom = H_u_full
        H_v_mom = H_v_full

    # Continuity: eta_t = -div(H u, H v) + Q.
    Fx = H_u_flux * u
    Fy = H_v_flux * v
    deta_dt = -op.divergence(Fx, Fy, grid.dx, grid.dy) + Q_eta

    f_u = coriolis_on_u(grid, params)
    f_v = coriolis_on_v(grid, params)
    v_u = op.v_on_u(v)
    u_v = op.u_on_v(u)

    if params.linear or not params.momentum_advection:
        d_etadx_u = op.grad_x_on_u(eta_total, grid.dx)
        d_etady_v = op.grad_y_on_v(eta_total, grid.dy)
        denom_u = params.rho * H_u_mom
        denom_v = params.rho * H_v_mom
        du_dt = f_u * v_u - params.g * d_etadx_u + (taux_u / denom_u) - params.r * u
        dv_dt = -f_v * u_v - params.g * d_etady_v + (tauy_v / denom_v) - params.r * v
    else:
        Uc = op.avg_u_to_center(u)
        Vc = op.avg_v_to_center(v)
        zeta_c = (channel.curl_on_center(u, v, grid.dx, grid.dy) if grid.periodic_x
                  else _curl_on_center_local(u, v, grid.dx, grid.dy))
        K_c = 0.5 * (Uc * Uc + Vc * Vc)
        S_c = params.g * eta_total + K_c
        dSdx_u = op.grad_x_on_u(S_c, grid.dx)
        dSdy_v = op.grad_y_on_v(S_c, grid.dy)
        qabs_u = op.avg_center_to_u(zeta_c) + f_u
        qabs_v = op.avg_center_to_v(zeta_c) + f_v

        qmax = float(getattr(params, "qmax", 0.0))
        if qmax > 0.0:
            qabs_u = np.clip(qabs_u, -qmax, qmax)
            qabs_v = np.clip(qabs_v, -qmax, qmax)

        denom_u = params.rho * H_u_mom
        denom_v = params.rho * H_v_mom
        du_dt = qabs_u * v_u - dSdx_u + (taux_u / denom_u) - params.r * u
        dv_dt = -qabs_v * u_v - dSdy_v + (tauy_v / denom_v) - params.r * v

    Ah = float(getattr(params, "Ah", 0.0))
    if Ah > 0.0:
        du_dt += Ah * op.laplacian_u(u, grid.dx, grid.dy)
        dv_dt += Ah * op.laplacian_v(v, grid.dx, grid.dy)

    if hooks:
        add_eta = np.zeros_like(eta)
        add_u = np.zeros_like(u)
        add_v = np.zeros_like(v)
        for h in hooks:
            d_eta_h, d_u_h, d_v_h = h(state, t, grid, params)
            if d_eta_h is not None:
                add_eta += d_eta_h
            if d_u_h is not None:
                add_u += d_u_h
            if d_v_h is not None:
                add_v += d_v_h
        deta_dt += add_eta
        du_dt += add_u
        dv_dt += add_v

    enforce_bcs(du_dt, dv_dt, grid.periodic_x)
    return deta_dt, du_dt, dv_dt
