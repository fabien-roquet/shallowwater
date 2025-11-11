import numpy as np
from .operators import avg_center_to_u, avg_center_to_v, grad_x_on_u, grad_y_on_v, v_on_u, u_on_v, divergence

def enforce_bcs(u, v):
    u[:, 0] = 0.0
    u[:, -1] = 0.0
    v[0, :] = 0.0
    v[-1, :] = 0.0

def coriolis_on_u(grid, params):
    return params.f0 + params.beta * (grid.y_u[:, None] - params.y0)

def coriolis_on_v(grid, params):
    return params.f0 + params.beta * (grid.y_v[:, None] - params.y0)

def tendencies(state, t, grid, params, forcing_fn, hooks=None):
    eta = state["eta"]
    u = state["u"].copy()
    v = state["v"].copy()

    enforce_bcs(u, v)

    # --- Forcing unpack: allow (taux_u, tauy_v, Q_eta) or (taux_u, tauy_v, Q_eta, phi_eta) ---
    f_out = forcing_fn(t, grid, params)
    if isinstance(f_out, tuple) and len(f_out) == 4:
        taux_u, tauy_v, Q_eta, phi_eta = f_out
    else:
        taux_u, tauy_v, Q_eta = f_out
        phi_eta = None

    if Q_eta is None: Q_eta = np.zeros_like(eta)
    if taux_u is None: taux_u = np.zeros_like(u)
    if tauy_v is None: tauy_v = np.zeros_like(v)
    if phi_eta is None: phi_eta = np.zeros_like(eta)

    # ---- Pressure gradients include equilibrium tide: eta_total = eta + phi/g ----
    eta_total = eta + (phi_eta / params.g)

    d_etadx_u = grad_x_on_u(eta_total, grid.dx)
    d_etady_v = grad_y_on_v(eta_total, grid.dy)

    f_u = coriolis_on_u(grid, params)
    f_v = coriolis_on_v(grid, params)

    v_u = v_on_u(v)
    u_v = u_on_v(u)

    eta_u = avg_center_to_u(eta_total)
    eta_v = avg_center_to_v(eta_total)
    Fx = (params.H + (0.0 if params.linear else eta_u)) * u
    Fy = (params.H + (0.0 if params.linear else eta_v)) * v
    divF = divergence(Fx, Fy, grid.dx, grid.dy)
    deta_dt = -divF + Q_eta

    denom_u = params.rho * (params.H + (0.0 if params.linear else eta_u))
    denom_v = params.rho * (params.H + (0.0 if params.linear else eta_v))

    du_dt = f_u * v_u - params.g * d_etadx_u + taux_u / denom_u - params.r * u
    dv_dt = -f_v * u_v - params.g * d_etady_v + tauy_v / denom_v - params.r * v

    if hooks:
        add_eta = np.zeros_like(eta); add_u = np.zeros_like(u); add_v = np.zeros_like(v)
        for h in hooks:
            d_eta_h, d_u_h, d_v_h = h(state, t, grid, params)
            if d_eta_h is not None: add_eta += d_eta_h
            if d_u_h is not None: add_u += d_u_h
            if d_v_h is not None: add_v += d_v_h
        deta_dt += add_eta; du_dt += add_u; dv_dt += add_v

    return deta_dt, du_dt, dv_dt
