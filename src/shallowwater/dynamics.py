import numpy as np
from .operators import avg_center_to_u, avg_center_to_v, grad_x_on_u, grad_y_on_v, v_on_u, u_on_v, divergence
from .operators import avg_u_to_center, avg_v_to_center, curl_on_center,laplacian_u, laplacian_v

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

    # Mass flux in continuity (you already have this):
    if params.linear:
        H_u = params.H
        H_v = params.H
    else:
        eta_u = avg_center_to_u(eta)
        eta_v = avg_center_to_v(eta)
        H_u = params.H + eta_u
        H_v = params.H + eta_v
        # clip to avoid vanishing/negative depth
        Hmin = getattr(params, "Hmin_frac", 0.0) * params.H
        if Hmin > 0.0:
            H_u = np.maximum(H_u, Hmin)
            H_v = np.maximum(H_v, Hmin)
    
    Fx = H_u * u            # (Ny, Nx+1)
    Fy = H_v * v            # (Ny+1, Nx)
    divF = divergence(Fx, Fy, grid.dx, grid.dy)
    deta_dt = -divF + Q_eta
    
    # Coriolis on native lines and cross-grid velocities
    f_u = coriolis_on_u(grid, params)
    f_v = coriolis_on_v(grid, params)
    v_u = v_on_u(state["v"])
    u_v = u_on_v(state["u"])
    
    eta_total = state["eta"] + (phi_eta / params.g if phi_eta is not None else 0.0)
    
    if params.linear:
        # gradients from centers to faces
        d_etadx_u = grad_x_on_u(eta_total, grid.dx)
        d_etady_v = grad_y_on_v(eta_total, grid.dy)
        denom_u = params.rho * params.H
        denom_v = params.rho * params.H
        du_dt = f_u * v_u - params.g * d_etadx_u + (taux_u / denom_u) - params.r * state["u"]
        dv_dt = -f_v * u_v - params.g * d_etady_v + (tauy_v / denom_v) - params.r * state["v"]
    else:
        # --- vector-invariant nonlinear momentum ---
        # velocities & vorticity at centers
        Uc = avg_u_to_center(state["u"])
        Vc = avg_v_to_center(state["v"])
        zeta_c = curl_on_center(state["u"], state["v"], grid.dx, grid.dy)
    
        # kinetic energy with optional safety cap to prevent runaway IG blow-up
        Ucap = float(getattr(params, "Ucap", 0.0))  # 0 disables capping
        if Ucap > 0.0:
            Uc2 = np.minimum(Uc * Uc, Ucap * Ucap)
            Vc2 = np.minimum(Vc * Vc, Ucap * Ucap)
            K_c = 0.5 * (Uc2 + Vc2)
        else:
            K_c = 0.5 * (Uc * Uc + Vc * Vc)
    
        # scalar potential S = g*eta_total + K at centers, then gradients to faces
        S_c = params.g * eta_total + K_c
        dSdx_u = grad_x_on_u(S_c, grid.dx)
        dSdy_v = grad_y_on_v(S_c, grid.dy)
    
        # absolute vorticity on faces
        qabs_u = avg_center_to_u(zeta_c) + f_u
        qabs_v = avg_center_to_v(zeta_c) + f_v
    
        # reuse clipped depths for stress terms
        denom_u = params.rho * H_u
        denom_v = params.rho * H_v
    
        du_dt = qabs_u * v_u - dSdx_u + (taux_u / denom_u) - params.r * state["u"]
        dv_dt = -qabs_v * u_v - dSdy_v + (tauy_v / denom_v) - params.r * state["v"]
    
    # optional lateral viscosity (guarded)
    Ah = float(getattr(params, "Ah", 0.0))
    if Ah > 0.0:
        du_dt += Ah * laplacian_u(state["u"], grid.dx, grid.dy)
        dv_dt += Ah * laplacian_v(state["v"], grid.dx, grid.dy)

    if hooks:
        add_eta = np.zeros_like(eta); add_u = np.zeros_like(u); add_v = np.zeros_like(v)
        for h in hooks:
            d_eta_h, d_u_h, d_v_h = h(state, t, grid, params)
            if d_eta_h is not None: add_eta += d_eta_h
            if d_u_h is not None: add_u += d_u_h
            if d_v_h is not None: add_v += d_v_h
        deta_dt += add_eta; du_dt += add_u; dv_dt += add_v

    return deta_dt, du_dt, dv_dt
