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
    eta_u = avg_center_to_u(eta)
    eta_v = avg_center_to_v(eta)
    Fx = (params.H + (0.0 if params.linear else eta_u)) * u
    Fy = (params.H + (0.0 if params.linear else eta_v)) * v
    divF = divergence(Fx, Fy, grid.dx, grid.dy)
    deta_dt = -divF + Q_eta
    
    # Linear Coriolis pieces (still needed in nonlinear too)
    f_u = coriolis_on_u(grid, params)    # f on u-lines
    f_v = coriolis_on_v(grid, params)    # f on v-lines
    v_u = v_on_u(v)                      # v interpolated to u
    u_v = u_on_v(u)                      # u interpolated to v
    
    if params.linear:
        # --- your current linear momentum ---
        d_etadx_u = grad_x_on_u(eta_total, grid.dx)  # if you kept tidal phi in eta_total
        d_etady_v = grad_y_on_v(eta_total, grid.dy)
        denom_u = params.rho * (params.H + eta_u)
        denom_v = params.rho * (params.H + eta_v)
        du_dt = f_u*v_u - params.g*d_etadx_u + taux_u/denom_u - params.r*u
        dv_dt = -f_v*u_v - params.g*d_etady_v + tauy_v/denom_v - params.r*v
    
    else:
        # --- NEW: vector-invariant nonlinear momentum ---
        # 1) velocities & vorticity at centers (eta-grid)
        Uc = avg_u_to_center(u)          # e.g., 0.5*(u[:,1:]+u[:,:-1])
        Vc = avg_v_to_center(v)          # e.g., 0.5*(v[1:,:]+v[:-1,:])
        zeta_c = curl_on_center(u, v, grid.dx, grid.dy)   # (∂x v - ∂y u) on centers
    
        # 2) kinetic energy at centers
        K_c = 0.5*(Uc*Uc + Vc*Vc)
    
        # 3) S = g*eta_total + K  at centers, then gradients on u/v
        S_c = params.g*eta_total + K_c
        dSdx_u = grad_x_on_u(S_c, grid.dx)
        dSdy_v = grad_y_on_v(S_c, grid.dy)
    
        # 4) absolute vorticity (f+zeta) at u/v lines
        qabs_u = avg_center_to_u(zeta_c) + f_u
        qabs_v = avg_center_to_v(zeta_c) + f_v
    
        # 5) stresses with free-surface depth (H+eta) in denominator
        H_u = params.H + (0.0 if params.linear else eta_u)
        H_v = params.H + (0.0 if params.linear else eta_v)
        Hmin = params.Hmin_frac * params.H
        if Hmin > 0.0:
            H_u = np.maximum(H_u, Hmin)
            H_v = np.maximum(H_v, Hmin)        
        denom_u = params.rho * H_u
        denom_v = params.rho * H_v

    
        # 6) final tendencies
        du_dt = qabs_u * v_u - dSdx_u + (taux_u/denom_u) - params.r*u
        dv_dt = -qabs_v * u_v - dSdy_v + (tauy_v/denom_v) - params.r*v
    
    if params.Ah > 0:
        du_dt += params.Ah * laplacian_u(u, grid.dx, grid.dy)
        dv_dt += params.Ah * laplacian_v(v, grid.dx, grid.dy)

    if hooks:
        add_eta = np.zeros_like(eta); add_u = np.zeros_like(u); add_v = np.zeros_like(v)
        for h in hooks:
            d_eta_h, d_u_h, d_v_h = h(state, t, grid, params)
            if d_eta_h is not None: add_eta += d_eta_h
            if d_u_h is not None: add_u += d_u_h
            if d_v_h is not None: add_v += d_v_h
        deta_dt += add_eta; du_dt += add_u; dv_dt += add_v

    return deta_dt, du_dt, dv_dt
