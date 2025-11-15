import numpy as np
from .operators import (avg_center_to_u, avg_center_to_v,
                        grad_x_on_u, grad_y_on_v,
                        v_on_u, u_on_v, divergence,
                        avg_u_to_center, avg_v_to_center,
                        curl_on_center, laplacian_u, laplacian_v)

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

    # --- Nonlinear debug switches ----------------------------------------
    # You can set these from the notebook as attributes on params, e.g.
    # params.nl_mass = False
    # If the attribute does not exist, the default (True) is used.
    nl_mass       = getattr(params, "nl_mass", True)       # nonlinear continuity (H+η in flux)
    nl_H_denom    = getattr(params, "nl_H_denom", True)    # H+η in momentum denominators
    nl_vort_term  = getattr(params, "nl_vort_term", True)  # (ζ+f)×V term
    nl_ke_term    = getattr(params, "nl_ke_term", True)    # ∇K term

    eta = state["eta"]
    u = state["u"].copy()
    v = state["v"].copy()

    enforce_bcs(u, v)

    # --- Forcing: allow (taux_u, tauy_v, Q_eta) or (taux_u, tauy_v, Q_eta, phi_eta) ---
    f_out = forcing_fn(t, grid, params)
    if isinstance(f_out, tuple) and len(f_out) == 4:
        taux_u, tauy_v, Q_eta, phi_eta = f_out
    else:
        taux_u, tauy_v, Q_eta = f_out
        phi_eta = None

    # --- Total free surface (η + φ/g if provided) ---
    if phi_eta is not None:
        eta_total = eta + phi_eta / params.g
    else:
        eta_total = eta

    # --- Depth on u/v (for continuity and/or momentum) ---
    # First compute H+η on faces
    eta_u = avg_center_to_u(eta)
    eta_v = avg_center_to_v(eta)
    H_u_full = params.H + eta_u
    H_v_full = params.H + eta_v
    Hmin = getattr(params, "Hmin_frac", 0.0) * params.H
    if Hmin > 0.0:
        H_u_full = np.maximum(H_u_full, Hmin)
        H_v_full = np.maximum(H_v_full, Hmin)

    # Decide which depth to use where:
    if params.linear or not nl_mass:
        H_u_flux = params.H
        H_v_flux = params.H
    else:
        H_u_flux = H_u_full
        H_v_flux = H_v_full

    if params.linear or not nl_H_denom:
        H_u_mom = params.H
        H_v_mom = params.H
    else:
        H_u_mom = H_u_full
        H_v_mom = H_v_full

    # --- Continuity: ∂η/∂t = -∇⋅(H u, H v) + Q ---
    Fx = H_u_flux * u            # (Ny, Nx+1)
    Fy = H_v_flux * v            # (Ny+1, Nx)
    divF = divergence(Fx, Fy, grid.dx, grid.dy)
    deta_dt = -divF + Q_eta

    # --- Coriolis and cross-grid velocities ---
    f_u = coriolis_on_u(grid, params)
    f_v = coriolis_on_v(grid, params)
    v_u = v_on_u(v)                      # v interpolated to u
    u_v = u_on_v(u)                      # u interpolated to v

    if params.linear:
        # --- linear momentum ---
        d_etadx_u = grad_x_on_u(eta_total, grid.dx)
        d_etady_v = grad_y_on_v(eta_total, grid.dy)
        denom_u = params.rho * params.H
        denom_v = params.rho * params.H
        du_dt = f_u * v_u - params.g * d_etadx_u + (taux_u / denom_u) - params.r * u
        dv_dt = -f_v * u_v - params.g * d_etady_v + (tauy_v / denom_v) - params.r * v

    else:
        # ---- NONLINEAR: vector-invariant form ----

        # 1) velocities & vorticity at centers (η-grid)
        Uc = avg_u_to_center(u)          # (Ny, Nx)
        Vc = avg_v_to_center(v)          # (Ny, Nx)
        zeta_c = curl_on_center(u, v, grid.dx, grid.dy)   # (Ny, Nx)

        # 2) kinetic energy at centers
        K_c = 0.5 * (Uc * Uc + Vc * Vc)

        # 3) scalar potential S = g*η_total + K at centers, then gradients on u/v
        S_c = params.g * eta_total + K_c
        dSdx_u = grad_x_on_u(S_c, grid.dx)   # (Ny, Nx+1)
        dSdy_v = grad_y_on_v(S_c, grid.dy)   # (Ny+1, Nx)

        # 4) absolute vorticity (f+ζ) at u/v lines
        qabs_u = avg_center_to_u(zeta_c) + f_u   # (Ny, Nx+1)
        qabs_v = avg_center_to_v(zeta_c) + f_v   # (Ny+1, Nx)

        # --- optional safety limiter on |q| to avoid grid-scale blow-ups ---
        qmax = float(getattr(params, "qmax", 0.0))  # 0 => no limit
        if qmax > 0.0:
            qabs_u = np.clip(qabs_u, -qmax, qmax)
            qabs_v = np.clip(qabs_v, -qmax, qmax)

        # 5) stresses with free-surface depth (H+η) in denominator (clipped)
        eta_u = avg_center_to_u(eta)
        eta_v = avg_center_to_v(eta)
        H_u = params.H + eta_u
        H_v = params.H + eta_v
        Hmin = params.Hmin_frac * params.H
        if Hmin > 0.0:
            H_u = np.maximum(H_u, Hmin)
            H_v = np.maximum(H_v, Hmin)
        denom_u = params.rho * H_u
        denom_v = params.rho * H_v

        # 6) final nonlinear tendencies
        du_dt = qabs_u * v_u - dSdx_u + (taux_u / denom_u) - params.r * u
        dv_dt = -qabs_v * u_v - dSdy_v + (tauy_v / denom_v) - params.r * v

    # --- Lateral viscosity (optional) ---
    Ah = float(getattr(params, "Ah", 0.0))
    if Ah > 0.0:
        du_dt += Ah * laplacian_u(u, grid.dx, grid.dy)
        dv_dt += Ah * laplacian_v(v, grid.dx, grid.dy)

    # --- Hooks (optional) ---
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

    return deta_dt, du_dt, dv_dt





