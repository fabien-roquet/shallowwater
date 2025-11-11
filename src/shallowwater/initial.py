import numpy as np

def setup_initial_state(grid, params, mode="rest", **kwargs):

    eta = np.zeros((grid.Ny, grid.Nx))
    u = np.zeros((grid.Ny, grid.Nx + 1))
    v = np.zeros((grid.Ny + 1, grid.Nx))

    if mode == "rest":
        return eta, u, v

    if mode == "gaussian_bump":
        amp = float(kwargs.get("amp", 0.01))
        R = float(kwargs.get("R", min(grid.Lx, grid.Ly) / 10.0))
        x0 = float(kwargs.get("x0", grid.Lx / 2.0))
        y0 = float(kwargs.get("y0", grid.Ly / 2.0))
        X, Y = np.meshgrid(grid.x_c, grid.y_c)
        eta = amp * np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (R ** 2))
        return eta, u, v

    raise ValueError(f"Unknown mode: {mode}")


# --- Geostrophic initialization utilities -----------------------------------
def geostrophic_velocities_from_eta(
    eta: np.ndarray,
    grid,
    params,
    *,
    degree_of_balance: float = 1.0,   # 1 = fully geostrophic, 0 = no initial flow
    alpha: float = 1.0,               # gentle scale-back (e.g., 0.9–0.95) to reduce IG noise
    sponge: int = 0,                  # cosine taper (cells) near each boundary
    fmin: float = 1e-6                # avoid division by ~0 if beta-plane near equator
):
    """
    Compute geostrophic (u,v) from a given eta field on an Arakawa C-grid.

      u_g = -(g/f) * ∂η/∂y   on the u-grid (Ny, Nx+1)
      v_g = +(g/f) * ∂η/∂x   on the v-grid (Ny+1, Nx)

    If params.beta != 0, uses local f(y) on the corresponding staggered lines.
    """
    Ny, Nx = eta.shape
    dx, dy = float(grid.dx), float(grid.dy)

    # --- ∂η/∂y on u-grid -----------------------------------------------------
    # gradient at v-lines (Ny+1, Nx)
    deta_dy_v = np.zeros((Ny + 1, Nx))
    deta_dy_v[1:Ny, :] = (eta[1:, :] - eta[:-1, :]) / dy
    deta_dy_v[0,  :] = deta_dy_v[1, :]
    deta_dy_v[-1, :] = deta_dy_v[-2, :]

    # average to centers (Ny, Nx), then to u-faces (Ny, Nx+1)
    deta_dy_c = 0.5 * (deta_dy_v[1:, :] + deta_dy_v[:-1, :])
    deta_dy_u = np.zeros((Ny, Nx + 1))
    if Nx > 1:
        deta_dy_u[:, 1:Nx] = 0.5 * (deta_dy_c[:, 0:Nx-1] + deta_dy_c[:, 1:Nx])
        deta_dy_u[:, 0]  = deta_dy_c[:, 0]
        deta_dy_u[:, -1] = deta_dy_c[:, -1]
    else:
        deta_dy_u[:, :] = deta_dy_c  # degenerate case

    # --- ∂η/∂x on v-grid -----------------------------------------------------
    # gradient at u-lines (Ny, Nx+1)
    deta_dx_u = np.zeros((Ny, Nx + 1))
    deta_dx_u[:, 1:Nx] = (eta[:, 1:] - eta[:, :-1]) / dx
    deta_dx_u[:, 0]  = deta_dx_u[:, 1]
    deta_dx_u[:, -1] = deta_dx_u[:, -2]

    # average to centers (Ny, Nx), then to v-edges (Ny+1, Nx)
    deta_dx_c = 0.5 * (deta_dx_u[:, 0:Nx] + deta_dx_u[:, 1:Nx+1])
    deta_dx_v = np.zeros((Ny + 1, Nx))
    if Ny > 1:
        deta_dx_v[1:Ny, :] = 0.5 * (deta_dx_c[0:Ny-1, :] + deta_dx_c[1:Ny, :])
        deta_dx_v[0,  :] = deta_dx_v[1, :]
        deta_dx_v[-1, :] = deta_dx_v[-2, :]
    else:
        deta_dx_v[:, :] = deta_dx_c  # degenerate case

    # --- f on the staggered lines -------------------------------------------
    # u-grid uses y_u (Ny,), v-grid uses y_v (Ny+1,)
    if getattr(params, "beta", 0.0) == 0.0:
        f_u = np.full((Ny, 1), float(params.f0))
        f_v = np.full((Ny + 1, 1), float(params.f0))
    else:
        f_u_vals = params.f0 + params.beta * (grid.y_u - params.y0)  # (Ny,)
        f_v_vals = params.f0 + params.beta * (grid.y_v - params.y0)  # (Ny+1,)
        f_u = f_u_vals[:, None]
        f_v = f_v_vals[:, None]

    # clip small |f| to avoid blow-ups (esp. near equator)
    f_u = np.where(np.abs(f_u) < fmin, np.sign(f_u) * fmin + (f_u == 0) * fmin, f_u)
    f_v = np.where(np.abs(f_v) < fmin, np.sign(f_v) * fmin + (f_v == 0) * fmin, f_v)

    # --- Geostrophic velocities on native grids ------------------------------
    u_g = -(params.g / f_u) * deta_dy_u   # (Ny, Nx+1)
    v_g = +(params.g / f_v) * deta_dx_v   # (Ny+1, Nx)

    # --- Optional cosine taper near boundaries -------------------------------
    def cosine_taper_1d(n, s):
        w = np.ones(n)
        if s > 0:
            i = np.arange(s)
            ramp = 0.5 * (1 - np.cos(np.pi * (i + 1) / (s + 1)))  # 0→1
            w[:s] = ramp
            w[-s:] = ramp[::-1]
        return w

    if sponge and sponge > 0:
        wy_u = cosine_taper_1d(Ny, sponge)[:, None]
        wx_u = cosine_taper_1d(Nx + 1, sponge)[None, :]
        wy_v = cosine_taper_1d(Ny + 1, sponge)[:, None]
        wx_v = cosine_taper_1d(Nx, sponge)[None, :]
        u_g = (wy_u * wx_u) * u_g
        v_g = (wy_v * wx_v) * v_g

    # scale as requested
    scale = float(alpha) * float(degree_of_balance)
    u_g *= scale
    v_g *= scale
    return u_g, v_g


