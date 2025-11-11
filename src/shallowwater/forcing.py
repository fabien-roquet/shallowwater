# forcing.py (inside shallowwater)

import numpy as np

def zero_forcing(t, grid, params, **kwargs):
    taux = np.zeros((grid.Ny, grid.Nx + 1))
    tauy = np.zeros((grid.Ny + 1, grid.Nx))
    Q = np.zeros((grid.Ny, grid.Nx))
    return taux, tauy, Q

def wind_gyre_forcing(t, grid, params, tau0=0.1, **kwargs):
    y = grid.y_u
    profile = np.sin(np.pi * y / grid.Ly)
    taux = tau0 * profile[:, None] * np.ones((grid.Ny, grid.Nx + 1))
    tauy = np.zeros((grid.Ny + 1, grid.Nx))
    Q = np.zeros((grid.Ny, grid.Nx))
    return taux, tauy, Q
    
def tidal_potential_forcing(t, grid, params, *,
                            amp_eta_eq=0.2,
                            omega=None,
                            phase=0.0,
                            kx=0.0, ky=0.0):
    """
    Tidal equilibrium forcing via variable geopotential Φ = g * η_eq.

    Parameters
    ----------
    amp_eta_eq : float
        Amplitude of equilibrium tide height [m] at η-points.
    omega : float or None
        Angular frequency [rad/s]. If None, defaults to M2 (~12.42 h period).
    phase : float
        Phase offset [rad].
    kx, ky : float
        Plane-wave wavenumbers [rad/m] for a traveling equilibrium tide (defaults 0 => spatially uniform).

    Returns
    -------
    (taux_u, tauy_v, Q_eta, phi_eta)
        Stresses and surface mass source are zero; phi_eta = g * η_eq drives pressure gradients.
    """
    if omega is None:
        T_M2 = 12.42 * 3600.0
        omega = 2.0 * np.pi / T_M2

    Xc, Yc = np.meshgrid(grid.x_c, grid.y_c)  # η-points
    eta_eq = amp_eta_eq * np.cos(kx * Xc + ky * Yc + omega * t + phase)
    phi_eta = params.g * eta_eq

    taux_u = np.zeros((grid.Ny, grid.Nx + 1))
    tauy_v = np.zeros((grid.Ny + 1, grid.Nx))
    Q_eta  = np.zeros((grid.Ny, grid.Nx))
    return taux_u, tauy_v, Q_eta, phi_eta


def stommel_arons_forcing(t, grid, params, *,
                          Q0=2e-8,          # peak vertical velocity at the source [m/s]
                          R=1.5e5,          # e-folding radius of Gaussian source [m]
                          x0=None, y0=None, # source location (defaults to NE corner region)
                          time_ramp=None    # seconds; if set, ramps from 0→1 over this time
                          ):
    """
    Stommel–Arons-like source/sink:
      - Localized Gaussian *source* near NE corner
      - Uniform *sink* over the whole domain
      - Net Q over the domain is exactly zero at all times

    Parameters
    ----------
    Q0 : float
        Peak vertical velocity at the Gaussian center [m s^-1].
    R : float
        Gaussian e-folding radius [m].
    x0, y0 : float or None
        Source center [m]. Defaults: x0=0.85*Lx, y0=0.85*Ly (NE quadrant).
    time_ramp : float or None
        If provided, multiplies Q by min(1, t/time_ramp) to smoothly switch on the forcing.

    Returns
    -------
    (taux_u, tauy_v, Q_eta)
        τ fields are zero; Q_eta has zero spatial mean.
    """
    if x0 is None: x0 = 0.85 * grid.Lx
    if y0 is None: y0 = 0.85 * grid.Ly

    # Gaussian source at eta points
    Xc, Yc = np.meshgrid(grid.x_c, grid.y_c)
    G = np.exp(-((Xc - x0)**2 + (Yc - y0)**2) / (R**2))
    Q_source = Q0 * G

    # Uniform sink added so that ⟨Q⟩ = 0 exactly
    sink = -Q_source.mean()
    Q_eta = Q_source + sink

    # Optional smooth ramp-in
    if time_ramp is not None and time_ramp > 0.0:
        alpha = min(1.0, float(t) / float(time_ramp))
        Q_eta = alpha * Q_eta

    taux_u = np.zeros((grid.Ny, grid.Nx + 1))
    tauy_v = np.zeros((grid.Ny + 1, grid.Nx))
    return taux_u, tauy_v, Q_eta

# --- Storm surge forcing: moving pressure low + cyclonic winds (Rankine-like) ---
def storm_surge_forcing(t, grid, params, *,
                        center0=(0.30, 0.20),       # start as fractions of (Lx, Ly)
                        velocity=(5.0, 2.0),        # storm translation [m/s] (east, north)
                        Vmax=35.0,                  # max 10 m wind [m/s]
                        Rw=8.0e4,                   # wind-core radius [m]
                        delta_p=-4.0e3,             # surface pressure anomaly amplitude [Pa] (~ -40 hPa)
                        Rp=1.2e5,                   # pressure-core radius [m]
                        use_wind=True,
                        use_pressure=True,
                        Cd=1.5e-3,                  # drag coefficient
                        rho_air=1.225):             # air density [kg/m^3]
    """
    Returns (taux_u, tauy_v, Q_eta, phi_eta) at time t.

    - Pressure anomaly p'(x,y,t) = delta_p * exp(-r^2/Rp^2)
      enters momentum via phi_eta = -(p'/rho_water), i.e. inverse barometer.
    - Wind field is a simple Rankine-like cyclone with Vtheta profile,
      τ = rho_air * Cd * |U10| * U10 at eta points, then averaged to u/v grids.
    - Q_eta = 0 (no mass source).

    Parameters are chosen to be in a realistic ballpark; tweak to taste.
    """
    # Center position (meters)
    xc = center0[0] * grid.Lx + velocity[0] * t
    yc = center0[1] * grid.Ly + velocity[1] * t

    # Eta-point coordinates
    Xc, Yc = np.meshgrid(grid.x_c, grid.y_c)
    dx = Xc - xc
    dy = Yc - yc
    r = np.hypot(dx, dy) + 1e-12

    # --- Wind (Rankine-like) ---
    if use_wind:
        # Piecewise Vtheta
        Vtheta = np.where(r < Rw, Vmax * (r / Rw), Vmax * (Rw / r))
        # Tangential unit vector (cyclonic: CCW)
        e_tx = -dy / r
        e_ty =  dx / r
        u10 = Vtheta * e_tx
        v10 = Vtheta * e_ty
        speed = np.hypot(u10, v10)
        taux_c = rho_air * Cd * speed * u10
        tauy_c = rho_air * Cd * speed * v10
    else:
        taux_c = np.zeros((grid.Ny, grid.Nx))
        tauy_c = np.zeros_like(taux_c)

    # --- Pressure anomaly (inverse barometer) ---
    if use_pressure:
        p_anom = delta_p * np.exp(-(r**2) / (Rp**2))
        # momentum uses eta_total = eta + phi/g. For inverse barometer, phi = -p'/rho_water
        phi_eta = -p_anom / params.rho
    else:
        phi_eta = np.zeros((grid.Ny, grid.Nx))

    # Map center (eta) stresses to u/v locations by simple face/edge averaging
    Ny, Nx = grid.Ny, grid.Nx
    taux_u = np.zeros((Ny, Nx + 1))
    tauy_v = np.zeros((Ny + 1, Nx))
    taux_u[:, 1:Nx] = 0.5 * (taux_c[:, :-1] + taux_c[:, 1:])
    taux_u[:, 0] = taux_c[:, 0]
    taux_u[:, -1] = taux_c[:, -1]
    tauy_v[1:Ny, :] = 0.5 * (tauy_c[:-1, :] + tauy_c[1:, :])
    tauy_v[0, :] = tauy_c[0, :]
    tauy_v[-1, :] = tauy_c[-1, :]

    Q_eta = np.zeros((grid.Ny, grid.Nx))
    return taux_u, tauy_v, Q_eta, phi_eta


# --- Alongshore coastal-band wind to build coastal setup / Kelvin wave ---
def coastal_alongshore_wind_forcing(
    t, grid, params, *,
    coast: str = "east",       # {"east","west","north","south"}
    direction: str = "north",  # alongshore wind direction
    tau0: float = 0.15,        # peak |tau| [N m^-2]
    Lw: float = 8.0e4,         # offshore e-folding [m]
    t_ramp: float = 6*3600.0,  # ramp-up time [s]
    t_off: float | None = 18*3600.0  # shutoff time [s] (None = keep steady)
):
    """
    Returns (taux_u, tauy_v, Q_eta). Q=0.

    Choose wind so Ekman transport is onshore to build coastal setup.
    NH rules of thumb:
      - East coast: direction="north" → onshore Ekman (to the right = east).
      - West coast: direction="south" → onshore Ekman (to the right = west).
      - North coast: direction="west"  → onshore Ekman (to the right = north).
      - South coast: direction="east"  → onshore Ekman (to the right = south).

    After shutoff (t > t_off), the setup releases as a coastal Kelvin wave.
    """
    import numpy as np

    # Time envelope
    if t_ramp is None or t_ramp <= 0:
        ramp_up = 1.0
    else:
        ramp_up = np.clip(t / t_ramp, 0.0, 1.0)

    if t_off is None or t <= t_off:
        env = ramp_up
    else:
        if t_ramp is None or t_ramp <= 0:
            env = 0.0
        else:
            xi = np.clip((t - t_off) / t_ramp, 0.0, 1.0)
            # smooth cosine taper 1→0
            env = 0.5 * (1.0 + np.cos(np.pi * xi))

    Ny, Nx = grid.Ny, grid.Nx
    taux_u = np.zeros((Ny, Nx + 1))
    tauy_v = np.zeros((Ny + 1, Nx))
    Q_eta  = np.zeros((Ny, Nx))

    # EAST/WEST coasts: alongshore stress is tau_y on v-points (Ny+1, Nx)
    if coast.lower() == "east":
        # distance offshore at v-points: d = Lx - x_v  (shape: (Nx,))
        d_x = (grid.Lx - grid.x_v)
        profile = np.exp(-d_x / Lw)                 # (Nx,)
        sgn = +1.0 if direction.lower() == "north" else -1.0
        tau_band = sgn * tau0 * env * np.repeat(profile[np.newaxis, :], Ny + 1, axis=0)  # (Ny+1, Nx)
        tauy_v[:, :] = tau_band

    elif coast.lower() == "west":
        d_x = grid.x_v                               # (Nx,)
        profile = np.exp(-d_x / Lw)                 # (Nx,)
        sgn = -1.0 if direction.lower() == "south" else +1.0
        tau_band = sgn * tau0 * env * np.repeat(profile[np.newaxis, :], Ny + 1, axis=0)  # (Ny+1, Nx)
        tauy_v[:, :] = tau_band

    # NORTH/SOUTH coasts: alongshore stress is tau_x on u-points (Ny, Nx+1)
    elif coast.lower() == "north":
        # distance offshore at u-points: d = Ly - y_u  (shape: (Ny,))
        d_y = (grid.Ly - grid.y_u)
        profile = np.exp(-d_y / Lw)                 # (Ny,)
        sgn = -1.0 if direction.lower() == "west" else +1.0
        tau_band = sgn * tau0 * env * np.repeat(profile[:, np.newaxis], Nx + 1, axis=1)  # (Ny, Nx+1)
        taux_u[:, :] = tau_band

    elif coast.lower() == "south":
        d_y = grid.y_u                               # (Ny,)
        profile = np.exp(-d_y / Lw)                 # (Ny,)
        sgn = +1.0 if direction.lower() == "east" else -1.0
        tau_band = sgn * tau0 * env * np.repeat(profile[:, np.newaxis], Nx + 1, axis=1)  # (Ny, Nx+1)
        taux_u[:, :] = tau_band

    else:
        raise ValueError("coast must be one of {'east','west','north','south'}")

    return taux_u, tauy_v, Q_eta
