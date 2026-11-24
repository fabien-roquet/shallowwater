#!/usr/bin/env python3
"""Rossby-wave dispersion from balanced eddies above and below the deformation scale.

Run from the repository root or from the scripts/ directory. Generated movies
are written to animations/, which is intentionally ignored by git.
Use --case tiny for the resolved Ld/10 Gaussian and short-wave-filtered cases.
"""
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
ANIMATION_DIR = PROJECT_ROOT / "animations"
ANIMATION_DIR.mkdir(exist_ok=True)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib
matplotlib.use("Agg")


# Source notebook: 11_Rossby_wave_propagation.ipynb


# %% Cell 0
from shallowwater import (ModelParams, make_grid, setup_initial_state,
                          zero_forcing, run_model, compute_dt_cfl)
# If you exported it at top-level:
from shallowwater import geostrophic_velocities_from_eta
# Otherwise (as you mentioned) it lives in initial.py:
# from shallowwater.initial import geostrophic_velocities_from_eta

import numpy as np
from _rossby_plotting import render_eta_animation


# %% Cell 1
# Initial condition: nearly geostrophic Gaussian at mid-latitude, east of center
def ic_rossby_bump(g, p,
                   amp=0.15,                   # meters
                   R=None,                     # set to Ld by default
                   x0=None, y0=None,
                   degree_of_balance=1,     # 1 = fully geo, <1 leaves a touch of IG
                   alpha=0.95, sponge=6):
    if R  is None: R  = np.sqrt(p.g*p.H)/p.f0
    if x0 is None: x0 = 0.5*g.Lx             # start at the centre to watch westward propagation
    if y0 is None: y0 = p.y0                  # mid-lat line (f ≈ f0)

    # eta field
    eta0, _, _ = setup_initial_state(g, p, mode="gaussian_bump", amp=amp, R=R, x0=x0, y0=y0)
    eta0 -= eta0.mean()
    # geostrophic (u,v) from eta, scaled
    u0, v0 = geostrophic_velocities_from_eta(eta0, g, p,
                                             degree_of_balance=degree_of_balance,
                                             alpha=alpha, sponge=sponge)
    return eta0, u0, v0


def run_existing_cases():
    import numpy as np
    # ### Eddy with radius = Ld


    # %% Cell 3
    # --- Mid-latitude Rossby wave from a balanced Gaussian bump ---

    # Basin & grid
    Nx, Ny = 192, 128
    Lx, Ly = 4.0e6, 3.0e6     # 4000 x 3000 km
    grid = make_grid(Nx, Ny, Lx, Ly)

    # Mid-latitude β-plane, mild Rayleigh (damp IG waves, keep Rossby clean), linear dynamics
    params = ModelParams(H=100.0, g=9.81, rho=1025.0,
                         f0=1.0e-4, beta=2.0e-11, y0=Ly/2,
                         r=1/(15*86400),  # ~15-day e-fold
                         linear=True)

    # Useful scales
    c0   = np.sqrt(params.g*params.H)              # external gravity-wave speed [m/s]
    Ld   = c0/params.f0                            # barotropic deformation radius [m]
    cR   = -params.beta * Ld**2                    # long barotropic Rossby phase speed [m/s] (westward)
    print(f"Ld ≈ {Ld/1e3:.0f} km,  c_R ≈ {cR:.2f} m/s ({cR*86400/1e3:.1f} km/day)")

    # Stable time step
    dt = compute_dt_cfl(grid, params, cfl=0.5)

    # Initial condition: nearly geostrophic Gaussian at mid-latitude, east of center
    ic_fn = lambda g, p: ic_rossby_bump(g, p, R=Ld)

    # No forcing
    forcing_fn = lambda t, g, p: zero_forcing(t, g, p)

    # Long enough to cross a good fraction of the basin at |cR|
    tmax = 20 * 86400.0    # 20 days

    # Save ~200 frames
    N = int(np.ceil(tmax/dt))
    save_every = max(1, int(round(N/199)))

    out = run_model(tmax, dt, grid, params, forcing_fn, ic_fn,
                    save_every=save_every, out_vars=("eta","u","v"))
    print("Saved steps:", len(out["time"]))


    # %% Cell 4
    import matplotlib.pyplot as plt

    render_eta_animation(out, grid, ANIMATION_DIR, 'medium_eddy',
                         label='Medium Gaussian eddy', radius=1.0*Ld, radius_label='Ld')
    np.savez_compressed(ANIMATION_DIR / "experiment_11_medium_eddy.npz",
                        eta=np.asarray(out["eta"]), time=np.asarray(out["time"]),
                        x=grid.x_c, y=grid.y_c)
    plt.close("all")


    # %% Cell 5
    import numpy as np
    import matplotlib.pyplot as plt

    # Stack and extract a mid-latitude Hovmöller
    ETA  = np.stack(out["eta"])                 # (T, Ny, Nx)
    tt   = np.array(out["time"])                # s
    jj   = np.argmin(np.abs(grid.y_c - params.y0))  # closest to mid-lat
    hov  = ETA[:, jj, :]                        # η(x,t) at y = y0

    # Axes in convenient units: x in 10^6 m (= 1000 km), time in days
    Xc_Mm  = grid.x_c / 1e6                     # centers along x (millions of meters)
    T_days = tt / 86400.0

    # Fixed display scale and 0.01 m contour spacing across experiment 11.
    vmin, vmax = -.08, .08
    levels = np.linspace(vmin, vmax, 17)

    plt.figure(figsize=(8, 4))

    # Image background (use extent to align with physical axes)
    ext = (Xc_Mm[0], Xc_Mm[-1], T_days[0], T_days[-1])
    plt.imshow(hov, aspect="auto", origin="lower", extent=ext,
               cmap="RdBu_r", vmin=vmin, vmax=vmax)

    # Contours over the same field in physical coordinates
    XX, TT = np.meshgrid(Xc_Mm, T_days)
    plt.contour(XX, TT, hov, levels=levels, colors="k", linewidths=0.6, alpha=0.8)

    plt.colorbar(label="η [m]")
    plt.xlabel("x [1000 km]")    # 1 on the x-axis = 1000 km
    plt.ylabel("time [days]")
    plt.xlim([Xc_Mm[0], Xc_Mm[-1]])
    plt.title("Hovmöller at mid-latitude (y ≈ y0)")

    # Overlay a long-wave Rossby ray: x(t) = x0 + cR * t
    x0_units = 0.50 * grid.Lx / 1e6               # same units as x-axis (10^6 m)
    t_ray    = np.linspace(T_days[0], T_days[-1], 200)
    x_ray    = x0_units + (cR * (t_ray * 86400.0)) / 1e6
    plt.plot(x_ray, t_ray, "k--", lw=1.5, label=f"c_R ≈ {cR*86400/1e3:.1f} km/day")
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


    # ### Large Eddy with radius = 3*Ld


    # %% Cell 8
    # --- Mid-latitude Rossby wave from a balanced Gaussian bump ---

    # Basin & grid
    Nx, Ny = 192, 128
    Lx, Ly = 4.0e6, 3.0e6     # 4000 x 3000 km
    grid = make_grid(Nx, Ny, Lx, Ly)

    # Mid-latitude β-plane, mild Rayleigh (damp IG waves, keep Rossby clean), linear dynamics
    params = ModelParams(H=100.0, g=9.81, rho=1025.0,
                         f0=1.0e-4, beta=2.0e-11, y0=Ly/2,
                         r=1/(15*86400),  # ~15-day e-fold
                         linear=True)

    # Useful scales
    c0   = np.sqrt(params.g*params.H)              # external gravity-wave speed [m/s]
    Ld   = c0/params.f0                            # barotropic deformation radius [m]
    cR   = -params.beta * Ld**2                    # long barotropic Rossby phase speed [m/s] (westward)
    print(f"Ld ≈ {Ld/1e3:.0f} km,  c_R ≈ {cR:.2f} m/s ({cR*86400/1e3:.1f} km/day)")

    # Stable time step
    dt = compute_dt_cfl(grid, params, cfl=0.5)

    # Initial condition: nearly geostrophic Gaussian at mid-latitude, east of center
    ic_fn = lambda g, p: ic_rossby_bump(g, p, R=3*Ld)

    # No forcing
    forcing_fn = lambda t, g, p: zero_forcing(t, g, p)

    # Long enough to cross a good fraction of the basin at |cR|
    tmax = 20 * 86400.0    # 20 days

    # Save ~200 frames
    N = int(np.ceil(tmax/dt))
    save_every = max(1, int(round(N/199)))

    out = run_model(tmax, dt, grid, params, forcing_fn, ic_fn,
                    save_every=save_every, out_vars=("eta","u","v"))
    print("Saved steps:", len(out["time"]))


    # %% Cell 9

    render_eta_animation(out, grid, ANIMATION_DIR, 'large_eddy',
                         label='Large Gaussian eddy', radius=3.0*Ld, radius_label='3 Ld')
    np.savez_compressed(ANIMATION_DIR / "experiment_11_large_eddy.npz",
                        eta=np.asarray(out["eta"]), time=np.asarray(out["time"]),
                        x=grid.x_c, y=grid.y_c)
    plt.close("all")


    # %% Cell 10
    import numpy as np
    import matplotlib.pyplot as plt

    # Stack and extract a mid-latitude Hovmöller
    ETA  = np.stack(out["eta"])                 # (T, Ny, Nx)
    tt   = np.array(out["time"])                # s
    jj   = np.argmin(np.abs(grid.y_c - params.y0))  # closest to mid-lat
    hov  = ETA[:, jj, :]                        # η(x,t) at y = y0

    # Axes in convenient units: x in 10^6 m (= 1000 km), time in days
    Xc_Mm  = grid.x_c / 1e6                     # centers along x (millions of meters)
    T_days = tt / 86400.0

    # Fixed display scale and 0.01 m contour spacing across experiment 11.
    vmin, vmax = -.08, .08
    levels = np.linspace(vmin, vmax, 17)

    plt.figure(figsize=(8, 4))

    # Image background (use extent to align with physical axes)
    ext = (Xc_Mm[0], Xc_Mm[-1], T_days[0], T_days[-1])
    plt.imshow(hov, aspect="auto", origin="lower", extent=ext,
               cmap="RdBu_r", vmin=vmin, vmax=vmax)

    # Contours over the same field in physical coordinates
    XX, TT = np.meshgrid(Xc_Mm, T_days)
    plt.contour(XX, TT, hov, levels=levels, colors="k", linewidths=0.6, alpha=0.8)

    plt.colorbar(label="η [m]")
    plt.xlabel("x [1000 km]")    # 1 on the x-axis = 1000 km
    plt.ylabel("time [days]")
    plt.xlim([Xc_Mm[0], Xc_Mm[-1]])
    plt.title("Hovmöller at mid-latitude (y ≈ y0)")

    # Overlay a long-wave Rossby ray: x(t) = x0 + cR * t
    x0_units = 0.50 * grid.Lx / 1e6               # same units as x-axis (10^6 m)
    t_ray    = np.linspace(T_days[0], T_days[-1], 200)
    x_ray    = x0_units + (cR * (t_ray * 86400.0)) / 1e6
    plt.plot(x_ray, t_ray, "k--", lw=1.5, label=f"c_R ≈ {cR*86400/1e3:.1f} km/day")
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


    # ### Small Eddy with radius = Ld/3


    # %% Cell 13
    # --- Mid-latitude Rossby wave from a balanced Gaussian bump ---

    # Basin & grid
    Nx, Ny = 192, 128
    Lx, Ly = 4.0e6, 3.0e6     # 4000 x 3000 km
    grid = make_grid(Nx, Ny, Lx, Ly)

    # Mid-latitude β-plane, mild Rayleigh (damp IG waves, keep Rossby clean), linear dynamics
    params = ModelParams(H=100.0, g=9.81, rho=1025.0,
                         f0=1.0e-4, beta=2.0e-11, y0=Ly/2,
                         r=1/(15*86400),  # ~15-day e-fold
                         linear=True)

    # Useful scales
    c0   = np.sqrt(params.g*params.H)              # external gravity-wave speed [m/s]
    Ld   = c0/params.f0                            # barotropic deformation radius [m]
    cR   = -params.beta * Ld**2                    # long barotropic Rossby phase speed [m/s] (westward)
    print(f"Ld ≈ {Ld/1e3:.0f} km,  c_R ≈ {cR:.2f} m/s ({cR*86400/1e3:.1f} km/day)")

    # Stable time step
    dt = compute_dt_cfl(grid, params, cfl=0.5)

    # Initial condition: nearly geostrophic Gaussian at mid-latitude, east of center
    ic_fn = lambda g, p: ic_rossby_bump(g, p, R=Ld/3)

    # No forcing
    forcing_fn = lambda t, g, p: zero_forcing(t, g, p)

    # Long enough to cross a good fraction of the basin at |cR|
    tmax = 20 * 86400.0    # 20 days

    # Save ~200 frames
    N = int(np.ceil(tmax/dt))
    save_every = max(1, int(round(N/199)))

    out = run_model(tmax, dt, grid, params, forcing_fn, ic_fn,
                    save_every=save_every, out_vars=("eta","u","v"))
    print("Saved steps:", len(out["time"]))


    # %% Cell 14

    render_eta_animation(out, grid, ANIMATION_DIR, 'small_eddy',
                         label='Small Gaussian eddy', radius=0.3333333333333333*Ld, radius_label='Ld/3')
    np.savez_compressed(ANIMATION_DIR / "experiment_11_small_eddy.npz",
                        eta=np.asarray(out["eta"]), time=np.asarray(out["time"]),
                        x=grid.x_c, y=grid.y_c)
    plt.close("all")


    # %% Cell 15
    import numpy as np
    import matplotlib.pyplot as plt

    # Stack and extract a mid-latitude Hovmöller
    ETA  = np.stack(out["eta"])                 # (T, Ny, Nx)
    tt   = np.array(out["time"])                # s
    jj   = np.argmin(np.abs(grid.y_c - params.y0))  # closest to mid-lat
    hov  = ETA[:, jj, :]                        # η(x,t) at y = y0

    # Axes in convenient units: x in 10^6 m (= 1000 km), time in days
    Xc_Mm  = grid.x_c / 1e6                     # centers along x (millions of meters)
    T_days = tt / 86400.0

    # Fixed display scale and 0.01 m contour spacing across experiment 11.
    vmin, vmax = -.08, .08
    levels = np.linspace(vmin, vmax, 17)

    plt.figure(figsize=(8, 4))

    # Image background (use extent to align with physical axes)
    ext = (Xc_Mm[0], Xc_Mm[-1], T_days[0], T_days[-1])
    plt.imshow(hov, aspect="auto", origin="lower", extent=ext,
               cmap="RdBu_r", vmin=vmin, vmax=vmax)

    # Contours over the same field in physical coordinates
    XX, TT = np.meshgrid(Xc_Mm, T_days)
    plt.contour(XX, TT, hov, levels=levels, colors="k", linewidths=0.6, alpha=0.8)

    plt.colorbar(label="η [m]")
    plt.xlabel("x [1000 km]")    # 1 on the x-axis = 1000 km
    plt.ylabel("time [days]")
    plt.xlim([Xc_Mm[0], Xc_Mm[-1]])
    plt.title("Hovmöller at mid-latitude (y ≈ y0)")

    # Overlay a long-wave Rossby ray: x(t) = x0 + cR * t
    x0_units = 0.50 * grid.Lx / 1e6               # same units as x-axis (10^6 m)
    t_ray    = np.linspace(T_days[0], T_days[-1], 200)
    x_ray    = x0_units + (cR * (t_ray * 86400.0)) / 1e6
    plt.plot(x_ray, t_ray, "k--", lw=1.5, label=f"c_R ≈ {cR*86400/1e3:.1f} km/day")
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.show()



def replot_existing_cases(folder):
    """Redraw the three original cases from their saved numerical histories."""
    ld = np.sqrt(9.81*100.)/1e-4
    for stem, label, factor, radius_label in (
            ('large_eddy', 'Large Gaussian eddy', 3., '3 Ld'),
            ('medium_eddy', 'Medium Gaussian eddy', 1., 'Ld'),
            ('small_eddy', 'Small Gaussian eddy', 1/3, 'Ld/3')):
        with np.load(folder/f'experiment_11_{stem}.npz', allow_pickle=False) as saved:
            data = {key: saved[key] for key in saved.files}
        x, y = data['x'], data['y']
        grid = make_grid(len(x), len(y), len(x)*(x[1]-x[0]), len(y)*(y[1]-y[0]))
        print(f'11 {stem}: redraw saved history', flush=True)
        render_eta_animation(data, grid, folder, stem, label=label,
                             radius=factor*ld, radius_label=radius_label)


def make_parser():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('all', 'existing', 'tiny'), default='all')
    parser.add_argument('--variant', choices=('both', 'gaussian', 'short-waves'), default='both')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--days', type=float, default=60.)
    parser.add_argument('--frames', type=int, default=121)
    parser.add_argument('--output-dir', type=Path, default=ANIMATION_DIR)
    parser.add_argument('--replot', action='store_true')
    return parser


if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    if args.days <= 0 or args.frames < 2:
        parser.error('Use positive days and at least two frames.')
    if args.case in ('all', 'existing'):
        if args.replot:
            replot_existing_cases(args.output_dir)
        else:
            run_existing_cases()
    if args.case in ('all', 'tiny'):
        from _rossby_small_scales import run_small_scales
        run_small_scales(args)
