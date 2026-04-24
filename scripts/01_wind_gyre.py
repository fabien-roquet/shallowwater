#!/usr/bin/env python3
"""Converted tutorial script.

Run from the repository root or from the scripts/ directory. Generated movies
are written to animations/, which is intentionally ignored by git.
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


# Source notebook: 01_wind_gyre.ipynb


# %% Cell 0
#os.environ["SHALLOWWATER_USE_NUMBA"] = "0"  # if you want to disable numba
from shallowwater import (ModelParams, make_grid, setup_initial_state,
                          wind_gyre_forcing,run_model, compute_dt_cfl)
import numpy as np
import matplotlib.pyplot as plt


# %% Cell 1
# tiny arrays just to trigger compilation
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def deco(f): return f
        return deco
    def prange(*args):
        return range(*args)

USE_NUMBA = os.getenv("SHALLOWWATER_USE_NUMBA", "0") == "1"

if USE_NUMBA and NUMBA_AVAILABLE:
    import numpy as np
    from shallowwater.operators import avg_u_to_center, avg_v_to_center
    u = np.zeros((8, 9)); v = np.zeros((9, 8))
    _ = avg_u_to_center(u); _ = avg_v_to_center(v)
else:
    print('not using numba')


# %% Cell 2
import math

def save_every_for_target_frames(tmax, dt, n_frames, include_initial=True):
    """
    Choose a save_every stride to produce about n_frames snapshots over [0, tmax].
    """
    if n_frames <= 1:
        # only initial (or final) frame
        return max(1, math.ceil(tmax/dt))  # save at start; plot just one frame
    N = math.ceil(tmax / dt)
    intervals = (n_frames - 1) if include_initial else n_frames
    return max(1, int(round(N / intervals)))


# ### Linear case


# %% Cell 4
# Grid & params
Nx, Ny = 128, 128
Lx, Ly = 2.0e6, 2.0e6
grid = make_grid(Nx, Ny, Lx, Ly)

params = ModelParams(
    H=1000.0, g=9.81, rho=1025.0,
    f0=1e-4, beta=2e-11, y0=Ly/2,
    r=1/(12*86400),    # a bit more damping than linear case
    linear=True,       # <— enable nonlinear terms
    Ah=1000.0,          # (optional) lateral viscosity if needed
)

dt = compute_dt_cfl(grid, params, cfl=0.5)
tmax = 10 * 86400.0
save_every = save_every_for_target_frames(tmax, dt, n_frames=120)

ic_fn = lambda g, p: setup_initial_state(g, p, mode="rest")
forcing_fn = lambda t, g, p: wind_gyre_forcing(t, g, p, tau0=0.1)
 
out = run_model(tmax, dt, grid, params, forcing_fn, ic_fn,
                save_every=save_every, out_vars=("eta","u","v"),
                show_progress=True,
                progress_kwargs={"desc": "Linear wind gyre", "unit": "step"}
               )

print('Saved steps:', len(out['time']))


# %% Cell 5
# Quick look animation (if you added visualize.py earlier)
from shallowwater.visualize import animate_eta

anim = animate_eta(out, grid, interval=120, title="η evolution")

# GIF (no external dependencies)
anim.save(str(ANIMATION_DIR / "eta_01.gif"), fps=10)


# %% Cell 6
from shallowwater.visualize import plot_forcings

fig_tx = plot_forcings(forcing_fn, t=0.0, grid=grid, params=params, what="taux",
                       title="τx only")


# %% Cell 7
from shallowwater.visualize import coast_hovmoller

fig = coast_hovmoller(out, grid, units_x="km",
                      title="η along coastline (W→N→E→S) with corners marked")


# ### nonlinear case (momentum advection + lateral viscosity)


# %% Cell 10
# Grid & params
Nx, Ny = 128, 128
Lx, Ly = 2.0e6, 2.0e6
grid = make_grid(Nx, Ny, Lx, Ly)

params = ModelParams(
    H=1000.0, g=9.81, rho=1025.0,
    f0=1e-4, beta=2e-11, y0=Ly/2,
    r=1/(12*86400),
    linear=False,
    Ah=1000.0,
    Hmin_frac=0.02,
    qmax=5e-4,
)

# Use a stricter but reasonable wave CFL for nonlinear runs
dt = compute_dt_cfl(grid, params, cfl=0.25)
tmax = 10*86400.0
save_every = save_every_for_target_frames(tmax, dt, n_frames=120)

ic_fn = lambda g, p: setup_initial_state(g, p, mode="rest")

def wind_gyre_forcing_ramped(t, g, p, tau0=0.1, tramp=3*86400):
    s = 1.0 if t >= tramp else (t / float(tramp))
    taux_u, tauy_v, Q = wind_gyre_forcing(t, g, p, tau0=tau0)
    return s*taux_u, s*tauy_v, s*Q

forcing_fn = lambda t, g, p: wind_gyre_forcing_ramped(t, g, p, tau0=0.1, tramp=3*86400)


# %% Cell 11
# Quick look animation (if you added visualize.py earlier)
from shallowwater.visualize import animate_eta

anim = animate_eta(out, grid, interval=120, title="η evolution")

# GIF (no external dependencies)
anim.save(str(ANIMATION_DIR / "eta_01b_nonlinear.gif"), fps=10)
