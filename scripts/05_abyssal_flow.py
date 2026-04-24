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


# Source notebook: 05_abyssal_flow.ipynb


# %% Cell 0
from shallowwater import (ModelParams, make_grid, setup_initial_state, 
                          stommel_arons_forcing, run_model, compute_dt_cfl)
import numpy as np
import matplotlib.pyplot as plt


# %% Cell 1
# --- Stommel–Arons source (NE) + uniform sink (zero net flux) ---

Nx, Ny = 128, 128
Lx, Ly = 2.0e6, 2.0e6
grid = make_grid(Nx, Ny, Lx, Ly)

# Beta-plane + weak Rayleigh to allow a western boundary current–like response
params = ModelParams(H=1000.0, g=9.81, rho=1025.0,
                     f0=1e-4, beta=2e-11, y0=Ly/2,
                     r=1/(50*86400),     # ~50-day e-fold; adjust as you like
                     linear=True)

dt = compute_dt_cfl(grid, params, cfl=0.5)
tmax = 30 * 86400.0      # ~30 days to see large-scale adjustment

# Start from rest
ic_fn = lambda g, p: setup_initial_state(g, p, mode='rest')

# Point source (NE), uniform sink; ramp in over 2 days to avoid a shock
forcing_fn = lambda t, g, p: stommel_arons_forcing(
    t, g, p,
    Q0=2e-8,       # ~2 cm/day vertical velocity at the source peak
    R=1.5e5,       # ~150 km source radius
    x0=0.95*Lx, y0=0.95*Ly,
    time_ramp=2*86400
)

# Save η,u,v for diagnostics/plots
out = run_model(tmax, dt, grid, params, forcing_fn, ic_fn,
                save_every=120, out_vars=("eta","u","v"))

print('Saved steps:', len(out['time']))


# %% Cell 2
from shallowwater.visualize import animate_eta

# mean-removed RdBu with contours
anim = animate_eta(out, grid, interval=120, contours=True, contour_levels=21,
                   title="Stommel–Arons response (η, demeaned)")


# %% Cell 3
# GIF (no external dependencies)
anim.save(str(ANIMATION_DIR / "eta_05.gif"), fps=10)

# MP4 (requires ffmpeg available on your PATH)
# anim.save(str(ANIMATION_DIR / "eta.mp4"), fps=20)


# %% Cell 4
from shallowwater.visualize import plot_forcings

fig_tx = plot_forcings(forcing_fn, t=0.0, grid=grid, params=params, what="Q",
                       title="Q only")


# %% Cell 5
from shallowwater.visualize import animate_eta_spectrum

spec = animate_eta_spectrum(out, grid, quadrant="ur", log10=True, interval=80,
                            title="η power spectrum (upper-right quadrant)")
# spec  # displays inline
# spec.save(str(ANIMATION_DIR / "eta_spectrum_ur.gif"), fps=20)


# %% Cell 6
from shallowwater.visualize import coast_hovmoller

fig = coast_hovmoller(out, grid, units_x="km",
                      title="η along coastline (W→N→E→S) with corners marked")
