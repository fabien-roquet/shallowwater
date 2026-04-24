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


# Source notebook: 03_tsunami_shoaling_sponge.ipynb


# # Tsunami shoaling with variable depth and a sponge layer
#
# This notebook is a kid-friendly tsunami experiment. The model has a deep ocean on the west side and a shallow coast on the east side. A broad sea-surface bump travels toward the coast. As the water gets shallower, the wave slows down and its height grows.
#
# The sponge layer near the west boundary absorbs the wave travelling away from the coast, which reduces reflections.
#
# This is a demonstration of **shoaling**, not a real tsunami warning or inundation model.


# %% Cell 1
import numpy as np
import matplotlib.pyplot as plt

from shallowwater import (
    ModelParams, make_grid, zero_forcing, setup_initial_state,
    compute_dt_cfl, run_model, animate_eta,
    shelf_bathymetry, wave_speed, make_sponge_hook, sponge_mask_eta,
)


# ## 1. Build an ocean with a shallow coast
#
# The coast is on the right/east side. The water is deep on the left/west side and becomes shallow near the coast.


# %% Cell 3
Nx, Ny = 160, 36
Lx, Ly = 2_400e3, 500e3
grid = make_grid(Nx, Ny, Lx, Ly)

H_deep = 4000.0      # deep ocean depth [m]
H_coast = 80.0       # shallow coastal depth [m]
shelf_width = 900e3  # width of the sloping shelf [m]

H = shelf_bathymetry(
    grid, H_deep=H_deep, H_coast=H_coast,
    shelf_width=shelf_width, coast="east", power=1.4,
)

params = ModelParams(H=H, f0=0.0, beta=0.0, r=0.0, linear=True)


# %% Cell 4
fig, ax1 = plt.subplots(figsize=(8, 3))
x_km = grid.x_c / 1000
ax1.plot(x_km, H[Ny//2, :])
ax1.invert_yaxis()
ax1.set_xlabel("distance east [km]")
ax1.set_ylabel("depth H [m]")
ax1.set_title("Ocean bottom: deep ocean to shallow coast")
ax1.grid(True)
plt.close(fig)

fig, ax2 = plt.subplots(figsize=(8, 3))
c = wave_speed(grid, params)
ax2.plot(x_km, c[Ny//2, :])
ax2.set_xlabel("distance east [km]")
ax2.set_ylabel("wave speed sqrt(gH) [m/s]")
ax2.set_title("Long waves are fastest in deep water")
ax2.grid(True)
plt.close(fig)


# ## 2. Initial tsunami bump
#
# The initial bump is a simplified earthquake displacement of the sea surface. It starts in deep water and travels both east and west. The east-going wave heads toward the coast.


# %% Cell 6
def tsunami_initial_condition(grid, params):
    return setup_initial_state(
        grid, params, mode="gaussian_bump",
        amp=0.20,          # initial height [m]
        R=120e3,           # bump radius [m]
        x0=450e3,          # start in the deep ocean
        y0=0.5 * grid.Ly,
    )

eta0, u0, v0 = tsunami_initial_condition(grid, params)

plt.figure(figsize=(8, 3))
plt.pcolormesh(grid.x_c/1000, grid.y_c/1000, eta0, shading="auto")
plt.colorbar(label="eta [m]")
plt.xlabel("x [km]")
plt.ylabel("y [km]")
plt.title("Initial sea-surface displacement")
plt.close("all")


# ## 3. Add a sponge layer
#
# The sponge layer is strongest at the west boundary and fades into the interior. It damps waves so they do not reflect as strongly from that boundary.


# %% Cell 8
sponge_width = 300e3
sponge = make_sponge_hook(width=sponge_width, tau=900.0, sides=("west",), power=2.0)
mask = sponge_mask_eta(grid, width=sponge_width, sides=("west",), power=2.0)

plt.figure(figsize=(8, 2.5))
plt.pcolormesh(grid.x_c/1000, grid.y_c/1000, mask, shading="auto", vmin=0, vmax=1)
plt.colorbar(label="sponge strength")
plt.xlabel("x [km]")
plt.ylabel("y [km]")
plt.title("Western sponge layer")
plt.close("all")


# ## 4. Run the model
#
# The time step is chosen from the fastest wave speed, which occurs in the deepest water.


# %% Cell 10
dt = compute_dt_cfl(grid, params, cfl=0.45)
tmax = 4.0 * 3600.0
save_every = max(1, int((5 * 60) / dt))  # save about every 5 minutes

print(f"dt = {dt:.1f} s, save_every = {save_every} steps")

out = run_model(
    tmax, dt, grid, params, zero_forcing, tsunami_initial_condition,
    save_every=save_every, hooks=[sponge], show_progress=True,
)


# ## 5. Watch the wave
#
# Run the animation cell. Look for the wave slowing and growing near the shallow coast.


# %% Cell 12
anim = animate_eta(
    out, grid, interval=120, title="Tsunami shoaling experiment",
    contours=True, show_colorbar=True, remove_mean=False,
)
anim.save(str(ANIMATION_DIR / "eta_03_tsunami_shoaling_sponge.gif"), fps=10)


# ## 6. Measure how the wave grows
#
# Here we compute the largest absolute surface height seen at each x-position during the simulation.


# %% Cell 14
eta_stack = np.stack(out["eta"], axis=0)
max_abs_eta_x = np.max(np.abs(eta_stack), axis=(0, 1))

fig, ax1 = plt.subplots(figsize=(8, 3.5))
ax1.plot(x_km, max_abs_eta_x, label="maximum wave height")
ax1.set_xlabel("distance east [km]")
ax1.set_ylabel("max |eta| [m]")
ax1.grid(True)
ax1.legend(loc="upper left")

ax2 = ax1.twinx()
ax2.plot(x_km, H[Ny//2, :], linestyle="--", label="depth")
ax2.invert_yaxis()
ax2.set_ylabel("depth H [m]")
ax2.legend(loc="upper right")
plt.title("Wave height grows where the ocean gets shallow")
plt.close("all")


# ## 7. Try experiments
#
# Change one thing at a time and run the notebook again.
#
# Questions to test:
#
# 1. What happens if `H_coast` is 300 m instead of 80 m?
# 2. What happens if `H_coast` is 30 m?
# 3. What happens if `shelf_width` is larger or smaller?
# 4. What happens if you remove the sponge hook?
# 5. Why is this still not a real flooding model?
