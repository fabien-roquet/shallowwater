# shallowwater

A minimal shallow-water equations solver on an **Arakawa C-grid** over a (f, β)-plane,
with **Rayleigh friction** and **SSP-RK3** time stepping.

- Flat bottom, square basin, closed walls (no-normal-flow).
- Linear or Momentum advection (vector-invariant form).
- Staggering: `η[j,i]  →  (Ny, Nx)`; `u[j,i]  →  (Ny, Nx+1)`; `v[j,i]  →  (Ny+1, Nx)`.
- Forcing at the surface via wind stress `(τx on u, τy on v)`, mass source `Q` (at η),
  and optional **tidal geopotential** `φ` (at η) such that momentum sees `η_total = η + φ/g`.

---

## Install (editable)

```bash
cd shallowwater
python -m venv .venv && source .venv/bin/activate  # or use conda/uv/mamba
pip install -e .
````

## Run the example notebook

Launch Jupyter from the same environment:

```bash
jupyter notebook notebooks/01_wind_gyre.ipynb
```

## Batch-running notebooks

The repo includes a helper script `run_notebooks.sh` to execute notebooks **in place** (outputs are written back into the `.ipynb` files using `jupyter nbconvert`).

Make it executable:

```bash
chmod +x run_notebooks.sh
```

Run a **single** notebook:

```bash
./run_notebooks.sh -n notebooks/01_wind_gyre.ipynb
```

Run **all** notebooks in the current directory:

```bash
./run_notebooks.sh -a
```

Common options:

* `-t SECONDS` — cell timeout (default: `3600`)
* `-k NAME` — kernel name (e.g. `python3`)
* `-x GLOB` — exclude pattern (repeatable), e.g. `-x "*WIP*.ipynb"`
* `-c` — continue on errors (maps to `--allow-errors`)

---

## What’s inside (model at a glance)

**Equations (linearized, flat bottom)**
At η-points (centers):

* Continuity:  (\eta_t = -\nabla\cdot \mathbf{F} + Q), with (\mathbf{F} = (H u, H v)).

At u/v points (faces/edges):

* Zonal momentum:    (u_t = +f v - g,\partial_x(\eta+\phi/g) + \tau_x/(\rho H) - r,u).
* Meridional momentum: (v_t = -f u - g,\partial_y(\eta+\phi/g) + \tau_y/(\rho H) - r,v).

Optionally, momentum advection and laplacian viscosity can be added.

Here (f=f_0+\beta (y-y_0)). `φ` is the **surface geopotential perturbation** (e.g., for tides or pressure loading).
When `φ` is omitted, the code assumes `φ=0`.

**Time stepping**

* Strong-stability-preserving **RK3**; use `compute_dt_cfl(...)` for a safe CFL step based on (c=\sqrt{gH}).

**Boundary conditions**

* Rectangular closed basin with **no-normal-flow** on walls (C-grid-friendly masking/ghosting).

---

## Source layout

```
src/
  shallowwater/
    __init__.py            # exports core API
    initial.py             # initial states (e.g., gaussian bump); 
                           # also: geostrophic_velocities_from_eta(η, grid, params, ...)
    forcing.py             # wind/pressure/mass forcings (wind gyre, tides via φ, storms, etc.)
    visualize.py           # animate_eta, coast_hovmoller, plot_forcings, animate_eta_spectrum
    ...                    # (operators/dynamics/integrator are used internally)
```

Top-level:

```
pyproject.toml
README.md
run_notebooks.sh
notebooks/   # study cases (see list below)
tests/
```

---

## Public API (Python)

Shapes use the C-grid staggering noted above.

* `ModelParams(H, g, rho, f0, beta, y0, r, linear=True)`
* `make_grid(Nx, Ny, Lx, Ly)`
  Returns a grid object with `dx, dy, Lx, Ly` and C-grid coordinates (`x_c, y_c, x_u, y_u, x_v, y_v`).
* `compute_dt_cfl(grid, params, cfl=0.5)`
* `setup_initial_state(grid, params, mode="rest" | "gaussian_bump", **kwargs) -> (eta, u, v)`
* `geostrophic_velocities_from_eta(eta, grid, params, *, degree_of_balance=1.0, alpha=1.0, sponge=0, fmin=1e-6) -> (u, v)`
  Compute (u, v) that are geostrophic w.r.t. a given `η`. Useful to start from a balanced (or partially balanced) state.
* **Forcings** (return conventions):

  * **3-tuple** `(taux_u, tauy_v, Q_eta)` — wind stress on `u/v`, mass source on `η`.
  * **4-tuple** `(taux_u, tauy_v, Q_eta, phi_eta)` — same, plus geopotential `φ` at `η` (for tides/pressure).
    Provided helpers include:
  * `zero_forcing(...)`
  * `wind_gyre_forcing(t, grid, params, tau0=...)`
  * `tidal_potential_forcing(t, grid, params, amp_eta_eq=..., omega=..., kx=..., ky=...)` → via `φ`
  * `stommel_arons_forcing(t, grid, params, Q0=..., R=..., time_ramp=...)`
  * `storm_surge_forcing(t, grid, params, Vmax=..., delta_p=..., ...)` → wind + inverse barometer (`φ`)
  * `coastal_alongshore_wind_forcing(t, grid, params, coast=..., direction=..., ...)`
* `tendencies(state, t, grid, params, forcing_fn, hooks=None) -> (deta_dt, du_dt, dv_dt)`
  (internally uses either 3- or 4-tuple forcing; `φ` is optional).
* `run_model(tmax, dt, grid, params, forcing_fn, ic_fn, save_every=10, out_vars=('eta','u','v'), hooks=None)`
* **Visualization** (`shallowwater.visualize`):

  * `animate_eta(out, grid, remove_mean=True, cmap="RdBu_r", contours=False, frames=None, ...)`
  * `coast_hovmoller(out, grid, units_x="km")`
  * `plot_forcings(forcing_fn, t, grid, params, what="all"|"wind"|["taux","|tau|","Q"], ...)`
  * `animate_eta_spectrum(out, grid, quadrant="full"|"ur", log10=True, ...)`

---

## Typical workflow

```python
from shallowwater import (ModelParams, make_grid, setup_initial_state,
                          compute_dt_cfl, run_model, zero_forcing)

Nx, Ny = 128, 128
Lx, Ly = 2.0e6, 2.0e6
grid = make_grid(Nx, Ny, Lx, Ly)
params = ModelParams(H=1000.0, g=9.81, rho=1025.0, f0=1e-4, beta=2e-11, y0=Ly/2, r=1/(10*86400), linear=True)

dt = compute_dt_cfl(grid, params, cfl=0.5)
ic_fn = lambda g, p: setup_initial_state(g, p, mode="gaussian_bump", amp=0.1, R=2e5)
forcing_fn = lambda t, g, p: zero_forcing(t, g, p)

out = run_model(tmax=5*86400, dt=dt, grid=grid, params=params,
                forcing_fn=forcing_fn, ic_fn=ic_fn,
                save_every=24, out_vars=("eta",))
```

To start **near geostrophic balance**:

```python
from shallowwater.initial import geostrophic_velocities_from_eta
def ic_balanced(g, p):
    eta, _, _ = setup_initial_state(g, p, mode="gaussian_bump", amp=0.5, R=2e5)
    u, v = geostrophic_velocities_from_eta(eta, g, p, degree_of_balance=0.9, alpha=0.95, sponge=6)
    return eta, u, v
```

---

## Study cases (notebooks)

All study cases live in `notebooks/` (animated previews `eta_*.gif` are included):

1. **01_wind_gyre.ipynb** — classic β-plane wind-driven gyre (Sverdrup interior + western boundary current).
2. **02_gravity_waves.ipynb** — linear gravity/Poincaré waves from a perturbed surface.
3. **03_tsunami.ipynb** — unforced propagation from a localized uplift (deep water; no rotation/friction).
4. **04_tides.ipynb** — equilibrium-tide forcing via variable geopotential `φ` (M2-like).
5. **05_abyssal_flow.ipynb** — Stommel–Arons-like source (NE) + uniform sink (zero net `Q`) on a β-plane.
6. **06_seiche.ipynb** — standing basin modes (m,n) with no forcing (clean f-plane seiche).
7. **07_equatorial_waves.ipynb** — equatorial Kelvin/Rossby packets (β-plane, equator at mid-domain).
8. **08_storm_surge.ipynb** — moving cyclone: wind + inverse barometer (`φ`) over a shallow shelf.
9. **09_wind_driven_kelvin_wave.ipynb** — alongshore wind band builds coastal setup; after shutoff a **coastal Kelvin wave** propagates.
10. **10_geostrophic_adjustment.ipynb** — Gaussian dome adjusts on an f-plane; start from rest or from a **partially geostrophic** state using `geostrophic_velocities_from_eta`.
11. **11_Rossby_wave_propagation.ipynb** — Gaussian dome adjusts on an beta-plane; start from **geostrophic** state using `geostrophic_velocities_from_eta`.

---

## Notes & conventions

* **Units**: SI throughout. Typical `H` in meters, `τ` in N m⁻², `Q` in m s⁻¹, `φ` in m² s⁻², space in meters, time in seconds.
* **C-grid shapes**:

  * `eta: (Ny, Nx)` at cell centers `(x_c, y_c)`
  * `u: (Ny, Nx+1)` staggered in x at `(x_u, y_u)`
  * `v: (Ny+1, Nx)` staggered in y at `(x_v, y_v)`
* **Forcing return**: 3-tuple `(τx_u, τy_v, Q)` **or** 4-tuple `(τx_u, τy_v, Q, φ)`. If you don’t need `φ`, omit it.
* **CFL**: external celerity (c=\sqrt{gH}). `compute_dt_cfl` picks a safe step using min(dx,dy).

---

## Testing

A `tests/` folder is provided as a placeholder; feel free to contribute simple regression checks
(e.g., energy decay with Rayleigh friction, tide period checks, or Kelvin phase speed comparisons).

---

## License

MIT.


