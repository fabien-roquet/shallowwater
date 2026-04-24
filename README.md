# shallowwater

A minimal shallow-water equations solver on an **Arakawa C-grid** over a (f, β)-plane,
with **Rayleigh friction** and **SSP-RK3** time stepping.

- Scalar or variable bathymetry, rectangular basin, closed walls (no-normal-flow).
- Linear or Momentum advection (vector-invariant form).
- Staggering: `η[j,i]  →  (Ny, Nx)`; `u[j,i]  →  (Ny, Nx+1)`; `v[j,i]  →  (Ny+1, Nx)`.
- Forcing at the surface via wind stress `(τx on u, τy on v)`, mass source `Q` (at η),
  and optional **tidal geopotential** `φ` (at η) such that momentum sees `η_total = η + φ/g`.

---

## Install (from pyPI)

```bash
pip install shallowwater
````

## Install (editable)

```bash
git clone https://github.com/fabien-roquet/shallowwater
cd shallowwater
python -m venv .venv && source .venv/bin/activate  # or use conda/uv/mamba
pip install -e .
````

## Run a guided lab notebook

Launch Jupyter from the same environment:

```bash
jupyter notebook notebooks/01_waves_in_a_box.ipynb
```

The `notebooks/` folder now contains the guided lab notebooks only. Notebook outputs should not be committed; the included git hook strips outputs and execution counts before commits.

## Run the converted tutorial scripts

The original tutorial notebooks have been converted to Python scripts in `scripts/`. These scripts generate animations in `animations/`, which is ignored by git.

Make the runner executable:

```bash
chmod +x scripts/run_scripts.sh
```

Run one script:

```bash
./scripts/run_scripts.sh -s scripts/03_tsunami_shoaling_sponge.py
```

Run all converted tutorial scripts:

```bash
./scripts/run_scripts.sh -a
```

Common options:

* `-x GLOB` — exclude a script pattern, e.g. `-x "*Rossby*"`
* `-c` — continue on errors

Generated files are written to:

```text
animations/
```

That folder is intentionally not tracked by git.

## What’s inside (model at a glance)

**Equations (linearized form; scalar or variable bathymetry)**
At η-points (centers):

* Continuity:  (\eta_t = -\nabla\cdot \mathbf{F} + Q), with (\mathbf{F} = (H u, H v)).

At u/v points (faces/edges):

* Zonal momentum:    (u_t = +f v - g,\partial_x(\eta+\phi/g) + \tau_x/(\rho H) - r,u).
* Meridional momentum: (v_t = -f u - g,\partial_y(\eta+\phi/g) + \tau_y/(\rho H) - r,v).

Optionally, momentum advection and laplacian viscosity can be added.

Here (f=f_0+\beta (y-y_0)). `φ` is the **surface geopotential perturbation** (e.g., for tides or pressure loading).
When `φ` is omitted, the code assumes `φ=0`.

**Time stepping**

* Strong-stability-preserving **RK3**; use `compute_dt_cfl(...)` for a safe CFL step based on the fastest shallow-water wave speed, using max depth when `H` is spatially variable.

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
labs/        # written lab instructions
notebooks/   # guided lab notebooks, kept without outputs
scripts/     # converted tutorial scripts and helper scripts
animations/  # generated movies; ignored by git
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

## Study cases (scripts)

The original tutorial notebooks are now executable Python scripts in `scripts/`. Run them with `scripts/run_scripts.sh`; generated movies are saved in `animations/`.

1. **scripts/01_wind_gyre.py** — classic β-plane wind-driven gyre.
2. **scripts/02_gravity_waves.py** — linear gravity/Poincaré waves from a perturbed surface.
3. **scripts/03_tsunami.py** — unforced propagation from a localized uplift over constant depth.
4. **scripts/03_tsunami_shoaling_sponge.py** — tsunami-like wave over variable bathymetry with a sponge layer.
5. **scripts/04_tides.py** — equilibrium-tide forcing via variable geopotential `φ`.
6. **scripts/05_abyssal_flow.py** — Stommel–Arons-like source and sink on a β-plane.
7. **scripts/06_seiche.py** — standing basin modes with no forcing.
8. **scripts/07_equatorial_waves.py** — equatorial Kelvin/Rossby packets.
9. **scripts/08_storm_surge.py** — moving cyclone: wind plus inverse barometer over a shallow shelf.
10. **scripts/09_wind_driven_kelvin_wave.py** — coastal setup and Kelvin-wave release.
11. **scripts/10_geostrophic_adjustment.py** — Gaussian dome adjustment on an f-plane.
12. **scripts/11_Rossby_wave_propagation.py** — balanced Gaussian dome on a β-plane.
13. **scripts/12_wind_gyre_reduced_gravity.py** — reduced-gravity wind-driven gyre examples.


## Teaching labs

A guided five-lab sequence is available in `labs/`, with classroom notebooks directly in `notebooks/`:

1. **Waves in a Box** — `notebooks/01_waves_in_a_box.ipynb`
2. **Tsunami Across the Ocean** — `notebooks/02_tsunami_shoaling.ipynb`; written instructions are available in English and Swedish (`labs/02_tsunami_shoaling_sv.md`)
3. **Wind, Storms, and Coastal Sea Level** — `notebooks/03_wind_storms_and_coasts.ipynb`
4. **Rotation Changes Everything** — `notebooks/04_rotation_geostrophy_rossby.ipynb`
5. **Ocean Pathways** — `notebooks/05_ocean_pathways.ipynb`

The lab notebooks are short, guided, and classroom-friendly. The more technical examples live as scripts in `scripts/`.

## Keeping notebooks clean in git

Notebook outputs and execution counts should not be committed. Before committing, run:

```bash
python3 scripts/strip_notebooks.py
```

To make this automatic for this repository, install the included git hook once:

```bash
git config core.hooksPath tools/git-hooks
```

Or, if you use `pre-commit`:

```bash
pre-commit install
```

After either setup, notebook outputs are stripped before each commit.

## Notes & conventions

* **Units**: SI throughout. Typical `H` in meters, `τ` in N m⁻², `Q` in m s⁻¹, `φ` in m² s⁻², space in meters, time in seconds.
* **C-grid shapes**:

  * `eta: (Ny, Nx)` at cell centers `(x_c, y_c)`
  * `u: (Ny, Nx+1)` staggered in x at `(x_u, y_u)`
  * `v: (Ny+1, Nx)` staggered in y at `(x_v, y_v)`
* **Forcing return**: 3-tuple `(τx_u, τy_v, Q)` **or** 4-tuple `(τx_u, τy_v, Q, φ)`. If you don’t need `φ`, omit it.
* **CFL**: external celerity (c=\sqrt{gH}). `compute_dt_cfl` picks a safe step using min(dx,dy) and the maximum depth when `H` is spatially variable.

---

## Testing

A `tests/` folder is provided as a placeholder; feel free to contribute simple regression checks
(e.g., energy decay with Rayleigh friction, tide period checks, or Kelvin phase speed comparisons).

---

## License

MIT.



## Tsunami shoaling lab

This version includes a simple educational tsunami-shoaling example with variable bathymetry and a sponge layer:

- `src/shallowwater/bathymetry.py` supports scalar, 1-D, or 2-D `params.H`.
- `src/shallowwater/sponge.py` provides `make_sponge_hook(...)` for boundary damping.
- `notebooks/02_tsunami_shoaling.ipynb` is the guided classroom notebook.
- `scripts/03_tsunami_shoaling_sponge.py` is the converted technical example and writes its movie to `animations/`.
- `labs/02_tsunami_shoaling.md` and `labs/02_tsunami_shoaling_sv.md` contain English and Swedish lab instructions.

The example is intended to teach shoaling. It is not a run-up or inundation model.
