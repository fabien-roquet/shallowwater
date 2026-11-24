# Shallow Water Model

A small single-layer shallow-water model for teaching, experimentation, and simple geophysical-fluid-dynamics demonstrations.

The model is intentionally lightweight. It includes tutorial-style Python scripts, guided lab notebooks, and examples covering gravity waves, tsunamis, wind-driven circulation, storm surge, rotation, Rossby waves, tides, and simple abyssal/source-sink flows.

## Repository layout

```text
src/shallowwater/        Core model package
notebooks/              Guided lab notebooks
labs/                   Lab handouts and instructions
scripts/                Runnable example scripts and helper tools
animations/             Generated animations; not tracked by git
.github/                GitHub metadata/workflows; keep tracked
```

The `animations/` directory is for generated output and should be ignored by git.

## Installation from PyPI

Install the released package with pip:

```bash
python -m pip install shallowwater
```

Optional numba acceleration is available as an extra:

```bash
python -m pip install "shallowwater[numba]"
```

Then test that the package imports and inspect the active backend:

```bash
python -c "import shallowwater; print(shallowwater.backend_info())"
```

For development from a source checkout, install in editable mode:

```bash
python -m pip install -e .
```

Developers who use `uv` may instead run `uv sync`; the Master-level course
materials in `MT1562_python_lab_waves/` use only the pinned PyPI/pip workflow.

## Guided labs

The guided lab material is in `labs/` and `notebooks/`.

Recommended lab sequence:

1. `labs/01_waves_in_a_box.md`  
   Gravity waves, reflections, and seiches.

2. `labs/02_tsunami_shoaling.md`  
   Tsunami propagation, variable bathymetry, shoaling, and sponge layers.

3. `labs/02_tsunami_shoaling_sv.md`  
   Swedish version of the tsunami-shoaling lab.

4. `labs/03_wind_storms_and_coasts.md`  
   Wind forcing, coastal setup, storm surge, and coastal waves.

5. `labs/04_rotation_geostrophy_rossby.md`  
   Rotation, geostrophic adjustment, deformation radius, and Rossby waves.

6. `labs/05_ocean_pathways.md`  
   Gyres, tides, source-sink flows, and reduced-gravity behaviour.

The corresponding notebooks are kept directly in `notebooks/`:

```text
notebooks/01_waves_in_a_box.ipynb
notebooks/02_tsunami_shoaling.ipynb
notebooks/03_wind_storms_and_coasts.ipynb
notebooks/04_rotation_geostrophy_rossby.ipynb
notebooks/05_ocean_pathways.ipynb
```

Notebook outputs should not be committed. See [Keeping notebooks clean](#keeping-notebooks-clean).

## Running example scripts

The older tutorial notebooks have been converted to Python scripts under `scripts/`. These scripts can be used to regenerate examples and animations.

Generated animations are written to:

```text
animations/
```

This folder should not be tracked by git.

### Linux

List available scripts:

```bash
uv run ./scripts/run_scripts_linux.sh -l
```

Run one script:

```bash
uv run ./scripts/run_scripts_linux.sh scripts/01_wind_gyre.py
```

Run all scripts:

```bash
uv run ./scripts/run_scripts_linux.sh -a
```

### macOS

List available scripts:

```bash
uv run ./scripts/run_scripts_mac.sh -l
```

Run one script:

```bash
uv run ./scripts/run_scripts_mac.sh scripts/01_wind_gyre.py
```

Run all scripts:

```bash
uv run ./scripts/run_scripts_mac.sh -a
```

The macOS runner avoids Bash features such as `mapfile`, which are not available in the older Bash version shipped with macOS.

### Windows PowerShell

List available scripts:

```powershell
uv run powershell -ExecutionPolicy Bypass -File .\scripts\run_scripts_windows.ps1 -List
```

Run one script:

```powershell
uv run powershell -ExecutionPolicy Bypass -File .\scripts\run_scripts_windows.ps1 -Script scripts\01_wind_gyre.py
```

Run all scripts:

```powershell
uv run powershell -ExecutionPolicy Bypass -File .\scripts\run_scripts_windows.ps1 -All
```

## Optional numba acceleration

The model can optionally use numba-accelerated operators. By default, the example scripts run without numba so that the code works in the simplest possible environment.

If numba is disabled, scripts may print:

```text
not using numba
```

To enable numba on Linux or macOS, set:

```bash
export SHALLOWWATER_USE_NUMBA=1
```

Then run a script, for example:

```bash
uv run ./scripts/run_scripts_mac.sh scripts/01_wind_gyre.py
```

You can also enable numba for a single command:

```bash
SHALLOWWATER_USE_NUMBA=1 uv run ./scripts/run_scripts_mac.sh scripts/01_wind_gyre.py
```

Run all scripts with numba:

```bash
SHALLOWWATER_USE_NUMBA=1 uv run ./scripts/run_scripts_mac.sh -a
```

On Windows PowerShell:

```powershell
$env:SHALLOWWATER_USE_NUMBA = "1"
uv run powershell -ExecutionPolicy Bypass -File .\scripts\run_scripts_windows.ps1 -All
```

Check whether numba is installed:

```bash
uv run python -c "import numba; print(numba.__version__)"
```

Check whether the numba operators import correctly:

```bash
uv run python -c "from shallowwater import operators_numba; print('numba operators import OK')"
```

Numba acceleration is used only when all of the following are true:

1. `numba` is installed.
2. `src/shallowwater/operators_numba.py` imports successfully.
3. `SHALLOWWATER_USE_NUMBA=1` is set.

If any of these are false, the scripts should fall back to the pure NumPy implementation.

## Tsunami shoaling example

The tsunami-shoaling material demonstrates a long wave crossing from deep water toward a shallow coast.

Key ingredients:

- variable bathymetry
- shallow-water wave speed, `c = sqrt(g H)`
- shoaling as the wave enters shallower water
- a sponge layer that damps outgoing waves and reduces reflections

Relevant files:

```text
labs/02_tsunami_shoaling.md
labs/02_tsunami_shoaling_sv.md
notebooks/02_tsunami_shoaling.ipynb
src/shallowwater/bathymetry.py
src/shallowwater/sponge.py
```

This is a teaching example for wave propagation and shoaling. It is not a full coastal inundation or wetting-and-drying model.

## Bathymetry and wind maps from files

Version 0.1.4 adds small validated input helpers while preserving the existing
solver interfaces.

Bathymetry maps may be loaded from `.npy`, `.npz`, `.csv`, or `.txt` files:

```python
H = load_bathymetry("bathymetry.npz", grid)
params = ModelParams(H=H)
```

The map must have exact cell-centre shape `(Ny, Nx)`, contain finite positive
depth in metres, and use array order `(y, x)`. The loader does not interpolate,
reproject, or create dry land.

A static cell-centred wind-stress map can be loaded once and wrapped in the
normal forcing-callable interface:

```python
forcing_fn = make_wind_forcing_from_file(
    "wind.npz",
    grid,
    envelope=lambda t: min(1.0, t / 3600.0),
)
```

The `.npz` file contains `tau_x` and `tau_y`, both `(Ny, Nx)` in N m^-2.

## Keeping notebooks clean

Notebook outputs should not be committed to git.

To strip outputs manually:

```bash
uv run python scripts/strip_notebooks.py notebooks/*.ipynb
```

Then commit the cleaned notebooks:

```bash
git add notebooks/*.ipynb
git commit -m "Clean notebook outputs"
```

A git pre-commit hook may also be used, but it is optional. If a custom hook causes problems, disable it with:

```bash
git config --unset core.hooksPath
```

## Git ignore recommendations

The repository should keep `.github/` tracked, but ignore local environments, generated outputs, and Jupyter helper files.

Recommended `.gitignore` entries include:

```gitignore
# Python cache
__pycache__/
*.py[cod]
*$py.class

# uv / virtual environments
.venv/
.uv-cache/
uv-cache/

# Python build / packaging
build/
dist/
*.egg-info/
.eggs/

# Test / coverage output
.pytest_cache/
.coverage
htmlcov/
.tox/
.nox/

# Jupyter
.ipynb_checkpoints/
.virtual_documents/

# Generated model outputs
animations/
outputs/
figures/
movies/
*.mp4
*.mov
*.avi
*.gif
*.webm

# Generated data / diagnostics
*.nc
*.h5
*.hdf5
*.zarr/
*.npz

# Logs
*.log

# OS/editor files
.DS_Store
Thumbs.db
.vscode/
.idea/
*.swp
*.swo

# Local environment config
.env
.env.*
```

Do not ignore `.github/`; it is normally used for GitHub Actions, issue templates, pull-request templates, and other repository metadata.

## Development notes

Run a quick import check:

```bash
uv run python -c "import shallowwater; print('OK')"
```

Compile the scripts to catch syntax errors:

```bash
uv run python -m py_compile scripts/*.py
```

Run a single example before running all scripts:

```bash
uv run ./scripts/run_scripts_mac.sh scripts/01_wind_gyre.py
```

Replace `run_scripts_mac.sh` with the Linux or Windows runner as appropriate.

## Experiment 11: smaller Rossby disturbances

```bash
python scripts/11_Rossby_wave_propagation.py --case tiny
```

This adds two full linear shallow-water runs with radius `Ld/10` (31 km),
resolved with ten cells per radius on an 800 × 600 km regional grid. The
120-day drag time keeps the slower short-wave evolution visible over 60 days.
The older `Ld/3`, `Ld`, and `3*Ld` cases remain available with `--case existing`;
without a case selector the script runs both the old and new cases.

- `eta_11_tiny_eddy.gif` starts from the smaller Gaussian.
- `eta_11_short_waves.gif` removes initial basin sine modes with `K*Ld < 8`,
  tapering smoothly to full amplitude at `K*Ld = 10`. Retained total wavelengths
  are below `0.79*Ld`. The initial rings are a consequence of removing long
  wavelengths; a positive Gaussian alone cannot meet that restriction.

Use `--variant gaussian` or `--variant short-waves` to compute just one,
`--replot` to redraw matching saved histories, and `--smoke --frames 3` for a
short workflow check. Numerical histories and initial-spectrum/Hovmöller plots
are saved beside the movies. All five experiment-11 movies share the same plotter: kilometre axes, matching
layout and titles, fixed ±0.08 m color limits, and contours every 0.01 m.
Redraw the three older cases from saved histories with
`python scripts/11_Rossby_wave_propagation.py --case existing --replot`.
The cutoff applies to initial height, using its sine extension across basin
walls. It is a cutoff in total wavenumber, not both component wavenumbers.
The stationary spatial mean is removed from the display; varying Coriolis
parameter, boundaries and shallow-water adjustment can later introduce other
scales. These are regional cases, rather than a domain-size-matched comparison
to the older animations.

For a local balanced wave, `omega = -beta*k/(k²+l²+Ld⁻²)`. Broader waves have
faster westward zonal phase propagation, but group propagation can be eastward
or westward. The alternating lobes and chevrons are features of linear PV
wave dynamics and dispersion. In the short-wave limit, shrinking the eddy
slows its evolution. The notebook includes the dispersion and group-velocity
formulas and calls the same new implementation as the script.

## Experiments 13 and 14: forced jets and double-gyre adjustment

These experiments evolve the full shallow-water state from rest under wind
stress. They do not prescribe or linearize about a reference circulation.
Run them from the repository root (or use an absolute script path):

```bash
python scripts/13_barotropic_instability.py
python scripts/14_double_gyre_adjustment.py
```

Use `--quick` for a lower-resolution demonstration, `--smoke --frames 5` for a
short workflow check, and `--format mp4` for MP4 output instead of GIF (requires
FFmpeg). `--frames` sets the approximate number of saved frames, including the
initial and final states. `--output-dir PATH` changes the output location. The
NumPy backend works without numba; `SHALLOWWATER_USE_NUMBA=1` enables optional
acceleration. Default runs integrate many simulated days and take longer than
the smoke checks. Companion notebooks are `notebooks/13_barotropic_instability.ipynb` and
`notebooks/14_double_gyre_adjustment.ipynb`; they call the same scripts.
These scripts are guarded by `if __name__ == "__main__"`, so
importing them does not launch a simulation.

### 13 — Wind-driven jet and barotropic instability

`13_barotropic_instability.py` uses a 2000 × 1200 km channel, periodic in x and
closed with free-slip walls in y. A steady eastward surface stress proportional
to `sech²((y-yc)/W)`, with `W = 100 km`, builds a jet from rest for 25 days. A tiny
balanced periodic disturbance is then added, and the unchanged stress acts for
another 100 days. The main comparison uses strong wind with momentum advection
enabled and disabled. Add `--include-weak` to also run the weak-wind case for
numerical diagnostics. All cases retain nonlinear mass
transport and the dependence of wind acceleration on local layer thickness.

The default grid is 160 × 96 (`--quick`: 80 × 48). The weak and strong peak
stresses are 0.005 and 0.08 N m⁻². The seed combines zonal modes 4 and 5; its
surface-height amplitude coefficient is 0.002 m. The diagnostics include eddy
kinetic energy about the instantaneous zonal mean, mean velocity, and the
meridional gradient of the zonal-mean shallow-water PV `(f + zeta)/(H + eta)`.
A PV-gradient sign change is a necessary indicator in the appropriate inviscid
parallel-flow setting, not proof of a growing instability. Use the time history
of eddy energy and a resolution comparison as well. The advection-off case is a
controlled term-removal experiment, not the package's fully linear mode.

Generated files in `animations/`:

- `eta_13_barotropic_instability.gif`: two columns (advection on/off), with
  rotational velocity streamfunction and contours on top and relative vorticity
  below. Each row uses the same fixed color scale across both cases.
  The old spin-up animation is no longer generated; an existing copy is moved
  to `animations/archive/` when that archive name is free.
- `experiment_13_diagnostics.png`: eddy energy, final jet, and final PV gradient.
- `experiment_13_*.npz` and `.json`: numerical diagnostics, coordinates, stress,
  spin-up histories, streamfunction and vorticity fields, and run parameters.

The streamfunction satisfies `laplacian(psi) = zeta`, with periodic x and
constant values at the channel walls. Their difference preserves the integral
of the zonal-mean velocity across the channel. The convention is
`u_rot = -dpsi/dy`, `v_rot = dpsi/dx`, with a zero spatial-mean gauge. This
diagnoses the nondivergent part of velocity; shallow-water flow also contains
a divergent component. It is not an approximation based on surface height.

A perfectly symmetric unseeded numerical jet stays symmetric. The reproducible
seed makes the growth comparison explicit. The background is allowed to evolve
throughout; the seeded interval is not an eigenmode calculation about a frozen
jet. Use a wider domain or a different seed to explore permitted wavelengths,
and refine the grid before interpreting a measured growth rate quantitatively.

### 14 — Persistent patches and a linear/nonlinear wind ramp

Experiment 14 contains two independent comparisons, selected with `--part`:

```bash
python scripts/14_double_gyre_adjustment.py --part patches
python scripts/14_double_gyre_adjustment.py --part wind-ramp
```

Without `--part`, both run. `--quick` reduces resolution; `--smoke --frames 5`
checks the complete workflow with very short simulations. `--replot` redraws
saved fields without repeating integration; it also accepts `--part`.

The patch comparison uses a 2000 × 1600 km closed basin with sinusoidal wind
stress, 20-day Rayleigh drag, viscosity of 2000 m² s⁻¹, peak stress 0.015 N m⁻²,
and beta 5e-12 m⁻¹ s⁻¹. Both wind signs spin up independently for 240 days.
Each background branches into a control and an identical balanced central
patch (0.08 m, Gaussian radius 180 km), followed for 45 days.

`eta_14_balanced_patch.gif` uses a 2 × 2 layout: eastward central jet on top,
westward jet below, total height on the left and departure from the matched
control on the right. Black contours show the evolving control height. The
separate comparison and reversed-wind GIFs are retired. Stationarity depends
on wavelength, current and PV distribution; the patch Hovmöller diagrams in
`experiment_14_wave_propagation.png` show the actual propagation.

The new `eta_14_wind_step.gif` has **two panels: truly linear (`linear=True`)
and fully nonlinear (`linear=False`)**. Both begin nearly at rest with the same
tiny balanced seed and receive exactly the same wind history. The default uses
broad double-gyre stress `-tau0*cos(2*pi*y/Ly)`, rather than the earlier narrow
wind band. A 0.1-percent prescribed perturbation is identical in both cases.

The conservative default is a 2000 × 1600 km basin, beta 2e-11 m⁻¹ s⁻¹,
40-day Rayleigh drag, and uniform viscosity of 1000 m² s⁻¹. Wind grows from
zero to 0.04 N m⁻² over 120 days and is held for another 120 days. The default
grid is 128 × 96 (`--quick`: 96 × 64). These settings retain a western-intensified
circulation and moderate nonlinear recirculation. They are **not a demonstrated
barotropic-instability threshold**; the gentler default settles nearly steadily.
Experiment 13 remains the controlled barotropic-instability demonstration.

To continue the saved baseline through successive stronger winds:

```bash
python scripts/14_double_gyre_adjustment.py --part wind-ramp --increase-wind \
  --wind-levels .06 .08 .10 .12 .14 .16 .20 .24 .30 .40
```

Each level rises continuously over 60 days and is held for 60 days. Set
`--wind-stage-days` and `--wind-hold-days` to change those durations.
`--wind-drag-days` and `--wind-viscosity` optionally change dissipation in both
models at the start of continuation; omitting them preserves the baseline.
A first level of `.04` allows adjustment to changed dissipation before raising
the wind. Both panels follow identical stress histories and time steps.
Stage checkpoints include their parent state, forcing, solver and parameters.
The combined continuation is saved separately as `experiment_14_wind_sweep.npz`
and `.json`; the original baseline remains available. Rendering replaces
`eta_14_wind_step.gif`, preserving its baseline copy under `animations/archive/`.
Use `--increase-wind --replot --part wind-ramp` to redraw the continuation.

The sweep screens fluctuations during constant-wind holds and their estimated
mean-flow shear production. It adds a 120-day fixed-wind confirmation when a growing disturbance is found,
and stops if variability persists, or before including a stage whose eastern
boundary current dominates. These screens
cannot certify barotropic instability: growing disturbances must be confirmed
under fixed wind and checked for numerical convergence. A steady nonlinear
recirculation is not counted as instability.

In the continuation with the original dissipation, disturbances decay through
0.30 N m⁻² and grow during the 0.40 N m⁻² hold. They continue growing during an
additional 120 days at constant wind, with positive mean-shear energy production.
This brackets an onset in this particular run; it is not a measured critical
stress. The required forcing produces strong currents and much weaker western
intensification than the moderate baseline.

The current movie uses a cleaner continuation that halves both the viscosity
and the drag rate (80-day drag, 500 m² s⁻¹ viscosity). Allow the saved gyre to adjust at 0.04 N m⁻² before raising
the wind:

```bash
python scripts/14_double_gyre_adjustment.py --part wind-ramp --increase-wind \
  --wind-levels .04 .06 .08 .10 --wind-stage-days 60 --wind-hold-days 120 \
  --wind-drag-days 80 --wind-viscosity 500
```

At 0.10 N m⁻² the disturbance grows and then remains variable during the
additional 120-day fixed-wind check. The held-wind western/eastern current RMS
ratio is about 3.3 and the mean-shear production is positive. Peak speed during
that check is about 2.9 m s⁻¹. The movie displays the dissipation values, including
the change from baseline settings at the start of continuation. Modified-mixing
checkpoints live in their own subdirectory so the original sweep remains reusable.
The developed variability also persists on a 192 × 144 grid: over held-wind
days 30–80, mean kinetic energy differs by about 0.2% and temporal fluctuation
energy by about 7% from the 128 × 96 run. This checks persistence after
interpolating the disturbed state, not convergence of the onset threshold.
The fields and comparison are saved as `experiment_14_wind_sweep_refined.npz`
and `experiment_14_wind_sweep_validation.json`. Finite-window shear-production
averages depend on the oscillation phase and are not a closed eddy-energy budget.

The previous 6000 km narrow-band example used weak beta and long drag times,
and developed an eastern jet-impact/recirculation regime with much stronger
flow. Small numerical energy defects and grid agreement did not establish that
it represented the intended western-intensified ocean gyre. Its saved fields
and movie are archived under `animations/archive/` when replaced.

Use `--ramp-days`, `--ramp-hold-days`, `--ramp-tau-max`, `--ramp-drag-days`,
`--ramp-viscosity`, `--ramp-beta`, and `--ramp-length-km` for controlled
comparisons. `--ramp-wind-profile jet` retains a narrow-band sensitivity case;
`--ramp-wind-width-km` applies to that profile. Optional enhanced coastal mixing
is set with `--ramp-boundary-viscosity` and `--ramp-boundary-width-km`. The default
coastal value equals the interior value, so mixing is uniform. The optional
extra operator is conservative `div(nu grad(velocity))`, with a regression test
that its discrete kinetic-energy work is non-positive. It does not damp height.

The usual coloring is height; `--ramp-field vorticity` redraws the same
simulation with vorticity coloring. Each panel also has contours of its field.
The ramp archive `experiment_14_wind_ramp.npz` includes height and velocity
histories, vorticity, wind, temporal variability about a roughly 20-day mean,
a shear-production estimate, and an inviscid energy-budget defect. Statistics
near the first/last half-window are affected by temporal endpoint padding.

`experiment_14_wind_ramp_diagnostics.png` shows mean kinetic energy, temporal
variability, and western/eastern boundary-current strength. The index uses meridional-velocity RMS in
equal-width outer 15-percent strips, excluding the northern/southern 10 percent
of the basin. A ratio above one indicates western dominance. The run metadata
summarizes that ratio, speed and kinetic energy. Energy and variability alone
must not be treated as proof of a physically appropriate instability.

Individual cases can be computed using `--ramp-case linear` or `nonlinear`;
the default `both` assembles the comparison. At least 161 snapshots are saved
for full ramp runs. Matching cases can be reused for assembly and replotting.

The closed-basin vorticity stencil was corrected during development of this
case: both derivatives are now evaluated at C-grid corners before averaging
to centres. The previous offset derivatives could create spurious irregular
flow at stronger wind. NumPy and numba use the same corrected stencil, with
analytic-mode and boundary regressions. Numerical spin-up checkpoints include
a solver-source fingerprint and are invalidated when those sources change.

Patch spin-ups and wind-ramp cases are cached with their configurations.
Changing only patch amplitude can reuse its background; `--fresh-spinup` forces
a new patch spin-up. Generated `.npz`, `.json`, images and movies remain ignored
by git. The old step simulation and movie are preserved under
`animations/archive/` when the new ramp movie is rendered.

### Solver options used by these experiments

```python
grid = make_grid(Nx, Ny, Lx, Ly, periodic_x=True)
params = ModelParams(linear=False, momentum_advection=False)
```

`periodic_x` defaults to `False`. For a periodic C-grid, the first and last
columns of `u` represent the same face; the first is independent. The solver
enforces their equality, including in saved states, and retains zero normal
velocity at the y walls. Periodic stencils cover gradients, interpolations,
vorticity, depth averaging, wind-map averaging, and viscosity. Channel-specific
stencils use NumPy and reuse backend operators where applicable; the existing
closed-basin operators retain optional numba acceleration.

`momentum_advection` defaults to `True`. With `linear=False`, setting it to
`False` removes both the relative-vorticity and kinetic-energy-gradient terms
in the vector-invariant momentum equation, retaining Coriolis, surface pressure,
wind, drag, and viscosity. With `linear=True`, momentum advection is already
absent regardless of this flag. The original linear/nonlinear defaults remain
unchanged. Wall-normal momentum tendencies are now constrained at each RK stage,
so saved wall velocities obey the same boundary condition used for fluxes.
