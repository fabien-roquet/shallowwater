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

Run `python scripts/03_tsunami_shoaling_sponge.py` to generate
`animations/eta_03_tsunami_shoaling_sponge.gif`. The matching tutorial is
`notebooks/03_tsunami.ipynb`.

Experiment 03 uses a **4,800 × 4,800 km square domain**, 320 × 320 cells, and a
20 cm Gaussian source of radius 120 km at x = 1,600 km, y = 2,400 km. The
4,000 m deep ocean slopes to 80 m over the easternmost 900 km. A seven-hour
run shows propagation, geometric spreading and shoaling toward the right.
Weak bottom drag has a **24-hour momentum damping time**; 600 km absorbing
layers on the west, north and south reduce returning waves. The eastern
coast remains reflecting, so its maximum height includes reflection as well
as shoaling. The map keeps equal x/y scales, fixed ±4 cm colours and depth
contours. A centreline time–distance plot and peak-height profile are saved
as `experiment_03_shoaling_diagnostics.png`.

Use `--drag-hours 0` to disable bottom drag, `--hours` to set duration,
`--quick` for a 160 × 160 grid, and `--replot` to redraw a matching saved run.
`experiment_03_tsunami.npz` and its JSON record bathymetry, heights, times and
configuration fingerprints. The guided tsunami labs remain separate:

```text
labs/02_tsunami_shoaling.md
labs/02_tsunami_shoaling_sv.md
notebooks/02_tsunami_shoaling.ipynb
src/shallowwater/bathymetry.py
src/shallowwater/sponge.py
```

This is a teaching example for propagation and shoaling, with no wetting and
drying or inundation.

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

## Experiment 02: gravity waves with and without rotation

```bash
python scripts/02_gravity_waves.py
```

`eta_02.gif` shows rotation on the left and no rotation on the right, with
synchronized physical times and one fixed height color scale. Filled contours
(`contourf`) give smooth colour-band boundaries with 21 fixed levels. Equal-length
arrows show current direction only, every four cells (16 × 16 per panel).
Zero and negligible currents are hidden; arrow length does not encode speed.
Contours are lightened to keep the currents visible. Playback remains at
**10 fps**, with frames saved every two steps instead of four: physical motion
is half as fast. The run covers **1.5 days instead of three**, retaining the
original movie length of about 29 seconds. The separate nonrotating movie and
previous rotating movie are preserved under `animations/archive/`.

Height and C-grid velocities are cached in `experiment_02.npz` and
`experiment_02b_no_rotation.npz`, with model/parameter fingerprints. Use
`--replot` to redraw matching histories, `--fps` to adjust playback,
`--arrow-scale` to control direction-arrow length (smaller means longer), and
`--arrow-stride` for spacing. `--quick` and `--smoke` provide shorter workflows.
The comparison JSON records physical times, frame count, playback duration,
color limits and arrow settings. The notebook calls the same implementation
and retains the coastline diagnostics.

The public `animate_eta` helper also supports single-panel velocity overlays
through `velocity_arrows=True` and optional `arrow_stride`, `arrow_reference`
and `arrow_scale`. Save `out_vars=("eta", "u", "v")` when running the model.
Existing eta-only callers retain their original behavior by default.

## Experiment 04: tides with and without rotation

Run `python scripts/04_tides.py` to produce one `eta_04.gif`, with rotation
on the left and no rotation on the right. Both panels share physical times,
21 fixed filled-contour levels, a common colour bar, and 10 fps playback.
The original 128 × 128 grid, 4000 km square basin, 4000 m depth, three-day
integration, 15-day drag and identical M2 equilibrium-tide forcing are
retained; only f differs (10⁻⁴ s⁻¹ versus zero).

The old separate GIFs are preserved under `animations/archive/` after the
comparison has rendered successfully. Height/time caches and configuration
JSON are saved for both cases. Use `--replot` to redraw matching histories,
`--quick` for a smaller grid, or `--smoke` for a short check. The notebook
`notebooks/04_tides.ipynb` runs the same comparison and retains the coastal
time–distance diagnostics.

## Experiment 06: comparing seiche modes

`eta_06.gif` places `(m,n)=(1,0)` on the left and `(1,1)` on the right,
at common physical times and with a shared ±0.10 m scale. The left mode
varies only along x; the right mode has structure in both directions.
Their periods are about 11.22 and 7.93 hours respectively.

```bash
python scripts/06_seiche.py
python scripts/06_seiche.py --modes 0,0 1,0 0,1 1,1
```

The original basin, grid, depth and forcing-free, frictionless, nonrotating
linear equations are unchanged. Runs cover three periods of the slowest
selected oscillating mode, giving 33.65 hours for the default pair.
The optional `(0,0)` mode is a stationary uniform offset; spatial means are
retained so it remains visible. A stationary-only selection uses the `(1,0)`
period as its observation window. Mode histories and configuration/diagnostic
JSON are cached. Use `--replot` to redraw matching simulations; the notebook
calls the same code.

## Experiment 10: adjustment toward a near-steady vortex

```bash
python scripts/10_geostrophic_adjustment.py
```

`eta_10.gif` now uses coastal momentum friction to absorb returning waves,
instead of the former uniform three-day Rayleigh drag. The damping increases
quadratically toward all four walls across an 800 km band, with a one-hour
limiting coastal time. The basin centre is undamped, and height is not directly
relaxed, preserving mass while allowing the balanced circulation to remain.
The original f-plane, Gaussian bump, partial geostrophic initialization and
160 × 160 grid are retained. The run extends to eight inertial periods
(5.82 days) to show the nearly settled state.

At default resolution, the final height maximum is about 0.376 m. Over the
last inertial period, interior height variability is about 0.008% of its RMS
anomaly, with pressure/Coriolis residual acceleration about 0.0025% of the
Coriolis scale. This is evolved model output, not a prescribed equilibrium.
Diagnostics refer to the undamped interior; exact geostrophic balance is not
imposed within the coastal friction band.

Use `--periods`, `--sponge-hours` and `--sponge-width-km` to explore damping,
`--quick` or `--smoke` for shorter workflows, and `--replot` to redraw matching
saved height/velocity histories. The notebook calls the same implementation.
`experiment_10_adjustment_diagnostics.png` shows the energy and balance
histories; `experiment_10_friction.png` shows the damping field. The configuration
and final diagnostics are recorded beside the numerical cache. The sponge
applies only nonpositive kinetic-energy work and leaves the mass tendency
unchanged apart from the model's original continuity equation.

## Experiment 11: linear and nonlinear Rossby eddies

```bash
python scripts/11_Rossby_wave_propagation.py
```

All five Gaussian movies now show **linear SW on the left and full nonlinear SW
on the right**, using identical initial fields, grids, drag and time steps:

- `eta_11_large_eddy.gif`: R = 3 Ld.
- `eta_11_medium_eddy.gif`: R = Ld.
- `eta_11_small_eddy.gif`: R = Ld/3.
- `eta_11_tiny_eddy.gif`: R = Ld/10, positive height, anticyclonic.
- `eta_11_tiny_eddy_cyclonic.gif`: the exact sign reversal of the tiny initial
  height and geostrophic velocities; all other settings match the anticyclone.

The source Gaussian amplitude is 15 cm and Ld = 313 km; the older three
initial height fields have their basin mean removed. The three older cases retain
the 4000 × 3000 km basin, 192 × 128 grid, 15-day drag and 20-day duration. Tiny
cases retain the 800 × 600 km basin, 256 × 192 grid, 120-day drag and 60-day run.
The new nonlinear branch uses `linear=False`, retaining both momentum and
thickness advection; the existing linear reference uses `linear=True`.
The older velocity initialization retains its 0.95 balance factor and six-cell
wall taper; the tiny cases retain factor 1 and no taper. No nonlinear
gradient-wind correction is applied.
In particular, the tiny anticyclone is deliberately permitted to adjust under
centrifugal acceleration. Its cyclone counterpart illustrates the sign asymmetry.
The f-plane gradient-wind limit for the specified positive Gaussian profile is
A ≤ f² R²/(8g) = 12.5 cm at R = Ld/10. This is a balance restriction for that
profile, not a prediction of numerical failure or complete vortex destruction.
With the default settings, the nonlinear anticyclonic peak falls from about
15 cm to 12.5 cm during the first day; the cyclone deepens to about 17 cm.
Both nonlinear vortices remain compact through day 60, with southward
anticyclonic and northward cyclonic deflection. The initial shape adjusts;
the anticyclone is not destroyed outright.

The first three days of each tiny movie use finer saved-time spacing (about an
hour), followed by the original later spacing. The current sampling regime is
stated in the title; both panels always share the same physical time. Numerical
integration uses a fixed small time step throughout. Continuations at saved
integer steps are checked against uninterrupted integration.

All panels share kilometre axes, a fixed ±0.08 m height scale, and contours every
0.01 m. Each case saves separate `experiment_11_*_linear.npz` and
`experiment_11_*_nonlinear.npz` histories with configuration and diagnostic JSON.
Compatible old tiny linear histories can supply later frames; missing early
frames are integrated, never interpolated. Caches include the exact initial
fields, saved steps, parameters, solver source and experiment fingerprints.
`experiment_11_*_comparison.png` shows peak signed height, early adjustment and
the relative RMS height difference; the matching JSON preserves those series.
Signed height is eta for anticyclones and minus eta for the cyclone, and its
maximum is not a material-vortex tracker.

Select one case with `--case large`, `medium`, `small`, `tiny` or `tiny-cyclonic`;
`--case existing` runs only the three older radii. Use `--replot` for matching
saved pairs, `--quick` for lower spatial resolution, or `--smoke --frames 4` for
a short workflow check. `--days` and `--frames` control the tiny runs. For example:

```bash
python scripts/11_Rossby_wave_propagation.py --case tiny-cyclonic
python scripts/11_Rossby_wave_propagation.py --case existing --replot
```

The filtered-wave GIF remains removed from the default family. Its diagnostic
can still be explicitly requested with `--variant short-waves`; `--variant both`
adds it to the requested Gaussian comparisons. That optional initial-height
filter removes basin sine modes with K Ld < 8, so retained total wavelengths
are below 0.79 Ld. The restriction applies at initialization and does not
suppress later scale coupling. The notebook calls the same comparison code and
retains the linear dispersion/group-velocity explanation.

## Experiment 12: reduced-gravity wind-gyre comparison

```bash
python scripts/12_wind_gyre_reduced_gravity.py
```

`eta_12.gif` now shows linear SW on the left and nonlinear SW on the right,
with one common fixed height scale and contour levels. Both runs start at rest
and use the original 3000 × 3000 km basin, 128 × 128 grid, H = 500 m,
g′ = 0.2 m/s², f₀ = 10⁻⁴ s⁻¹, β = 2 × 10⁻¹¹ m⁻¹ s⁻¹, r = 3 × 10⁻⁷ s⁻¹,
Ah = 500 m²/s and wind-stress amplitude 0.1 N/m². They run for 730 days.
The conservative nonlinear-case CFL bound is used for both, with 121 common
physical output times. The `linear` flag is the only differing model parameter.

`--replot` redraws matching saved numerical histories. `--case linear` or
`--case nonlinear` computes one branch without rendering, and the default
`--case both` assembles the comparison. Use `--quick`, `--smoke`, `--days`,
`--frames`, `--output-dir`, or `--progress` as needed. Caches contain solver,
experiment and parameter fingerprints. Kinetic-energy histories and the shared
color scale/times are saved as comparison diagnostics. The notebook calls the
same implementation and retains the wind and coastline diagnostics.
The former separate movies are preserved under `animations/archive/` when the
comparison is first rendered; `eta_12b_nonlinear.gif` is no longer generated.

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

## Experiment 15: inertial motion and inertia–gravity waves

```bash
python scripts/15_inertial_waves.py
```

This family uses the existing linear shallow-water model on an f-plane, with
periodic x, impermeable y walls, no forcing, no drag and no viscosity. It
illustrates horizontal inertial motion and Poincaré waves in a depth-averaged
layer; it does not represent the vertical structure of three-dimensional
inertial waves. The parameters are H = 100 m, g = 9.81 m s⁻², and
|f| = 10⁻⁴ s⁻¹, giving a reference inertial period Tf = 17.45 hours and
deformation radius Ld = 313 km.

- `velocity_15_inertial_rotation.gif`: a near-inertial standing mode with
  K Ld = 0.2, compared between hemispheres. Arrows show model velocities; the
  lower panels trace velocity space, showing clockwise northern and
  anticlockwise southern rotation. The dotted circle is the analytical pure
  inertial limit, not an additional numerical run or a particle trajectory.
  The finite-wavelength mode has frequency 1.02 |f| and is slightly elliptical.
- `eta_15_poincare_regimes.gif`: the same initial velocity amplitude at
  K Ld = 0.2, 1 and 5. The channel widths differ to set these scales; all panels
  use the same physical clock, velocity-arrow scale and fixed height scale.
  The period decreases and the motion becomes more gravity dominated as
  omega² = f² + g H K². `experiment_15_dispersion.png` shows the dispersion
  relation and numerical velocity time series.
- `eta_15_wave_packet.gif`: a localized meridional velocity packet, with and
  without rotation. It starts with zero height, zonal velocity and linear PV,
  so it excites the wave branch rather than an initially balanced eddy.
  Height maps, arrows and Hovmöller plots show splitting, propagation and the
  oscillatory response. The run ends before the main packet reaches the walls;
  faint dispersive tails may travel faster than the central packet.

All flows are independent of x, avoiding complications from zonal propagation
and Kelvin-wave excitation. Uniform cross-wall inertial flow is not imposed.
The standing mode has v = V sin(pi y/Ly) initially, with u = eta = 0, and
satisfies the wall boundary conditions exactly. Spatial C-grid averaging
slightly changes its frequency; tests compare to the exact discrete mode and
check hemisphere symmetry, mass and energy. Rotation also limits the time step
so a coarse gravity-wave CFL cannot leave inertial oscillations unresolved.

Use `--case rotation`, `--case regimes` or `--case packet` to run one movie;
`--quick` reduces meridional resolution, `--smoke` exercises the full workflow
briefly, and `--replot` reuses matching saved numerical histories. Each case
writes a numerical `.npz` history and `.json` configuration/diagnostics; caches
include solver and experiment fingerprints. The matching notebook calls this
same script. See [Price's Coriolis tutorial, part 2](https://ocw.mit.edu/courses/res-12-001-topics-in-fluid-dynamics-fall-2024/mitres_12_001_f24_essay3_pt2.pdf)
for the shallow-water inertia–gravity dispersion relation and inertial limit.
