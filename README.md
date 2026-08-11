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
