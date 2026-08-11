# Shallow-water waves laboratory

This folder contains a compact Master-level teaching sequence about long surface
waves. Students work in groups of two to four and progress from guided model use
to an independent numerical investigation.

## Course sequence

1. A two-hour lecture on progressive surface waves and the shallow-water limit.
2. **Part A — Waves in a box:** installation, wave speed, reflection, and a
   controlled depth experiment.
3. **Part B — Variable bathymetry:** a tsunami-like long wave, virtual gauges,
   file-backed bathymetry and wind maps, and preparation for the project.
4. **Part C — Group investigation:** a supervised open project.

Only the completed Part C notebook is submitted. Parts A and B have separate
instructor solutions.

## Files for students

- `lecture/shallow_water_waves_lecture.pdf`
- `notebooks/part_a_waves_student.ipynb`
- `notebooks/part_b_bathymetry_student.ipynb`
- `notebooks/part_c_project.ipynb`
- `data/example_bathymetry.npz`
- `data/example_wind_forcing.npz`

Keep the `notebooks/` and `data/` directories next to one another. The Part B
notebook locates the supplied maps through a relative path.

The generated default cases are intentionally small. Validation on a laptop
completed Part A in about 2 seconds, Part B in about 3 seconds, and the untouched
Part C baseline in about 1 second after kernel startup. Allow substantially more
time on student machines; the release limits are 5 minutes for Part A, 8 minutes
for Part B, and 3 minutes for the Part C baseline without numba.

## Installation with pip

The course targets `shallowwater` version 0.1.4. After that version has been
published to PyPI, install it with:

```text
python -m pip install "shallowwater==0.1.4"
```

On Windows, `py -m pip` may be used when `python` is not available as a command.

Using a dedicated environment is recommended. On macOS or Linux:

```text
python -m venv shallowwater-course-env
source shallowwater-course-env/bin/activate
python -m pip install --upgrade pip
python -m pip install "shallowwater==0.1.4"
```

On Windows PowerShell:

```text
py -m venv shallowwater-course-env
.\shallowwater-course-env\Scripts\Activate.ps1
py -m pip install --upgrade pip
py -m pip install "shallowwater==0.1.4"
```

Optional numba acceleration is installed with:

```text
python -m pip install "shallowwater[numba]==0.1.4"
```

Numba is not required. Its first model run can take longer while numerical
operators are compiled. The NumPy and numba backends should give equivalent
results within floating-point tolerance.

To verify the environment, start Python or a notebook and run:

```python
import shallowwater
print(shallowwater.backend_info())
```

The displayed `shallowwater` version must be `0.1.4`.

### Troubleshooting

- If `pip` installs into a different interpreter, repeat the command as
  `python -m pip ...` from the same environment used to start Jupyter.
- If the version is not 0.1.4, restart the kernel after upgrading the package.
- If optional numba fails on a platform, use the base installation and restart
  Jupyter with `SHALLOWWATER_USE_NUMBA=0` set before importing the package.
- If Part B cannot find `../data`, restore the distributed directory layout.
- If a run becomes slow, first return to the default grid and duration; doubling
  both horizontal grid dimensions roughly quadruples the number of grid cells.

## Model scope

The model solves one-layer shallow-water equations in a rectangular, fully wet
domain. It supports uniform and variable positive depth, solid side walls,
several forcing functions, and optional rotation and damping. It does not
represent dry land, wetting and drying, wave breaking, run-up, inundation, or a
resolved bottom boundary layer.

Bathymetry maps use positive depth in metres and exact cell-centre array shape
`(Ny, Nx)`. Wind-map files use cell-centred `tau_x` and `tau_y` in N m^-2.

## Part C submission rule

Submit one executed `.ipynb` file named with the group identifier. Required
figures and numerical outputs must remain visible, but large animations should
be removed.

Because only one notebook is submitted, external data must be reproducible. A
project may use a supplied course map, generate its map in the notebook, or
include a compact documented data representation in a cell. Unexplained local
absolute paths are not accepted.

## Instructor and development material

- `instructor/teaching_notes.md` contains timings and facilitation notes.
- `instructor/expected_results.md` records expected numerical ranges.
- `scripts/build_course_notebooks.py` generates all notebooks and example data.
- `scripts/validate_course_materials.py` checks the release structure and can
  execute notebooks and compile the lecture.
- `TODO.md` records the implementation and validation backlog.

Do not edit generated student notebooks independently of the build script; the
script is the canonical source for student/solution pairs.
