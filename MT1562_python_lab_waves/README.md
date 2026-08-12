# Shallow-water waves laboratory

This folder contains a compact Master-level teaching sequence about long surface
waves. Students work in groups of two to four and progress from guided model use
to an independent numerical investigation.

## Course sequence

1. A 90-minute lecture on progressive surface waves, the shallow-water limit,
   and observations of surface waves.
2. One 90-minute block for **Part A — Waves in a box:** environment check,
   animated propagation and reflection, wave speed, and a depth comparison.
3. One 90-minute block for **Part B — Variable bathymetry:** an animated
   tsunami-like long wave, dispersion and scattering, virtual gauges, a
   controlled bathymetry experiment, and a short wind-forcing demonstration.
4. A 45-minute **Part C project-description and toolbox demonstration:** mapped
   bathymetry and wind forcing, experiment controls, and project design.
5. **Part C short exploratory project:** run a small controlled comparison and
   communicate one clear result during the remaining project time.

Only the completed Part C notebook is submitted. Parts A and B have separate
instructor solutions.

## Files for students

- `lecture/shallow_water_waves_lecture.pdf`
- `INSTALLATION.md`
- `notebooks/part_a_waves_student.ipynb`
- `notebooks/part_b_bathymetry_student.ipynb`
- `notebooks/part_c_project_description.ipynb`
- `notebooks/part_c_project_report_template.ipynb`
- `data/example_bathymetry.npz`
- `data/example_wind_forcing.npz`
- `animations/` (GIFs are generated here when Parts A and B run)

Keep the `notebooks/` and `data/` directories next to one another. The Part C
project-description notebook locates the supplied maps through a relative path.

The generated model cases are intentionally small. GIF encoding and inline
display take longer than the numerical integrations, especially on the first
run. The release limits are 5 minutes for Part A, 8 minutes for Part B, and 3
minutes for one default-sized Part C case without numba.

## Installation

Follow [`INSTALLATION.md`](INSTALLATION.md) for the Windows, macOS, and Linux
setup. It uses Miniconda, a dedicated `shallowwater-lab` environment, VS Code,
and Jupyter notebooks inside VS Code. The course targets `shallowwater` version
0.1.4 and installs it from PyPI with `pip`, not `uv`:

```text
python -m pip install "shallowwater==0.1.4"
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

Platform-specific setup and troubleshooting are kept in the installation guide
so Part A can begin with the wave rather than a long setup section.

## Model scope

The model solves one-layer shallow-water equations in a rectangular, fully wet
domain. It supports uniform and variable positive depth, solid side walls,
several forcing functions, and optional rotation and damping. It does not
represent dry land, wetting and drying, wave breaking, run-up, inundation, or a
resolved bottom boundary layer.

Bathymetry maps use positive depth in metres and exact cell-centre array shape
`(Ny, Nx)`. Wind-map files use cell-centred `tau_x` and `tau_y` in
$\mathrm{N\,m^{-2}}$.

## Part C submission rule

Start from `part_c_project_report_template.ipynb`, rename the copy with the group
identifier, and submit that executed report notebook. Do not submit the
project-description/toolbox notebook. Required figures and numerical outputs
must remain visible, but large animations should be removed.

The project is graded G (Godkänt) or U (Underkänt), with no points or weighted
criteria. It is intentionally small: a baseline, at least one controlled
variation, and one quantitative result are sufficient when they answer a focused
question clearly.

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
