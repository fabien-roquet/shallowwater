# Shallow-water waves course materials: implementation TODO

This document is the implementation backlog for the two guided laboratories and
the final group project. Work should proceed in the order below. A stage is not
complete until its validation checklist passes.

## Implementation status (12 August 2026)

All locally implementable items are complete and checked below. The remaining
unchecked items are release or teaching-event gates that cannot be completed in
this workspace:

- publish version 0.1.4 to PyPI and repeat clean installs from PyPI;
- run the configured CI jobs on the hosted Linux runner;
- perform the live 90-minute lecture rehearsal with the added observation material;
- validate the student bundle on a Windows machine;
- freeze/tag the final release archive and record its final git commit.

Local validation used the built 0.1.4 wheel, both NumPy and numba backends, all
six generated notebooks, eight generated GIFs, the compiled lecture, and all
twelve legacy scripts.

## Agreed course structure

- One 90-minute lecture on progressive surface waves, the shallow-water limit,
  and observations of surface waves.
- One 90-minute Part A block: environment check, animated propagation, speed,
  reflection, and a small controlled change.
- One 90-minute Part B block: animated variable-depth experiment followed by an
  animated uniform-wind setup and release.
- A 45-minute Part C project-description/toolbox demonstration at the start of
  the two 90-minute Part C blocks.
- A separate Part C report template for the group project.
- Four student-facing notebooks: Parts A and B, the Part C toolbox demonstration,
  and the Part C report template.
- Instructor solutions for Parts A and B only.
- Students submit only their completed Part C notebook.
- Course installation instructions use PyPI and `pip`, not `uv`.
- `numba` is an optional acceleration extra.
- Rotation may remain available to projects but is not a required taught part.

## Proposed file layout

```text
MT1562_python_lab_waves/
  README.md
  INSTALLATION.md
  TODO.md
  animations/
  lecture/
    shallow_water_waves_lecture.tex
    shallow_water_waves_lecture.pdf
  notebooks/
    part_a_waves_student.ipynb
    part_a_waves_solutions.ipynb
    part_b_bathymetry_student.ipynb
    part_b_bathymetry_solutions.ipynb
    part_c_project_description.ipynb
    part_c_project_report_template.ipynb
  data/
    example_bathymetry.npz
    example_wind_forcing.npz
  instructor/
    teaching_notes.md
    expected_results.md
  scripts/
    build_course_notebooks.py
    validate_course_materials.py
```

Generated PDFs may be committed if that is how course materials will be
distributed. Executed notebook outputs should normally be stripped from student
notebooks; selected outputs may be retained in instructor solutions if useful.

## Definition of done for every implementation stage

- [x] New behaviour is covered by an automated test where practical.
- [x] Relevant student and solution notebooks pass "Restart kernel and run all".
- [x] The pure NumPy backend passes.
- [x] The optional numba backend is checked when numba is installed.
- [x] Existing public APIs remain compatible unless a deliberate breaking change
      is documented and postponed to a later release.
- [x] Existing example scripts are syntax-checked and the relevant representative
      scripts are smoke-run.
- [x] No notebook depends on the repository source tree; it imports the installed
      PyPI package in the same way students will.
- [x] Units, array shapes, sign conventions, and model limitations are stated.
- [x] Student-facing runtimes are measured on a non-numba installation.

---

## Stage 0: freeze scope, release target, and baseline behaviour

- [ ] Confirm the final PyPI version used in the course. The working expectation
      is `0.1.4`, but do not hard-code it in released teaching material until the
      package has actually been published and installed from PyPI successfully.
- [x] Decide whether students install into an existing environment or create a
      small virtual environment. Provide commands for Windows, macOS, and Linux.
- [x] Use `python -m pip` in instructions to reduce ambiguity about which Python
      interpreter receives the package.
- [x] Decide whether the lecture PDF and solution notebooks are tracked in git or
      produced only as release artifacts.
- [x] Set a target runtime for each notebook. Recommended limits without numba:
      Part A <= 5 minutes, Part B <= 8 minutes, and each default-sized Part C
      case <= 3 minutes.
- [x] Record current baseline results for representative uniform-depth,
      variable-depth, and wind-forced cases before changing model code.
- [x] Run and record the status of all existing scripts before model changes.

### Baseline validation

- [x] `python -m pip install -e .` works in a clean environment.
- [x] `python -c "import shallowwater"` works without numba.
- [x] `python -m compileall src/shallowwater scripts` passes.
- [x] All scripts under `scripts/` are run once with a non-interactive plotting
      backend; failures and unusually long cases are documented.
- [x] Representative cases are also run with numba enabled and compared against
      NumPy within a stated tolerance.

---

## Stage 1: establish enforceable regression testing

There is currently no `tests/` directory, and CI runs `pytest -q || true`, which
allows test failures. Model changes should not begin until a small reliable test
suite can fail CI.

- [x] Add `tests/` with fast tests for grid construction, initial conditions,
      scalar and array bathymetry, CFL calculation, boundary enforcement, and a
      short model integration.
- [x] Add a uniform-depth wave-speed regression test. Use a robust arrival-time
      or Hovmoller diagnostic and a tolerance appropriate to the grid.
- [x] Add a variable-depth smoke test that checks finite output and stable array
      shapes rather than relying on a fragile exact field comparison.
- [x] Add a wind-forcing smoke test using an existing forcing callable.
- [x] Add NumPy/numba operator parity tests when numba is available.
- [x] Change CI so `pytest -q` is allowed to fail the job; remove `|| true` after
      the initial tests are stable.
- [x] Add a script syntax test and a curated quick script smoke-test set.
- [x] Keep a separate manual/full regression command for all existing scripts so
      normal CI does not become excessively slow.

### Acceptance tests

- [x] Tests fail when a deliberately incorrect wave-speed or boundary result is
      introduced, demonstrating that they are meaningful.
- [ ] CI passes both with the base dependencies and, in a separate optional job,
      with the `numba` extra.
- [x] Current example scripts require no API edits.

---

## Stage 2: define the course-facing model interface

The notebooks should share a small, stable vocabulary rather than repeat long
setup blocks. Prefer additions around the current API over changes to
`run_model`, `ModelParams`, or existing forcing functions.

- [x] Specify a compact course-facing `run_case(...)` pattern covering:
      grid size, physical domain size, bathymetry, initial state, wind forcing,
      damping, integration duration, and saved variables.
- [x] Decide whether `run_case(...)` belongs in the public package or is kept as
      a short, visible function inside each notebook. Prefer notebook-local code
      if it is teaching-specific.
- [x] Provide reusable diagnostics for centreline Hovmoller plots, virtual gauges,
      arrival time, measured speed, and parameter comparison tables.
- [x] Ensure changing physical domain size does not silently imply constant
      resolution. The notebook interface should display `dx`, `dy`, and CFL `dt`.
- [x] Provide initial-state choices that are simple to compare: cross-basin pulse,
      circular Gaussian bump, configurable amplitude/radius/location, and
      optionally an initial-velocity case.
- [x] Provide wind choices through existing callables or thin wrappers: no wind,
      uniform wind, ramped/time-limited wind, and an optional localized pattern.

### Acceptance tests

- [x] Every Part C control can be changed without editing solver internals.
- [x] Default cases use explicit SI units and print a concise configuration
      summary before integration.
- [x] A group can change exactly one parameter and reproduce the baseline with all
      other parameters unchanged.

---

## Stage 3: bathymetry map input

`ModelParams.H` already accepts a scalar, a length-`Nx` profile, or an
`(Ny, Nx)` array. The safest addition is therefore a validated file loader; the
solver and existing bathymetry paths should remain unchanged.

### File contract

- [x] Select the minimal supported formats. Recommended first version:
      `.npy`, `.npz`, and optionally plain `.csv` arrays.
- [x] Define the required variable name for `.npz` (`H`) and units (positive
      depth in metres).
- [x] Require exact `(Ny, Nx)` cell-centre shape initially. Do not silently
      interpolate or reproject maps.
- [x] Document array orientation: first index is south-to-north `y`, second index
      is west-to-east `x`.
- [x] Decide whether optional `x` and `y` coordinate arrays are validated when
      present.
- [x] Keep all depths finite and strictly positive; dry cells and land masks are
      outside the current model scope.

### Implementation

- [x] Add a function such as `load_bathymetry(path, grid, key="H")` in
      `src/shallowwater/bathymetry.py`.
- [x] Reuse `center_depth(...)` for final validation so loaded and in-memory
      bathymetry obey the same rules.
- [x] Export the loader through `shallowwater.__init__` without changing existing
      exports.
- [x] Add one small synthetic example map to the course `data/` directory.
- [x] Demonstrate an analytic shelf in Part B and a file-loaded map in the Part C
      project-description toolbox.

### Tests

- [x] Load valid `.npy`, `.npz`, and any supported text format.
- [x] Reject missing keys, incorrect shape, NaN/Inf, zero/negative depth, and
      unsupported formats with useful messages.
- [x] Verify that a loaded array produces the same short integration as passing
      that same array directly.
- [x] Re-run scalar, one-dimensional, and two-dimensional existing bathymetry
      tests to ensure compatibility.
- [x] Smoke-run `scripts/03_tsunami_shoaling_sponge.py` unchanged.

---

## Stage 4: forcing map input

The existing solver expects a callable returning C-grid stresses and mass/source
terms. Preserve that interface. File data should be loaded once, converted once,
and wrapped in a callable; no disk access should occur inside the timestep loop.

### File contract decision

- [x] Confirm what instructors mean by a "forcing map": wind stress only, or also
      mass and pressure forcing. Recommended first scope is wind stress only.
- [x] Use centre-point wind-stress arrays `tau_x` and `tau_y`, each `(Ny, Nx)`, in
      N m^-2. Convert them to the model's staggered `u` and `v` locations once.
- [x] Support a static spatial pattern multiplied by a notebook-defined temporal
      envelope in the first implementation.
- [x] Treat time-dependent files (`time, tau_x, tau_y`) as an optional second
      phase only if a course activity needs them; specify interpolation and
      out-of-range behaviour before implementation.
- [x] Use `.npz` as the primary format because it preserves names and multiple
      arrays without adding a dependency.

### Implementation

- [x] Add a helper such as
      `make_wind_forcing_from_file(path, grid, envelope=None)` that returns the
      existing forcing callable signature.
- [x] Reuse or add tested centre-to-C-grid interpolation helpers rather than
      duplicating indexing inside notebooks.
- [x] Return zero `Q_eta` and document that pressure/mass forcing is not implied.
- [x] Export the helper without altering existing forcing functions.
- [x] Add one small synthetic wind map to the course `data/` directory.
- [x] Demonstrate analytic wind in Part B and file-based wind in the Part C
      project-description toolbox.

### Tests

- [x] Validate keys, units documentation, finite values, and exact shapes.
- [x] Confirm a constant centre-point stress becomes the expected constant
      staggered stress, including edges.
- [x] Confirm the temporal envelope is evaluated at model time and zero/constant
      cases behave as documented.
- [x] Confirm the file is read once when constructing the callable, not during
      every model step.
- [x] Compare a file-backed wind case with the same arrays supplied in memory.
- [x] Smoke-run existing wind, tide, storm-surge, and coastal-wind scripts
      unchanged.

---

## Stage 5: PyPI and optional numba installation experience

- [x] Replace course-specific `uv` instructions with:

  ```text
  python -m pip install "shallowwater==<COURSE_VERSION>"
  ```

- [x] Provide the optional accelerated installation:

  ```text
  python -m pip install "shallowwater[numba]==<COURSE_VERSION>"
  ```

- [x] Explain that the first numba-backed run may pause for compilation and that
      numerical results should agree with the NumPy backend within tolerance.
- [x] Add a small backend/version diagnostic that reports package version, Python
      version, NumPy version, and whether numba operators are active.
- [x] Consider exposing `shallowwater.__version__` and a read-only backend-status
      helper if this cannot be done cleanly from public APIs.
- [x] Do not remove `uv` support from the general repository or its developer
      scripts; this decision applies to the course instructions only.
- [ ] After publication, test installation from PyPI in fresh environments on
      Windows, macOS, and Linux, rather than relying on an editable checkout.

### Acceptance tests

- [x] Base installation succeeds with no numba installed.
- [x] Extra installation succeeds and actually activates numba.
- [ ] Part A and Part B solutions execute against the published wheel/sdist.
- [x] The displayed package version matches the pinned course version.

---

## Stage 6: Part A notebook and solution

### Learning arc

- [x] State learning goals and expected duration.
- [x] Include the pip installation/import check and backend diagnostic.
- [x] Introduce the grid, `ModelParams`, initial condition, forcing callable, CFL
      timestep, and model output without touring solver internals.
- [x] Run a uniform-depth, non-rotating, broad disturbance.
- [x] Ask for a prediction of `sqrt(gH)` before showing the numerical result.
- [x] Plot an animation or selected snapshots for discovery.
- [x] Construct a centreline Hovmoller diagram and/or gauge records.
- [x] Measure wave speed and compute relative error from theory.
- [x] Observe reflection at a solid boundary.
- [x] Include one controlled activity changing depth or domain length.
- [x] Provide an optional initial-state or seiche extension clearly marked as
      optional.
- [x] End with a concise physics and model-limitations checkpoint.

### Student/solution production

- [x] Create `part_a_waves_solutions.ipynb` as the canonical authored notebook.
- [x] Tag answer/code cells consistently and generate
      `part_a_waves_student.ipynb` with a small build script.
- [x] Ensure student prompts remain visible and solution content is absent.
- [x] Keep the student notebook executable with sensible defaults; unanswered
      prose fields must not cause code errors.
- [x] Put expected numerical ranges in instructor notes rather than relying on a
      single exact floating-point value.

### Tests

- [x] Validate notebook JSON and cell tags.
- [x] Execute both student and solution notebooks from a clean installed package.
- [x] Assert the measured speed is within a documented tolerance of `sqrt(gH)`.
- [x] Confirm the changed-depth/domain result has the predicted direction and
      approximate scaling.
- [x] Confirm student notebook runtime and memory use meet the target.

---

## Stage 7: Part B notebook and solution

### Guided bathymetry experiment

- [x] Begin with a broad, nearly one-dimensional disturbance to reduce geometric
      spreading and simplify interpretation.
- [x] Run a deep-ocean-to-shallow-shelf case with offshore, slope, shelf, and
      near-coast gauges.
- [x] Ask students to predict speed and arrival-time changes before running.
- [x] Measure local propagation speed and compare with local `sqrt(gH)`.
- [x] Examine surface-elevation changes without calling the last wet-cell height
      inundation, run-up, or hazard.
- [x] Compare two coastal depths or two shelf widths as the required controlled
      experiment.
- [x] Explain the purpose of any sponge layer and distinguish it from a physical
      coast.
- [x] Explicitly state that all cells remain wet and that breaking, wetting and
      drying, and coastal inundation are absent.

### Short wind demonstration and hand-off

- [x] Retain a concise uniform-wind setup and release case as section 4.
- [x] Move file-backed bathymetry, mapped wind, project controls, and project
      design guidance out of Part B into a separate Part C toolbox notebook.

### Student/solution production and tests

- [x] Create `part_b_bathymetry_solutions.ipynb` as the canonical source and
      generate `part_b_bathymetry_student.ipynb` using the same process as Part A.
- [x] Execute both notebooks against a clean installed package.
- [x] Check qualitative and quantitative expected results with robust tolerances.
- [x] Test all toolbox examples, including both supplied input files.
- [x] Confirm Part B remains within the non-numba runtime budget.

---

## Stage 8: Part C project notebook

Part C is the only submitted notebook. It should be shorter than Parts A and B
and organized as a scientific report rather than another tutorial.

- [x] Add a project menu covering wind forcing, bathymetry, domain geometry,
      initial state, damping, periodic forcing, and optional rotation.
- [x] Require one main independent variable; allow a two-factor interaction only
      with instructor approval.
- [x] Provide concise sections for question, prediction, experiment, results and
      interpretation, conclusion, and member contributions.
- [x] Provide a compact parameter table and a reproducible random-seed field if
      any project could use randomness.
- [x] Provide a tested default configuration through `run_case(...)` without
      automatically running or prescribing the students' baseline.
- [x] Include reminders to restart the kernel and run all cells before submission.
- [x] Require outputs needed for assessment to remain visible in the submitted
      notebook while avoiding oversized embedded animations.
- [x] State the G/U assessment scheme and concise passing criteria for scientific
      reasoning, quantitative evidence, interpretation, clarity, and
      reproducibility without inventing weights or points.

### Notebook-only submission and external data

- [x] Resolve the reproducibility constraint before release. Because students
      submit only the notebook, a project must either:
      1. use a course-supplied data file with a stable package/course path,
      2. generate its map arrays inside the notebook, or
      3. adopt a revised submission rule allowing a small data attachment.
- [x] Do not allow an unexplained local absolute path to be the only source of a
      project's bathymetry or forcing.
- [x] Add a validation warning or submission checklist item for external file
      references.

### Tests

- [x] Execute the untouched template from a clean environment.
- [x] Execute one representative project from each main family: wind,
      bathymetry, domain size, and initial state.
- [x] Check the notebook for missing required headings and obvious absolute local
      paths before release.
- [x] Verify that submitted outputs remain readable at normal notebook width.

---

## Stage 9: Beamer lecture

- [x] Replace the existing installation-focused slide deck with a lecture deck
      built for the agreed laboratory sequence.
- [x] Provide approximately 20-24 main slides plus exercise/answer overlays; the
      instructor selects the relevant frames for the 90-minute delivery.
- [x] Cover progressive-wave notation, amplitude/wavelength/period/phase,
      propagation direction, phase speed, particle motion versus signal motion,
      reflection and standing waves, the surface-gravity-wave dispersion
      relation, the shallow-water limit, `c=sqrt(gH)`, and model applicability.
- [x] Include short exercises on interpreting a wave expression, classifying
      shallow/deep waves, speed ratios, travel times, and basin/seiche timescale.
- [x] Use tsunami, tide, seiche, and wind-forced-wave examples as applications,
      not as claims of full realism.
- [x] End with the prediction question that opens Part A.
- [x] Include one concise PyPI installation slide using the confirmed course
      version and optional numba extra.
- [x] Add attribution and accessible figure captions; avoid relying on colour
      alone to distinguish curves.
- [x] Prepare speaker notes or an instructor companion containing exercise
      solutions and suggested timings.

### Tests

- [x] Compile from a clean LaTeX environment with `latexmk` or `pdflatex`.
- [x] Check for missing assets, unresolved references, and serious overfull boxes.
- [x] Verify all equations, units, and numerical exercise answers independently.
- [ ] Present once against a 90-minute timing plan and trim delivery if needed.

---

## Stage 10: course packaging and end-to-end release check

- [x] Write a course README with session order, filenames, installation, expected
      runtimes, submission rule, and troubleshooting.
- [x] Write instructor notes with timing, likely misconceptions, expected plots,
      and recovery steps for installation or runtime problems.
- [x] Add a validation script that checks notebook structure, strips accidental
      student outputs where appropriate, executes release notebooks into a
      temporary directory, and compiles the lecture.
- [x] Ensure the notebook-build step is reproducible and fails if student and
      solution structures drift.
- [ ] Test all materials using only files intended for student distribution and a
      clean PyPI installation.
- [x] Test once with base NumPy and once with the numba extra.
- [ ] Test on at least one Windows machine because installation and animation
      behaviour often differ from macOS/Linux.
- [x] Run the complete pre-change example-script regression suite after all model
      changes.
- [x] Confirm `git status` contains no generated animations, notebook checkpoints,
      numba caches, or unintended executed notebook copies.
- [ ] Freeze the course release archive and record its package version and git
      commit in the instructor notes.

---

## Stage 11: 90-minute blocks and animation pass

- [x] Add a separate Miniconda + VS Code + Jupyter installation guide for
      Windows, macOS, and Linux using `pip` and optional numba.
- [x] Give Part A an immediate two-dimensional wave animation and a short,
      reusable `animate_and_save(...)` helper.
- [x] Save GIFs for every simulated Part A and Part B solution under the course
      `animations/` directory without tracking generated files in git.
- [x] Add a focused note distinguishing topographic scattering and numerical
      dispersion in the first Part B experiment from full surface-wave
      dispersion.
- [x] Expose constant rotation in Part C through the student-facing parameter
      `f`, and include the inertial period among possible theory comparisons.
- [x] Replace notebook LaTeX inline delimiters with `$...$` and display equations
      with `$$...$$`.
- [x] Update student and instructor timing to one 90-minute lecture, one block
      each for Parts A and B, and two blocks for Part C.
- [x] Regenerate all five notebooks and pass structural validation.
- [x] Execute all five notebooks with the NumPy backend and confirm GIF output.
- [x] Execute the two solution notebooks and Part C with the numba backend.
- [x] Run the package regression suite to confirm no existing scripts or APIs
      were broken by the course-material changes.

---

## Stage 12: Part C toolbox and report-template split

- [x] End Part B with the short uniform-wind demonstration as section 4.
- [x] Remove mapped-input and project-design material from Part B.
- [x] Add `part_c_project_description.ipynb` as a 45-minute, instructor-led
      toolbox demonstration rather than a student worksheet.
- [x] Animate a wave over file-backed bathymetry with persistent bottom-depth
      contour lines.
- [x] Animate the response to mapped wind with estimated wind-speed contours and
      subsampled wind-direction arrows.
- [x] Move the project-control menu and example experimental design into the
      toolbox notebook.
- [x] Rename the submission notebook to
      `part_c_project_report_template.ipynb` and update all references.
- [x] Regenerate all six notebooks and remove the legacy Part C filename.
- [x] Execute all notebooks with NumPy and the Part C notebooks with numba.
- [x] Run course validation and package regression tests.

---

## Stage 13: short exploratory report template

- [x] State explicitly that Part C is a short exploratory project graded only G
      (Godkänt) or U (Underkänt).
- [x] Keep the existing assessment criteria without numerical weights.
- [x] Limit the intended experiment to a baseline and one controlled variation,
      with an optional second variation only if time permits.
- [x] Keep only setup imports, `make_initial_state(...)`, and `run_case(...)` as
      provided code; leave case design, diagnostics, and plotting to students.
- [x] Replace the detailed baseline/variation/diagnostic scaffolding with compact
      Experiment and Results-and-interpretation sections.
- [x] Remove the separate Comparison with theory and Limitations sections.
- [x] Regenerate and execute the simplified report template.
- [x] Run course validation and regression tests.

## Suggested implementation order for later sessions

1. Stage 0: confirm scope/version and record baseline.
2. Stage 1: make regression tests enforceable.
3. Stage 2: settle the notebook-facing API.
4. Stages 3-4: implement and test file-backed bathymetry and wind forcing.
5. Stage 5: validate the PyPI/numba installation path.
6. Stage 6: build and test Part A plus its solution.
7. Stage 7: build and test Part B plus its solution.
8. Stage 8: build and test the Part C submission notebook.
9. Stage 9: write and compile the lecture using the final terminology and API.
10. Stage 10: run the end-to-end course release check.

The lecture is intentionally late in the implementation order even though it is
delivered first: installation commands, screenshots, function names, and exercise
results should reflect the tested notebooks and published package rather than an
earlier draft.
