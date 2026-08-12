# Instructor notes

## Overall timing

The sequence assumes one 90-minute lecture followed by four 90-minute laboratory
blocks: one for Part A, one for Part B, and two for Part C.

### Lecture: 90 minutes

- Slides 1–4, motivation and progressive-wave notation: 10 minutes.
- Slides 5–8, exercise and phase propagation: 15 minutes.
- Slides 9–13, surface-wave dispersion and shallow/deep classification:
  20 minutes.
- Added surface-wave observation material: 10 minutes.
- Slides 14–20, shallow-water speed, reflection, and seiches: 25 minutes.
- Slides 21–24, bathymetry, forcing, model scope, and laboratory launch: 10 minutes.

Exercise answers are revealed on separate frames so students can first work in
pairs. Encourage estimates and units before calculator precision.

### Part A: 90 minutes

- Environment and kernel check: 10 minutes.
- First circular-wave animation: prediction, run, and observation: 15 minutes.
- Right-going pulse, Hovmöller interpretation, and speed measurement: 30 minutes.
- Controlled depth comparison and its animation: 25 minutes.
- Checkpoint discussion or one optional change: 10 minutes.

Ask students to complete `INSTALLATION.md` in advance. Keep installation recovery
available for the first block; a student who is waiting for installation can
observe and interpret a partner's animation without losing the physical thread.

Students commonly confuse propagation of the signal with transport of the same
water across the basin. Ask them to distinguish surface displacement from
depth-averaged velocity. Another common issue is measuring after the pulse has
reflected; point out the pre-reflection branch in the Hovmöller diagram.

### Part B: 90 minutes

- Bathymetry, speed prediction, and first run: 15 minutes.
- Shelf animation and discussion of the dispersive-looking wake: 20 minutes.
- Virtual gauges: 20 minutes.
- Controlled coastal-depth comparison and animation: 20 minutes.
- Short uniform-wind setup/release demonstration: 15 minutes.

Avoid the phrase "most dangerous coast." The last model cell is permanently wet
and is not a beach. Ask students to report modeled surface elevation at a stated
cell and to separate shoaling, reflection, and geometrical effects.

### Part C, block 1: 90 minutes

- Instructor-led project-description/toolbox notebook: 45 minutes.
- Confirm each group's question, prediction, primary control, and diagnostic:
  15 minutes.
- Copy and rename the report template, then run the baseline: 30 minutes.

### Part C, block 2: 90 minutes

- Run one controlled variation and, if time permits, a second: 35 minutes.
- Quantitative diagnostic and interpretation of the main result: 25 minutes.
- Short peer review between groups: 10 minutes.
- Revision, one relevant caveat, conclusion, and run-all check: 20 minutes.

Require groups to show a successful baseline before running a variation.
If a group changes domain size, check whether it has unintentionally changed
resolution. If a group changes several parameters, ask which single causal claim
its design can support.

## Installation recovery

The scientific activity should not be lost to environment troubleshooting.
Before teaching, test the pinned PyPI release on the actual student platform and
keep a pre-created institutional environment or downloadable environment as a
fallback. The course instructions deliberately use `python -m pip`, not `uv`.

Numba is optional. If it causes a platform-specific problem, uninstall the extra
or set `SHALLOWWATER_USE_NUMBA=0` before starting Jupyter. Restart the kernel
after changing this environment variable because the backend is selected at
import time.

## Part C approval checklist

Approve a proposal when it has:

- one answerable question;
- a prediction based on a physical mechanism;
- one baseline and at least one controlled variation;
- one primary independent variable;
- a diagnostic that produces a number, not only an animation;
- a runtime likely to fit the session;
- a reproducible plan for any external input map.

Assessment is binary: G (Godkänt) or U (Underkänt). Do not turn the criteria into
an implicit points system. A modest two-case comparison can pass when the design
is controlled, the evidence is quantitative, and the interpretation is sound.

Optional rotation is suitable only when the chosen time and length scales make
it dynamically visible. The Part C interface exposes the constant Coriolis
parameter as `f`. Students should calculate an inertial period or deformation
radius rather than merely toggling `f`.

## Animation guidance

Parts A and B save each simulated solution as a GIF under `animations/` while
also displaying it inline. The directory is kept in the distribution, but its
generated contents are ignored by git. Animations are for noticing motion and
forming questions; require a gauge, speed, arrival time, period, or other number
for any assessed claim.

The Part C project-description notebook adds two instructor demonstrations. In
the bathymetry example, depth contours remain over the evolving free surface. In
the wind-map example, estimated wind-speed contours and direction arrows remain
visible. Explain that the solver consumes wind stress; displayed wind speed is a
diagnostic inferred with a fixed bulk drag law.
