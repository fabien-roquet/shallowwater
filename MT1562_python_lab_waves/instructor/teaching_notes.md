# Instructor notes

## Overall timing

The sequence assumes a two-hour lecture and three laboratory half-days of about
three hours each. Part C receives the full third half-day.

### Lecture: 120 minutes

- Slides 1–4, motivation and progressive-wave notation: 15 minutes.
- Slides 5–8, exercise and phase propagation: 20 minutes.
- Slides 9–13, surface-wave dispersion and shallow/deep classification:
  25 minutes.
- Break: 10 minutes.
- Slides 14–20, shallow-water speed, reflection, and seiches: 30 minutes.
- Slides 21–24, bathymetry, forcing, model scope, and laboratory launch:
  20 minutes.

Exercise answers are revealed on separate frames so students can first work in
pairs. Encourage estimates and units before calculator precision.

### Part A: about 3 hours

- Environment check and model vocabulary: 25 minutes.
- Baseline run and Hovmöller interpretation: 45 minutes.
- Speed prediction and measurement: 35 minutes.
- Break: 10 minutes.
- Controlled depth comparison: 40 minutes.
- Optional investigation and checkpoint discussion: 25 minutes.

Students commonly confuse propagation of the signal with transport of the same
water across the basin. Ask them to distinguish surface displacement from
depth-averaged velocity. Another common issue is measuring after the pulse has
reflected; point out the pre-reflection branch in the Hovmöller diagram.

### Part B: about 3 hours

- Bathymetry, prediction, and first run: 45 minutes.
- Virtual gauges and controlled coastal-depth comparison: 55 minutes.
- Break: 10 minutes.
- File-map and wind demonstrations: 35 minutes.
- Part C controls and proposal: 35 minutes.

Avoid the phrase "most dangerous coast." The last model cell is permanently wet
and is not a beach. Ask students to report modeled surface elevation at a stated
cell and to separate shoaling, reflection, and geometrical effects.

### Part C: about 3 hours

- Confirm question, prediction, controls, and baseline: 20 minutes.
- Main experiments: 90 minutes.
- Break: 10 minutes.
- Diagnostics and theoretical comparison: 35 minutes.
- Peer review and revision: 25 minutes.

Require groups to show a successful baseline before launching a parameter sweep.
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
- one baseline and at least two controlled variations;
- one primary independent variable;
- a diagnostic that produces a number, not only an animation;
- a runtime likely to fit the session;
- a reproducible plan for any external input map.

Optional rotation is suitable only when the chosen time and length scales make
it dynamically visible. Students should calculate an inertial period or
deformation radius rather than merely toggling `f0`.
