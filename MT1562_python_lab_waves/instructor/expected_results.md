# Expected numerical results

These ranges describe the generated notebooks using the NumPy backend. Small
differences are expected with platform, backend, and plotting/sample intervals.
On the validation laptop, the numerical integrations remain only a few seconds.
GIF encoding and inline HTML rendering now dominate the Part A and Part B run
time. The much looser release limits of 5 minutes for Part A, 8 minutes for Part
B, and 3 minutes for one default-sized Part C case accommodate slower student
machines.

## Part A

- Baseline: `H=400 m`, `Lx=1600 km`, `dx=10 km`.
- Theoretical speed: `sqrt(9.81*400) = 62.64 m/s`.
- Default peak-position measurement: approximately `62.3 m/s`, normally below
  2% relative error.
- The incident pulse reaches the eastern boundary after roughly 5.8 hours and
  then produces a westward reflected branch.
- For `H=900 m`, theoretical speed is `93.96 m/s`, exactly 1.5 times the
  baseline theoretical speed. Reflection occurs earlier.
- The opening circular bump spreads in all directions, reaches the nearer walls
  first, and then develops crossing reflected fronts.
- A complete run writes three multi-frame GIFs under `animations/`.

## Part B

- Deep and coastal depths: 3000 m and 120 m.
- Theoretical limiting speeds: approximately 171.6 m/s and 34.3 m/s.
- Typical positive-peak times for gauges at 700, 1400, 1900, and 2250 km are
  approximately 0.56, 1.73, 2.60, and 3.72 hours.
- Typical peak surface displacements are approximately 0.119, 0.117, 0.141,
  and 0.192 m. Treat these as grid-dependent model diagnostics, not run-up.
- With coastal depth changed to 300 m, the last-gauge peak occurs around
  3.62 hours with amplitude around 0.176 m.
- The example bathymetry has shape `(24, 160)`, minimum depth about 251 m, and
  maximum depth 2000 m.
- Eastward uniform wind produces positive setup at the eastern wall and negative
  displacement toward the west; free oscillations follow shut-off.
- The wavetrain behind the shelf-crossing pulse is tied to topographic
  scattering and interference, with a possible numerical-dispersion component.
  It is not the full finite-depth dispersion relation.
- A complete run writes shelf-120 m, shelf-300 m, and wind-release GIFs under
  `animations/`.

## Part C report helper defaults

- Default grid: `120 x 24`, domain `1200 x 240 km`, `H=400 m`.
- Default `dx=10 km` and CFL timestep approximately 67 s.
- Calling `run_case(...)` with its defaults runs a five-hour cross-basin pulse.
- The untouched report template does not run a baseline or prescribe a plot or
  diagnostic; those choices belong to the group.

## Part C project-description toolbox

- The file-backed bathymetry case uses the supplied `(24, 160)` depth map and
  writes `part_c_toolbox_bathymetry.gif` with persistent depth contours.
- The mapped-wind case derives a display-only wind speed from the stress map
  using $\rho_{air}=1.225\ \mathrm{kg\,m^{-3}}$ and $C_D=1.3\times10^{-3}$.
  Its animation includes wind-speed contours and subsampled direction arrows.
- Both animations are qualitative project demonstrations; assessed conclusions
  still require a numerical diagnostic.

## Interpretation tolerances

Do not mark on exact agreement with the numbers above. A correct measurement
should state sampling or grid uncertainty, preserve units, and support the
predicted direction or scaling. Strong submissions distinguish physical-model
limitations from resolution and experimental-design limitations.
