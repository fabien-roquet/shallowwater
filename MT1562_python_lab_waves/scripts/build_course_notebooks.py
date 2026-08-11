#!/usr/bin/env python3
"""Generate course notebooks and small example input maps.

The Python source is the canonical representation.  Student and solution
notebooks are generated together so their structure cannot drift silently.
"""

from pathlib import Path
from textwrap import dedent

import nbformat as nbf
import numpy as np


COURSE_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = COURSE_ROOT / "notebooks"
DATA_DIR = COURSE_ROOT / "data"


def _text(value):
    return dedent(value).strip() + "\n"


def markdown(source, *, tags=()):
    return {"kind": "markdown", "source": _text(source), "tags": list(tags)}


def code(source, *, tags=()):
    return {"kind": "code", "source": _text(source), "tags": list(tags)}


def answer(prompt, solution):
    return {
        "kind": "answer",
        "student": _text(f"""{prompt}\n\n**Your response:**  \n"""),
        "solution": _text(f"""{prompt}\n\n**Solution.** {solution}"""),
    }


def _render(specs, *, solution, prefix):
    cells = []
    for index, spec in enumerate(specs):
        kind = spec["kind"]
        tags = list(spec.get("tags", ()))
        if kind == "answer":
            source = spec["solution"] if solution else spec["student"]
            tags.append("solution" if solution else "student-response")
            cell = nbf.v4.new_markdown_cell(source, metadata={"tags": tags})
        elif kind == "markdown":
            cell = nbf.v4.new_markdown_cell(spec["source"], metadata={"tags": tags})
        elif kind == "code":
            cell = nbf.v4.new_code_cell(spec["source"], metadata={"tags": tags})
        else:
            raise ValueError(f"Unknown cell kind: {kind}")
        cell["id"] = f"{prefix}-{index:03d}"
        cells.append(cell)
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python", "version": "3"}
    notebook.metadata["course"] = {
        "part": prefix.upper(),
        "solution": bool(solution),
        "package_version": "0.1.4",
    }
    return notebook


def _write_pair(name, prefix, specs):
    student = _render(specs, solution=False, prefix=prefix)
    solution = _render(specs, solution=True, prefix=prefix)
    nbf.write(student, NOTEBOOK_DIR / f"{name}_student.ipynb")
    nbf.write(solution, NOTEBOOK_DIR / f"{name}_solutions.ipynb")


def part_a_specs():
    return [
        markdown(r"""
        # Part A — Waves in a box

        **Working time:** about 3 hours  
        **Work in groups of 2–4.** This notebook is guided and is not submitted.

        ## Learning goals

        By the end you should be able to:

        - configure and run the shallow-water model;
        - predict and measure the long-wave speed (c=\sqrt{gH});
        - identify an incident and reflected wave;
        - carry out a controlled comparison in which only one parameter changes.

        Before beginning, write down one sentence describing what you expect a
        localized elevation of the water surface to do in a closed basin.
        """),
        answer(
            "**Prediction 1.** What will a localized elevation do after the model starts?",
            "It will launch gravity waves. With the velocity chosen below, most of the signal travels eastward. It will reflect from the solid eastern wall and then travel westward.",
        ),
        markdown(r"""
        ## 1. Installation and imports

        The course uses the released PyPI package, not a source checkout. Install
        the standard version in a terminal with:

        ```text
        python -m pip install "shallowwater==0.1.4"
        ```

        Optional acceleration is installed with:

        ```text
        python -m pip install "shallowwater[numba]==0.1.4"
        ```

        The first numba-backed run can pause while functions are compiled. Both
        backends solve the same discrete equations.
        """),
        code("""
        import numpy as np
        import matplotlib.pyplot as plt

        from shallowwater import (
            ModelParams, backend_info, compute_dt_cfl, depth_on_u,
            make_grid, run_model, zero_forcing,
        )

        print(backend_info())
        """),
        markdown(r"""
        ## 2. A controlled right-going pulse

        The model variables are surface displacement (eta) and depth-averaged
        velocities (u,v). The initial velocity below is chosen so most of the
        disturbance travels toward increasing (x), which makes speed and
        reflection easier to measure.

        The model uses solid vertical walls. It does not include wave breaking,
        wetting and drying, or coastal inundation.
        """),
        code("""
        def cross_basin_pulse(grid, params, *, amplitude=0.10, radius=60e3, x0=300e3):
            eta_line = amplitude * np.exp(-((grid.x_c - x0) / radius) ** 2)
            eta = np.repeat(eta_line[None, :], grid.Ny, axis=0)

            H_u = depth_on_u(grid, params.H)
            eta_u = amplitude * np.exp(-((grid.x_u - x0) / radius) ** 2)
            u = np.repeat(eta_u[None, :], grid.Ny, axis=0) * np.sqrt(params.g / H_u)
            v = np.zeros((grid.Ny + 1, grid.Nx))
            return eta, u, v


        def run_uniform_case(*, H=400.0, Lx=1.6e6, Nx=160, tmax_hours=8.0):
            Ny, Ly = 20, 200e3
            grid = make_grid(Nx, Ny, Lx, Ly)
            params = ModelParams(H=H, g=9.81, f0=0.0, beta=0.0, r=0.0, linear=True)
            dt = compute_dt_cfl(grid, params, cfl=0.45)
            out = run_model(
                tmax=tmax_hours * 3600,
                dt=dt,
                grid=grid,
                params=params,
                forcing_fn=zero_forcing,
                ic_fn=lambda g, p: cross_basin_pulse(g, p),
                save_every=4,
                out_vars=("eta", "u"),
            )
            print(
                f"H={H:.0f} m, Lx={Lx/1e3:.0f} km, dx={grid.dx/1e3:.1f} km, "
                f"dt={dt:.1f} s, saved={len(out['time'])}"
            )
            return grid, params, out
        """),
        code("""
        grid, params, out = run_uniform_case()
        eta = np.asarray(out["eta"])
        times = np.asarray(out["time"])
        eta_line = eta.mean(axis=1)

        fig, ax = plt.subplots(figsize=(9, 4))
        image = ax.pcolormesh(grid.x_c / 1e3, times / 3600, eta_line, shading="auto", cmap="RdBu_r")
        ax.set(xlabel="x [km]", ylabel="time [hours]", title="Centreline Hovmöller diagram")
        fig.colorbar(image, ax=ax, label="surface displacement [m]")
        plt.show()
        """),
        answer(
            "**Observation 1.** Identify the incident and reflected branches in the Hovmöller diagram. What happens at the eastern wall?",
            "The first diagonal branch moves toward increasing x and is incident on the eastern wall. After landfall, a branch with the opposite slope moves westward. Surface elevation reflects with the same sign at a rigid wall, while the normal velocity reverses.",
        ),
        markdown(r"""
        ## 3. Predict and measure wave speed

        For a uniform-depth shallow-water wave,

        [
        c_{theory}=\sqrt{gH}.
        ]

        We measure the position of the maximum before the pulse reaches the wall.
        A grid introduces uncertainty of roughly one grid cell in position.
        """),
        code("""
        c_theory = np.sqrt(params.g * float(params.H))
        target_time = 2.5 * 3600
        time_index = int(np.argmin(np.abs(times - target_time)))
        x_initial = 300e3
        x_peak = grid.x_c[np.argmax(eta_line[time_index])]
        c_measured = (x_peak - x_initial) / times[time_index]
        relative_error = abs(c_measured - c_theory) / c_theory

        print(f"theoretical speed = {c_theory:.2f} m/s")
        print(f"measured speed    = {c_measured:.2f} m/s")
        print(f"relative error    = {100*relative_error:.1f} %")
        """),
        answer(
            "**Analysis 1.** Report the theoretical and measured speeds. Are they consistent given the grid spacing?",
            "The theoretical speed for H=400 m is 62.64 m/s. The measured value should be close, normally within a few percent. A one-cell position uncertainty is 10 km, already about 1.8% of the distance travelled in 2.5 hours.",
        ),
        markdown(r"""
        ## 4. Controlled depth experiment

        Change only the depth from 400 m to 900 m. Predict the speed ratio before
        running. Keep the domain and initial disturbance unchanged.
        """),
        answer(
            "**Prediction 2.** What is (c_{900}/c_{400})? Will reflection occur earlier or later?",
            r"The ratio is \(\sqrt{900/400}=1.5\). The deeper case is faster, so it reaches and reflects from the eastern wall earlier.",
        ),
        code("""
        grid_deep, params_deep, out_deep = run_uniform_case(H=900.0, tmax_hours=5.5)
        eta_deep = np.asarray(out_deep["eta"]).mean(axis=1)
        times_deep = np.asarray(out_deep["time"])

        fig, ax = plt.subplots(figsize=(9, 4))
        image = ax.pcolormesh(
            grid_deep.x_c / 1e3, times_deep / 3600, eta_deep,
            shading="auto", cmap="RdBu_r"
        )
        ax.set(xlabel="x [km]", ylabel="time [hours]", title="H = 900 m")
        fig.colorbar(image, ax=ax, label="surface displacement [m]")
        plt.show()
        """),
        answer(
            "**Analysis 2.** Does the numerical comparison support the predicted depth scaling? Give evidence from the plot or a measurement.",
            "Yes. The incident branch is about 1.5 times steeper in x-versus-time coordinates, and the reflection occurs earlier. A measured speed should be close to 94.0 m/s, compared with 62.6 m/s in the baseline.",
        ),
        markdown(r"""
        ## 5. Optional investigations

        If time permits, try one of these while changing only one primary factor:

        - Double the basin length. How does the wall-arrival time change?
        - Move the initial pulse. Which arrival times change and which speeds do not?
        - Change the pulse amplitude. Does speed change in the linear model?
        - Change `Nx` while keeping `Lx` fixed. How does numerical error change?

        ## Checkpoint

        Save a short record of the parameter changed, prediction, observation,
        quantitative evidence, and one model limitation.
        """),
        answer(
            "**Final reflection.** Name one physical conclusion and one limitation of this experiment.",
            "Physical conclusion: long waves propagate faster in deeper water and reflect from a solid coast. Limitation: this is a linear, hydrostatic, depth-averaged model with permanently wet cells, so it cannot represent breaking or inundation.",
        ),
    ]


def part_b_specs():
    return [
        markdown(r"""
        # Part B — A long wave over variable bathymetry

        **Working time:** about 3 hours  
        **Work in groups of 2–4.** This notebook is guided and is not submitted.

        Part B investigates a tsunami-like long disturbance. It is not a hazard
        or inundation model. Every grid cell remains wet, including the final
        coastal cell.

        ## Learning goals

        - Relate local long-wave speed to local water depth.
        - Use virtual gauges to compare arrival time and surface elevation.
        - Conduct one controlled bathymetry experiment.
        - Learn how to change wind forcing, bathymetry, domain size, and initial state.
        - Prepare a focused proposal for Part C.
        """),
        code("""
        from pathlib import Path

        import numpy as np
        import matplotlib.pyplot as plt

        from shallowwater import (
            ModelParams, backend_info, compute_dt_cfl, depth_on_u,
            load_bathymetry, make_grid,
            make_wind_forcing_from_file, run_model, shelf_bathymetry,
            uniform_wind_forcing, zero_forcing,
        )

        print(backend_info())
        DATA_DIR = next(
            path for path in (Path("../data"), Path("data"), Path("MT1562_python_lab_waves/data"))
            if path.exists()
        )
        print("Course data:", DATA_DIR.resolve())
        """),
        markdown(r"""
        ## 1. Build a shelf and make a prediction

        The basin is deep in the west and shallow near the eastern wall. The
        disturbance is broad and nearly one-dimensional, which reduces geometric
        spreading and makes the depth effect easier to isolate.
        """),
        code("""
        Nx, Ny = 160, 24
        Lx, Ly = 2.4e6, 360e3
        grid = make_grid(Nx, Ny, Lx, Ly)
        H = shelf_bathymetry(
            grid, H_deep=3000.0, H_coast=120.0,
            shelf_width=800e3, coast="east", power=1.5,
        )

        fig, ax = plt.subplots(figsize=(9, 3))
        ax.plot(grid.x_c / 1e3, H[Ny // 2], linewidth=2)
        ax.invert_yaxis()
        ax.set(xlabel="x [km]", ylabel="water depth [m]", title="Model bathymetry")
        ax.grid(alpha=0.25)
        plt.show()

        c_deep = np.sqrt(9.81 * 3000.0)
        c_coast = np.sqrt(9.81 * 120.0)
        print(f"deep-water long-wave speed: {c_deep:.1f} m/s")
        print(f"coastal long-wave speed:    {c_coast:.1f} m/s")
        """),
        answer(
            "**Prediction 1.** Where will the wave travel fastest? What changes do you expect as it crosses the shelf?",
            "It travels fastest over the 3000 m deep region and slows toward the 120 m coastal region. With an approximately conserved frequency, its wavelength shortens. Its amplitude can change because of shoaling and partial reflection, but this setup alone does not predict run-up or hazard.",
        ),
        code("""
        def cross_basin_pulse(grid, params, *, amplitude=0.12, radius=80e3, x0=350e3):
            eta_line = amplitude * np.exp(-((grid.x_c - x0) / radius) ** 2)
            eta = np.repeat(eta_line[None, :], grid.Ny, axis=0)
            H_u = depth_on_u(grid, params.H)
            eta_u = amplitude * np.exp(-((grid.x_u - x0) / radius) ** 2)
            u = np.repeat(eta_u[None, :], grid.Ny, axis=0) * np.sqrt(params.g / H_u)
            v = np.zeros((grid.Ny + 1, grid.Nx))
            return eta, u, v


        def run_shelf_case(H_coast, *, tmax_hours=7.0):
            depth = shelf_bathymetry(
                grid, H_deep=3000.0, H_coast=H_coast,
                shelf_width=800e3, coast="east", power=1.5,
            )
            params = ModelParams(H=depth, g=9.81, f0=0.0, beta=0.0, r=0.0, linear=True)
            dt = compute_dt_cfl(grid, params, cfl=0.42)
            out = run_model(
                tmax=tmax_hours * 3600,
                dt=dt,
                grid=grid,
                params=params,
                forcing_fn=zero_forcing,
                ic_fn=lambda g, p: cross_basin_pulse(g, p),
                save_every=5,
                out_vars=("eta",),
            )
            return params, out


        params_120, out_120 = run_shelf_case(120.0)
        eta_120 = np.asarray(out_120["eta"])
        times_120 = np.asarray(out_120["time"])
        eta_line_120 = eta_120.mean(axis=1)
        """),
        code("""
        fig, ax = plt.subplots(figsize=(9, 4))
        image = ax.pcolormesh(
            grid.x_c / 1e3, times_120 / 3600, eta_line_120,
            shading="auto", cmap="RdBu_r"
        )
        ax.set(xlabel="x [km]", ylabel="time [hours]", title="Wave crossing the shelf")
        fig.colorbar(image, ax=ax, label="surface displacement [m]")
        plt.show()
        """),
        markdown(r"""
        ## 2. Virtual gauges

        Four gauges sample the deep basin, the shelf break, the slope, and the
        coastal region. The largest peak is not always the first arrival, so use
        both the Hovmöller diagram and the time series.
        """),
        code("""
        gauge_x_km = [700, 1400, 1900, 2250]
        gauge_indices = [int(np.argmin(abs(grid.x_c / 1e3 - x))) for x in gauge_x_km]

        fig, ax = plt.subplots(figsize=(9, 4))
        for x_km, index in zip(gauge_x_km, gauge_indices):
            ax.plot(times_120 / 3600, eta_line_120[:, index], label=f"{x_km} km")
        ax.set(xlabel="time [hours]", ylabel="surface displacement [m]", title="Virtual gauges")
        ax.legend(ncol=2)
        ax.grid(alpha=0.25)
        plt.show()

        peak_times = []
        for x_km, index in zip(gauge_x_km, gauge_indices):
            signal = eta_line_120[:, index]
            peak_index = int(np.argmax(signal))
            peak_times.append(times_120[peak_index] / 3600)
            print(f"gauge {x_km:4.0f} km: largest positive peak at {peak_times[-1]:.2f} h, "
                  f"eta={signal[peak_index]:.3f} m")
        """),
        answer(
            "**Analysis 1.** Use the plots and gauge records to explain how depth affected propagation. Include at least one number.",
            "The Hovmöller branch is steep in the deep region and becomes less steep over the shelf, showing that propagation slows. The theoretical limiting speeds are about 172 m/s offshore and 34 m/s at the coast. Gauge peak times increase non-uniformly because the final part of the path crosses much shallower water.",
        ),
        markdown(r"""
        ## 3. Controlled coastal-depth experiment

        Repeat the case with a 300 m coastal depth. Everything else remains the
        same. Compare arrival time and the modeled elevation at the final gauge.
        """),
        answer(
            "**Prediction 2.** Will the 300 m coastal case arrive earlier or later than the 120 m case? What do you expect for elevation?",
            r"It should arrive earlier because \(\sqrt{gH}\) is larger over the coastal part of the path. The shallower case may show a larger surface elevation, but reflection and finite shelf geometry also influence the result, so this should be measured rather than assumed.",
        ),
        code("""
        params_300, out_300 = run_shelf_case(300.0)
        eta_line_300 = np.asarray(out_300["eta"]).mean(axis=1)
        times_300 = np.asarray(out_300["time"])
        coastal_index = gauge_indices[-1]

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(times_120 / 3600, eta_line_120[:, coastal_index], label="coastal depth 120 m")
        ax.plot(times_300 / 3600, eta_line_300[:, coastal_index], label="coastal depth 300 m")
        ax.set(xlabel="time [hours]", ylabel="surface displacement [m]",
               title="Near-coast model cell: controlled comparison")
        ax.legend()
        ax.grid(alpha=0.25)
        plt.show()

        for label, times, signal in (
            ("120 m", times_120, eta_line_120[:, coastal_index]),
            ("300 m", times_300, eta_line_300[:, coastal_index]),
        ):
            index = int(np.argmax(signal))
            print(f"{label}: peak time={times[index]/3600:.2f} h, peak eta={signal[index]:.3f} m")
        """),
        answer(
            "**Analysis 2.** Summarize the controlled comparison. Why must it not be interpreted as a prediction of coastal danger?",
            "The 300 m case reaches the coastal gauge earlier, consistent with its larger local wave speed. The peak elevations differ because depth changes shoaling and reflection. The comparison is not a hazard prediction because the model has no dry land, run-up, breaking, inundation, buildings, or realistic source/bathymetry.",
        ),
        markdown(r"""
        ## 4. Bathymetry supplied as a file

        The package accepts an in-memory depth array directly. Version 0.1.4 also
        provides a validated loader for `.npy`, `.npz`, `.csv`, and `.txt` maps.
        Maps must match the grid exactly, use positive depth in metres, and use
        array order `(y, x)`. No interpolation or land mask is applied.
        """),
        code("""
        H_file = load_bathymetry(DATA_DIR / "example_bathymetry.npz", grid)
        print("shape:", H_file.shape, "minimum depth:", H_file.min(), "maximum depth:", H_file.max())

        fig, ax = plt.subplots(figsize=(8, 3.5))
        image = ax.imshow(
            H_file, origin="lower", extent=[0, grid.Lx/1e3, 0, grid.Ly/1e3],
            aspect="auto", cmap="Blues"
        )
        ax.set(xlabel="x [km]", ylabel="y [km]", title="Bathymetry loaded from file")
        fig.colorbar(image, ax=ax, label="depth [m]")
        plt.show()
        """),
        answer(
            "**Check.** Describe one difference between the analytic shelf and the file-loaded map.",
            "The analytic shelf varies only with x, whereas the supplied file includes a two-dimensional ridge/channel pattern. Both remain fully wet and have exactly the grid's cell-centre shape.",
        ),
        markdown(r"""
        ## 5. Wind forcing supplied as a file

        A forcing file contains static cell-centred wind-stress maps `tau_x` and
        `tau_y` in N m(^{-2}). The file is read once and converted to a forcing
        function. A separate time envelope controls ramp-up or shut-off.
        """),
        code("""
        wind_from_file = make_wind_forcing_from_file(
            DATA_DIR / "example_wind_forcing.npz",
            grid,
            envelope=lambda t: min(1.0, t / (2 * 3600)),
        )
        taux, tauy, _ = wind_from_file(2 * 3600, grid, params_120)

        fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), constrained_layout=True)
        for ax, field, title in zip(
            axes, (taux[:, :-1], tauy[:-1, :]), ("eastward stress", "northward stress")
        ):
            image = ax.imshow(field, origin="lower", aspect="auto", cmap="RdBu_r")
            ax.set_title(title)
            fig.colorbar(image, ax=ax, label="N m$^{-2}$")
        plt.show()
        """),
        markdown(r"""
        ### Short wind-forced demonstration

        Here the model starts from rest. An eastward wind ramps up, is switched
        off after six hours, and pushes water toward the eastern wall.
        """),
        code("""
        wind_params = ModelParams(
            H=H, g=9.81, f0=0.0, beta=0.0,
            r=1/(2*86400), linear=True,
        )
        wind_dt = compute_dt_cfl(grid, wind_params, cfl=0.42)
        wind_forcing = lambda t, g, p: uniform_wind_forcing(
            t, g, p, tau_x=0.08, tau_y=0.0,
            t_ramp=2*3600, t_off=6*3600,
        )
        wind_out = run_model(
            tmax=8*3600, dt=wind_dt, grid=grid, params=wind_params,
            forcing_fn=wind_forcing,
            ic_fn=lambda g, p: (
                np.zeros((g.Ny, g.Nx)),
                np.zeros((g.Ny, g.Nx+1)),
                np.zeros((g.Ny+1, g.Nx)),
            ),
            save_every=8, out_vars=("eta",),
        )
        wind_eta = np.asarray(wind_out["eta"])
        wind_times = np.asarray(wind_out["time"])
        shutoff_index = int(np.argmin(abs(wind_times - 6*3600)))

        fig, ax = plt.subplots(figsize=(9, 3.5))
        ax.plot(
            grid.x_c/1e3, wind_eta[shutoff_index].mean(axis=0),
            label="near wind shut-off (setup)",
        )
        ax.plot(
            grid.x_c/1e3, wind_eta[-1].mean(axis=0),
            label="two hours later (free response)",
        )
        ax.axhline(0, color="0.4", linewidth=0.8)
        ax.set(xlabel="x [km]", ylabel="surface displacement [m]",
               title="Wind setup and release")
        ax.legend()
        ax.grid(alpha=0.25)
        plt.show()
        """),
        answer(
            "**Observation 3.** Which coast gains water under eastward wind? What happens after the wind stops?",
            "Water piles up toward the eastern wall and is lowered toward the west. After shut-off, the displaced surface is no longer in equilibrium and launches free basin oscillations.",
        ),
        markdown(r"""
        ## 6. Controls available in Part C

        You may investigate one main factor:

        | Category | Examples of controls |
        |---|---|
        | Wind | amplitude, direction, duration, spatial map |
        | Bottom | uniform depth, shelf depth/width, ridge, file map, damping |
        | Domain | `Lx`, `Ly`, aspect ratio, resolution |
        | Initial state | amplitude, radius, position, circular or cross-basin shape |
        | Optional | forcing period or rotation |

        Changing domain size while keeping `Nx` fixed also changes `dx`; always
        report both. Change one primary factor unless the instructor approves a
        two-factor experiment.

        ## 7. Part C proposal
        """),
        answer(
            "**Project question.** State a question that can be answered with a baseline and two controlled variations.",
            "Example: How does the duration of an eastward wind event affect maximum surface setup at the eastern wall?",
        ),
        answer(
            "**Prediction and mechanism.** What do you expect, and which physical argument supports it?",
            "Example: Longer wind duration should initially increase eastern setup because stress accelerates water toward the wall, although free oscillations and damping may prevent a simple proportional response.",
        ),
        answer(
            "**Experimental design.** List the baseline, two variations, the primary parameter, and the diagnostic you will use.",
            "Example: use identical 3 h, 6 h, and 9 h eastward wind events; keep grid, depth, stress, damping, and initial rest state fixed; compare maximum eta in the final four eastern columns and the post-shutoff gauge record.",
        ),
    ]


def part_c_specs():
    return [
        markdown(r"""
        # Part C — Group investigation

        **This is the notebook your group submits.** Rename it with your group
        identifier. Keep the numerical outputs that support your conclusions,
        but do not embed large GIF or video files.

        ## Assessment

        | Criterion | Weight |
        |---|---:|
        | Focused question, prediction, and controlled design | 25% |
        | Appropriate quantitative evidence | 30% |
        | Physical interpretation | 25% |
        | Limitations and reproducibility | 10% |
        | Clear notebook and contribution statement | 10% |

        Use one main independent variable. A two-factor study requires instructor
        approval. Your notebook must run from top to bottom before submission.
        """),
        markdown("""
        ## Group and research question

        **Group members:**  

        **Research question:**  

        **Why this question matters for long surface waves:**  
        """),
        markdown("""
        ## Prediction

        State your expected result **before** presenting simulation results. Name
        the physical mechanism or scaling that supports it.

        **Prediction:**  
        """),
        markdown(r"""
        ## Model scope

        This one-layer shallow-water model is hydrostatic and depth averaged. In
        the default linear configuration it assumes small surface displacement.
        All cells remain wet; it does not represent breaking, run-up, inundation,
        or a resolved bottom boundary layer. Rayleigh damping is an idealized
        energy-loss parameter.
        """),
        code("""
        from pathlib import Path

        import numpy as np
        import matplotlib.pyplot as plt

        from shallowwater import (
            ModelParams, backend_info, compute_dt_cfl, depth_on_u,
            load_bathymetry, make_grid, make_wind_forcing_from_file,
            run_model, shelf_bathymetry, uniform_wind_forcing, zero_forcing,
        )

        print(backend_info())
        """),
        markdown(r"""
        ## Reusable experiment function

        The function below makes the controls explicit. `depth` may be a positive
        scalar, an `(Ny, Nx)` array, or a function that creates such an array from
        the grid. Initial-state choices are `cross_pulse`, `gaussian`, and `rest`.
        Uniform wind is controlled by `wind_x`, `wind_y`, `wind_ramp_hours`, and
        `wind_off_hours`.

        If you use a custom external file, the submitted notebook must explain
        how it can be reproduced. Prefer generating arrays in this notebook or
        using the supplied course files; do not use an unexplained absolute path.
        """),
        code("""
        def make_initial_state(
            grid, params, *, kind="cross_pulse", amplitude=0.08,
            radius=60e3, x_fraction=0.25, y_fraction=0.50,
        ):
            x0, y0 = x_fraction * grid.Lx, y_fraction * grid.Ly
            eta = np.zeros((grid.Ny, grid.Nx))
            u = np.zeros((grid.Ny, grid.Nx + 1))
            v = np.zeros((grid.Ny + 1, grid.Nx))
            X, Y = np.meshgrid(grid.x_c, grid.y_c)

            if kind == "rest":
                return eta, u, v
            if kind == "gaussian":
                eta = amplitude * np.exp(-((X-x0)**2 + (Y-y0)**2) / radius**2)
                return eta, u, v
            if kind == "cross_pulse":
                eta_line = amplitude * np.exp(-((grid.x_c-x0) / radius)**2)
                eta = np.repeat(eta_line[None, :], grid.Ny, axis=0)
                H_u = depth_on_u(grid, params.H)
                eta_u = amplitude * np.exp(-((grid.x_u-x0) / radius)**2)
                u = np.repeat(eta_u[None, :], grid.Ny, axis=0) * np.sqrt(params.g/H_u)
                return eta, u, v
            raise ValueError("kind must be 'cross_pulse', 'gaussian', or 'rest'")


        def run_case(
            label,
            *,
            Nx=120, Ny=24, Lx=1.2e6, Ly=240e3,
            depth=400.0, f0=0.0, damping=0.0,
            initial_kind="cross_pulse", initial_amplitude=0.08,
            initial_radius=60e3, initial_x_fraction=0.25,
            wind_x=0.0, wind_y=0.0, wind_ramp_hours=1.0,
            wind_off_hours=None, wind_file=None,
            tmax_hours=5.0,
        ):
            grid = make_grid(Nx, Ny, Lx, Ly)
            H = depth(grid) if callable(depth) else depth
            if isinstance(H, (str, Path)):
                H = load_bathymetry(H, grid)
            params = ModelParams(
                H=H, g=9.81, f0=f0, beta=0.0,
                r=damping, linear=True,
            )
            dt = compute_dt_cfl(grid, params, cfl=0.42)

            if wind_file is not None:
                envelope = lambda t: min(1.0, t/(wind_ramp_hours*3600))
                forcing = make_wind_forcing_from_file(wind_file, grid, envelope=envelope)
            elif wind_x != 0.0 or wind_y != 0.0:
                t_off = None if wind_off_hours is None else wind_off_hours * 3600
                forcing = lambda t, g, p: uniform_wind_forcing(
                    t, g, p, tau_x=wind_x, tau_y=wind_y,
                    t_ramp=wind_ramp_hours*3600, t_off=t_off,
                )
            else:
                forcing = zero_forcing

            initial = lambda g, p: make_initial_state(
                g, p, kind=initial_kind, amplitude=initial_amplitude,
                radius=initial_radius, x_fraction=initial_x_fraction,
            )
            out = run_model(
                tmax=tmax_hours*3600, dt=dt, grid=grid, params=params,
                forcing_fn=forcing, ic_fn=initial, save_every=5,
                out_vars=("eta",),
            )
            print(
                f"{label}: Nx={Nx}, Ny={Ny}, Lx={Lx/1e3:.0f} km, "
                f"dx={grid.dx/1e3:.1f} km, dt={dt:.1f} s, frames={len(out['time'])}"
            )
            return {"label": label, "grid": grid, "params": params, "out": out}
        """),
        markdown("""
        ## Baseline configuration

        Describe the baseline and justify the domain, depth, initial state,
        forcing, duration, and resolution. The default below runs successfully;
        replace it with the baseline appropriate to your question.

        **Baseline justification:**  
        """),
        code("""
        baseline = run_case(
            "baseline",
            depth=400.0,
            initial_kind="cross_pulse",
            tmax_hours=5.0,
        )
        """),
        markdown("""
        ## Controlled variations

        Add at least two cases. Change one primary parameter while keeping the
        remaining controls fixed. Examples include wind duration, shelf width,
        domain length, or initial radius.

        **Independent variable:**  

        **Values tested:**  

        **Controls held fixed:**  
        """),
        code("""
        # Replace these examples with cases that answer your research question.
        # variation_1 = run_case("variation 1", depth=...)
        # variation_2 = run_case("variation 2", depth=...)
        cases = [baseline]  # add variation_1 and variation_2
        """),
        markdown("""
        ## Diagnostics

        Use at least one quantitative diagnostic, not only an animation. Suitable
        choices include arrival time, measured propagation speed, maximum surface
        elevation at a stated location, reflection time, oscillation period, or
        a comparison with a theoretical scaling.
        """),
        code("""
        def plot_hovmoller(case):
            grid, out = case["grid"], case["out"]
            eta_line = np.asarray(out["eta"]).mean(axis=1)
            times = np.asarray(out["time"])
            fig, ax = plt.subplots(figsize=(8.5, 3.8))
            image = ax.pcolormesh(
                grid.x_c/1e3, times/3600, eta_line,
                shading="auto", cmap="RdBu_r",
            )
            ax.set(xlabel="x [km]", ylabel="time [hours]", title=case["label"])
            fig.colorbar(image, ax=ax, label="surface displacement [m]")
            plt.show()


        for case in cases:
            plot_hovmoller(case)
        """),
        code("""
        # Add your quantitative measurement and comparison table here.
        summary = []
        for case in cases:
            eta = np.asarray(case["out"]["eta"])
            summary.append((case["label"], float(np.max(np.abs(eta)))))

        print("case | maximum absolute surface displacement [m]")
        for label, value in summary:
            print(f"{label:20s} | {value:.4f}")
        """),
        markdown("""
        ## Results

        Present the evidence needed to answer the question. Every figure needs
        labelled axes with units, a useful caption or nearby explanation, and a
        statement identifying what should be noticed.

        **Results and figure interpretation:**  
        """),
        markdown(r"""
        ## Comparison with theory

        Compare at least one result with a relevant prediction or scaling, such
        as (c=\sqrt{gH}), a crossing time (L/c), a basin period (2L/c),
        linear amplitude scaling, or a force/damping timescale.

        **Theoretical comparison, including units and discrepancy:**  
        """),
        markdown("""
        ## Limitations

        Discuss at least two limitations relevant to your particular conclusion.
        Distinguish limitations of the physical model from numerical resolution
        or experimental-design limitations.

        **Limitations:**  
        """),
        markdown("""
        ## Conclusion

        Answer the research question directly in a short paragraph. State whether
        the prediction was supported and cite the main quantitative evidence.

        **Conclusion:**  
        """),
        markdown("""
        ## Contributions and submission check

        **Member contributions:**  

        Before submission:

        - [ ] All group members are named.
        - [ ] The notebook uses no unexplained absolute file paths.
        - [ ] Required figures and numerical outputs are visible.
        - [ ] Large animations have been removed.
        - [ ] The kernel was restarted and all cells ran in order without error.
        - [ ] The filename contains the group identifier.
        """),
    ]


def build_example_data():
    Nx, Ny = 160, 24
    Lx, Ly = 2.4e6, 360e3
    dx, dy = Lx / Nx, Ly / Ny
    x = (np.arange(Nx) + 0.5) * dx
    y = (np.arange(Ny) + 0.5) * dy
    X, Y = np.meshgrid(x, y)

    shelf = 250.0 + 1750.0 * np.clip((Lx - X) / (0.65 * Lx), 0.0, 1.0) ** 1.4
    ridge = 550.0 * np.exp(-((X - 1.45e6) / 190e3) ** 2) * (
        0.55 + 0.45 * np.cos(2 * np.pi * (Y - Ly / 2) / Ly)
    )
    H = np.maximum(80.0, shelf - ridge)
    np.savez(
        DATA_DIR / "example_bathymetry.npz",
        H=H,
        x=x,
        y=y,
        units=np.array("m"),
        description=np.array("Synthetic fully wet shelf with a two-dimensional ridge"),
    )

    patch = np.exp(-((X - 1.25e6) / 520e3) ** 2 - ((Y - 0.55 * Ly) / 130e3) ** 2)
    tau_x = 0.12 * patch
    tau_y = 0.035 * np.sin(2 * np.pi * Y / Ly) * patch
    np.savez(
        DATA_DIR / "example_wind_forcing.npz",
        tau_x=tau_x,
        tau_y=tau_y,
        units=np.array("N m-2"),
        description=np.array("Synthetic static wind-stress pattern"),
    )


def main():
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _write_pair("part_a_waves", "a", part_a_specs())
    _write_pair("part_b_bathymetry", "b", part_b_specs())
    nbf.write(_render(part_c_specs(), solution=False, prefix="c"), NOTEBOOK_DIR / "part_c_project.ipynb")
    build_example_data()
    print(f"Wrote notebooks to {NOTEBOOK_DIR}")
    print(f"Wrote example data to {DATA_DIR}")


if __name__ == "__main__":
    main()
