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

        **Working time:** one 90-minute block
        **Work in groups of 2–4.** This notebook is guided and is not submitted.

        ## Learning goals

        By the end you should be able to:

        - configure and run the shallow-water model;
        - predict and measure the long-wave speed $c=\sqrt{gH}$;
        - identify an incident and reflected wave;
        - carry out a controlled comparison in which only one parameter changes.

        Start with the animation below: watch first, then use the Hovmöller
        diagram and a speed measurement to explain what you saw.
        """),
        answer(
            "**Prediction 1.** What will a localized elevation do after the model starts?",
            "It launches gravity waves that spread away from the initial bump. They reflect from the four solid walls and cross the basin again.",
        ),
        markdown(r"""
        ## 1. Environment check and imports

        Before the lab, follow `INSTALLATION.md` to create the `shallowwater-lab`
        Miniconda environment and select it as the kernel in VS Code. The package
        command used there is:

        ```text
        python -m pip install "shallowwater==0.1.4"
        ```

        Optional acceleration is installed with:

        ```text
        python -m pip install "shallowwater[numba]==0.1.4"
        ```

        Numba is optional. Its first run can pause while functions are compiled.
        Ask for help now if the import cell fails; do not change model code to
        repair an installation problem.
        """),
        code("""
        from pathlib import Path

        import numpy as np
        import matplotlib.pyplot as plt

        from shallowwater import (
            ModelParams, animate_eta, backend_info, compute_dt_cfl, depth_on_u,
            make_grid, run_model, setup_initial_state, zero_forcing,
        )

        print(backend_info())
        """),
        markdown(r"""
        ## 2. First run: watch a wave spread and reflect

        The initial state is a circular bump in surface elevation with zero
        velocity. Run the next two cells. The animation appears directly below
        the cell and is also saved as a GIF in the course `animations/` folder.

        The helper deliberately shows only about 40 evenly spaced frames. You
        can reuse it later with any model output that contains `eta`.
        """),
        code("""
        def find_course_root():
            candidates = (
                Path.cwd(),
                Path.cwd().parent,
                Path.cwd() / "MT1562_python_lab_waves",
            )
            return next(path for path in candidates if (path / "notebooks").exists())


        COURSE_ROOT = find_course_root()
        ANIMATION_DIR = COURSE_ROOT / "animations"
        ANIMATION_DIR.mkdir(exist_ok=True)


        def animate_and_save(out, grid, filename, *, title, max_frames=42):
            # Display a compact eta animation and save the same frames as a GIF.
            frame_count = len(out["time"])
            frames = np.unique(
                np.linspace(0, frame_count - 1, min(max_frames, frame_count), dtype=int)
            )
            animation = animate_eta(
                out, grid, frames=frames, interval=100, repeat=True,
                title=title, contours=False, remove_mean=False,
                figsize=(8.5, 3.6),
            )
            path = ANIMATION_DIR / filename
            animation.save(str(path), fps=10, dpi=85)
            print("Saved animation:", path.resolve())
            return animation
        """),
        code("""
        visual_grid = make_grid(96, 56, 1.2e6, 700e3)
        visual_params = ModelParams(
            H=500.0, g=9.81, f0=0.0, beta=0.0, r=0.0, linear=True,
        )
        visual_dt = compute_dt_cfl(visual_grid, visual_params, cfl=0.45)
        visual_out = run_model(
            tmax=5.0*3600, dt=visual_dt,
            grid=visual_grid, params=visual_params,
            forcing_fn=zero_forcing,
            ic_fn=lambda g, p: setup_initial_state(
                g, p, mode="gaussian_bump", amp=0.12, R=70e3,
                x0=0.35*g.Lx, y0=0.55*g.Ly,
            ),
            save_every=4, out_vars=("eta",),
        )

        first_wave_animation = animate_and_save(
            visual_out, visual_grid, "part_a_first_wave.gif",
            title="Circular shallow-water wave: propagation and reflection",
        )
        first_wave_animation
        """),
        answer(
            "**Observation 1.** Describe two things you saw before and after the first wall reflection.",
            "Before reflection, the initially circular disturbance expands outward. The nearest walls are reached first; reflected wave fronts then reverse their normal direction and interfere with waves arriving from other walls.",
        ),
        markdown(r"""
        ## 3. A controlled right-going pulse

        The model variables are surface displacement $\eta$ and depth-averaged
        velocities $(u,v)$. The initial velocity below is chosen so most of the
        disturbance travels toward increasing $x$, which makes speed and
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
        code("""
        uniform_animation = animate_and_save(
            out, grid, "part_a_uniform_400m.gif",
            title="Right-going pulse and reflection: H = 400 m",
        )
        uniform_animation
        """),
        answer(
            "**Observation 2.** Identify the incident and reflected branches in the Hovmöller diagram. What happens at the eastern wall?",
            "The first diagonal branch moves toward increasing x and is incident on the eastern wall. After landfall, a branch with the opposite slope moves westward. Surface elevation reflects with the same sign at a rigid wall, while the normal velocity reverses.",
        ),
        markdown(r"""
        ## 4. Predict and measure wave speed

        For a uniform-depth shallow-water wave,

        $$
        c_{theory}=\sqrt{gH}.
        $$

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
            "The theoretical speed for $H=400$ m is 62.64 m/s. The measured value should be close, normally within a few percent. A one-cell position uncertainty is 10 km, already about 1.8% of the distance travelled in 2.5 hours.",
        ),
        markdown(r"""
        ## 5. Controlled depth experiment

        Change only the depth from 400 m to 900 m. Predict the speed ratio before
        running. Keep the domain and initial disturbance unchanged.
        """),
        answer(
            "**Prediction 2.** What is $c_{900}/c_{400}$? Will reflection occur earlier or later?",
            r"The ratio is $\sqrt{900/400}=1.5$. The deeper case is faster, so it reaches and reflects from the eastern wall earlier.",
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
        code("""
        deep_animation = animate_and_save(
            out_deep, grid_deep, "part_a_uniform_900m.gif",
            title="Right-going pulse and reflection: H = 900 m",
        )
        deep_animation
        """),
        answer(
            "**Analysis 2.** Does the numerical comparison support the predicted depth scaling? Give evidence from the plot or a measurement.",
            "Yes. The incident branch is about 1.5 times steeper in x-versus-time coordinates, and the reflection occurs earlier. A measured speed should be close to 94.0 m/s, compared with 62.6 m/s in the baseline.",
        ),
        markdown(r"""
        ## 6. Optional investigations

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

        **Working time:** one 90-minute block
        **Work in groups of 2–4.** This notebook is guided and is not submitted.

        Part B investigates a tsunami-like long disturbance. It is not a hazard
        or inundation model. Every grid cell remains wet, including the final
        coastal cell.

        ## Learning goals

        - Relate local long-wave speed to local water depth.
        - Use virtual gauges to compare arrival time and surface elevation.
        - Conduct one controlled bathymetry experiment.
        - Observe wind setup and the free response after wind shut-off.
        """),
        code("""
        from pathlib import Path

        import numpy as np
        import matplotlib.pyplot as plt

        from shallowwater import (
            ModelParams, animate_eta, backend_info, compute_dt_cfl, depth_on_u,
            make_grid, run_model, shelf_bathymetry, uniform_wind_forcing,
            zero_forcing,
        )

        print(backend_info())


        def find_course_root():
            candidates = (
                Path.cwd(),
                Path.cwd().parent,
                Path.cwd() / "MT1562_python_lab_waves",
            )
            return next(path for path in candidates if (path / "notebooks").exists())


        COURSE_ROOT = find_course_root()
        ANIMATION_DIR = COURSE_ROOT / "animations"
        ANIMATION_DIR.mkdir(exist_ok=True)
        """),
        code("""
        def animate_and_save(out, grid, filename, *, title, max_frames=42):
            # Display a compact eta animation and save the same frames as a GIF.
            frame_count = len(out["time"])
            frames = np.unique(
                np.linspace(0, frame_count - 1, min(max_frames, frame_count), dtype=int)
            )
            animation = animate_eta(
                out, grid, frames=frames, interval=100, repeat=True,
                title=title, contours=False, remove_mean=False,
                figsize=(8.5, 3.6),
            )
            path = ANIMATION_DIR / filename
            animation.save(str(path), fps=10, dpi=85)
            print("Saved animation:", path.resolve())
            return animation
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
        code("""
        shelf_animation = animate_and_save(
            out_120, grid, "part_b_shelf_120m.gif",
            title="Long wave crossing a shelf: coastal depth = 120 m",
        )
        shelf_animation
        """),
        markdown(r"""
        ### A dispersive-looking wake — what is it?

        The leading pulse develops a wavetrain as it crosses the variable bottom.
        That is worth noticing: uniform-depth *linear shallow-water theory* is
        non-dispersive because all long wavelengths have speed $c=\sqrt{gH}$.
        Here the changing bathymetry scatters and partially reflects the wave, so
        different spatial components interfere and the signal spreads. The
        finite-difference grid can also add **numerical dispersion**, especially
        for features represented by only a few cells. This model does not contain
        the full finite-depth surface-wave dispersion relation from the lecture.

        A useful diagnostic is to repeat the case with a broader initial pulse or
        a finer grid. A wake that changes strongly with resolution is numerical;
        a robust wake tied to the shelf is evidence of topographic scattering.
        """),
        answer(
            "**Observation 1.** In the animation, where does the pulse first develop a visible wake, and which two mechanisms could contribute?",
            "The wake becomes clearest as the pulse reaches and crosses the slope. Partial reflection and interference caused by the variable bathymetry contribute physically, while the discretized model may add numerical dispersion at short resolved scales.",
        ),
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
            r"It should arrive earlier because $\sqrt{gH}$ is larger over the coastal part of the path. The shallower case may show a larger surface elevation, but reflection and finite shelf geometry also influence the result, so this should be measured rather than assumed.",
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
        code("""
        shelf_300_animation = animate_and_save(
            out_300, grid, "part_b_shelf_300m.gif",
            title="Controlled comparison: coastal depth = 300 m",
        )
        shelf_300_animation
        """),
        answer(
            "**Analysis 2.** Summarize the controlled comparison. Why must it not be interpreted as a prediction of coastal danger?",
            "The 300 m case reaches the coastal gauge earlier, consistent with its larger local wave speed. The peak elevations differ because depth changes shoaling and reflection. The comparison is not a hazard prediction because the model has no dry land, run-up, breaking, inundation, buildings, or realistic source/bathymetry.",
        ),
        markdown(r"""
        ## 4. Short wind-forced demonstration

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
        code("""
        wind_animation = animate_and_save(
            wind_out, grid, "part_b_wind_setup_release.gif",
            title="Wind setup followed by a free basin response",
        )
        wind_animation
        """),
        answer(
            "**Observation 2.** Which coast gains water under eastward wind? What happens after the wind stops?",
            "Water piles up toward the eastern wall and is lowered toward the west. After shut-off, the displaced surface is no longer in equilibrium and launches free basin oscillations.",
        ),
    ]


def part_c_project_description_specs():
    return [
        markdown(r"""
        # Part C — Project description and toolbox

        **Demonstration time:** about 45 minutes

        This notebook is an instructor-led tour of the controls available for
        the group project. It is a reference, not a worksheet and not a report
        template. The two demonstrations show how an input map becomes a model
        experiment and how to keep the mapped information visible in an
        animation.

        The project itself should still be a controlled experiment: start from
        one baseline, vary one primary factor, and support the conclusion with a
        quantitative diagnostic rather than an animation alone.
        """),
        markdown(r"""
        ## 1. From a physical question to a numerical experiment

        A useful project has four connected pieces:

        1. a focused question and a prediction based on a mechanism;
        2. a baseline configuration that runs successfully;
        3. one or, if time permits, two controlled variations of one parameter;
        4. a diagnostic such as speed, arrival time, amplitude, period, or
           reflection time.

        The model is a fully wet, one-layer shallow-water model. A coastline is
        a rigid wall; there is no run-up, inundation, breaking, or wetting and
        drying. File-backed fields must already match the model grid.
        """),
        markdown(r"""
        ## 2. Controls available for Part C

        | Category | Examples of controls |
        |---|---|
        | Wind | stress amplitude, direction, duration, spatial map |
        | Bottom | uniform depth, shelf depth/width, ridge, file map |
        | Domain | `Lx`, `Ly`, aspect ratio, `Nx`, `Ny` |
        | Initial state | amplitude, radius, position, circular or cross-basin shape |
        | Rotation | constant Coriolis parameter `f` |
        | Dissipation | Rayleigh damping timescale |

        Domain size and resolution are separate controls. Changing `Lx` while
        keeping `Nx` fixed also changes `dx`, so both must be reported. Rotation
        matters only if the integration time is a meaningful fraction of the
        inertial period $2\pi/|f|$ or the domain is comparable to the deformation
        radius $\sqrt{gH}/|f|$.
        """),
        markdown(r"""
        ## 3. Imports and reusable display helper

        `animate_with_overlay(...)` uses the same `animate_eta` function as Parts
        A and B. An optional drawing function adds persistent contours or arrows
        before the GIF is saved. This is useful for keeping the field that causes
        the response visible behind the evolving surface elevation $\eta$.
        """),
        code("""
        from pathlib import Path

        import numpy as np
        import matplotlib.pyplot as plt

        from shallowwater import (
            ModelParams, animate_eta, backend_info, compute_dt_cfl, depth_on_u,
            load_bathymetry, make_grid, make_wind_forcing_from_file,
            run_model, zero_forcing,
        )

        print(backend_info())


        def find_course_root():
            candidates = (
                Path.cwd(),
                Path.cwd().parent,
                Path.cwd() / "MT1562_python_lab_waves",
            )
            return next(path for path in candidates if (path / "notebooks").exists())


        COURSE_ROOT = find_course_root()
        DATA_DIR = COURSE_ROOT / "data"
        ANIMATION_DIR = COURSE_ROOT / "animations"
        ANIMATION_DIR.mkdir(exist_ok=True)
        print("Course data:", DATA_DIR.resolve())
        """),
        code("""
        def animate_with_overlay(
            out, grid, filename, *, title, draw_overlay=None, max_frames=42,
        ):
            frame_count = len(out["time"])
            frames = np.unique(
                np.linspace(0, frame_count - 1, min(max_frames, frame_count), dtype=int)
            )
            animation = animate_eta(
                out, grid, frames=frames, interval=100, repeat=True,
                title=title, contours=False, remove_mean=False,
                figsize=(9.0, 3.8),
            )
            if draw_overlay is not None:
                draw_overlay(animation.figure.axes[0])
            path = ANIMATION_DIR / filename
            animation.save(str(path), fps=10, dpi=90)
            print("Saved animation:", path.resolve())
            return animation


        def cross_basin_pulse(
            grid, params, *, amplitude=0.10, radius=80e3, x0=350e3,
        ):
            eta_line = amplitude * np.exp(-((grid.x_c - x0) / radius) ** 2)
            eta = np.repeat(eta_line[None, :], grid.Ny, axis=0)
            H_u = depth_on_u(grid, params.H)
            eta_u = amplitude * np.exp(-((grid.x_u - x0) / radius) ** 2)
            u = np.repeat(eta_u[None, :], grid.Ny, axis=0) * np.sqrt(params.g/H_u)
            v = np.zeros((grid.Ny + 1, grid.Nx))
            return eta, u, v


        def rest_state(grid, params):
            return (
                np.zeros((grid.Ny, grid.Nx)),
                np.zeros((grid.Ny, grid.Nx + 1)),
                np.zeros((grid.Ny + 1, grid.Nx)),
            )
        """),
        markdown(r"""
        ## 4. Bathymetry supplied as a file

        The bathymetry loader accepts `.npy`, `.npz`, `.csv`, and `.txt` arrays.
        Depth must be positive, finite, measured in metres, and have exact shape
        `(Ny, Nx)`. The first index is south-to-north $y$ and the second is
        west-to-east $x$. The loader does not interpolate, reproject, or create a
        land mask.

        Here the supplied two-dimensional shelf-and-ridge map becomes
        `ModelParams.H`. Contour lines keep the bottom geometry visible while the
        surface wave propagates over it.
        """),
        code("""
        Nx, Ny = 160, 24
        Lx, Ly = 2.4e6, 360e3
        map_grid = make_grid(Nx, Ny, Lx, Ly)
        H_map = load_bathymetry(DATA_DIR / "example_bathymetry.npz", map_grid)
        print(
            "bathymetry shape:", H_map.shape,
            "depth range:", f"{H_map.min():.0f}–{H_map.max():.0f} m",
        )

        bathy_levels = np.linspace(H_map.min(), H_map.max(), 7)[1:-1]
        fig, ax = plt.subplots(figsize=(9, 3.4))
        field = ax.contourf(
            map_grid.x_c/1e3, map_grid.y_c/1e3, H_map,
            levels=18, cmap="Blues",
        )
        lines = ax.contour(
            map_grid.x_c/1e3, map_grid.y_c/1e3, H_map,
            levels=bathy_levels, colors="0.2", linewidths=0.7,
        )
        ax.clabel(lines, fmt="%.0f m", fontsize=7)
        ax.set(xlabel="x [km]", ylabel="y [km]", title="Bathymetry input map")
        fig.colorbar(field, ax=ax, label="depth [m]")
        plt.show()
        """),
        code("""
        bathy_params = ModelParams(
            H=H_map, g=9.81, f0=0.0, beta=0.0, r=0.0, linear=True,
        )
        bathy_dt = compute_dt_cfl(map_grid, bathy_params, cfl=0.42)
        bathy_out = run_model(
            tmax=7*3600, dt=bathy_dt,
            grid=map_grid, params=bathy_params,
            forcing_fn=zero_forcing,
            ic_fn=lambda g, p: cross_basin_pulse(g, p),
            save_every=5, out_vars=("eta",),
        )


        def draw_bathymetry(ax):
            contours = ax.contour(
                map_grid.x_c, map_grid.y_c, H_map,
                levels=bathy_levels, colors="0.15", linewidths=0.65,
                alpha=0.75,
            )
            ax.clabel(contours, fmt="%.0f m", fontsize=6)


        bathymetry_animation = animate_with_overlay(
            bathy_out, map_grid, "part_c_toolbox_bathymetry.gif",
            title="Wave over file-backed bathymetry",
            draw_overlay=draw_bathymetry,
        )
        bathymetry_animation
        """),
        markdown(r"""
        ## 5. Wind forcing supplied as a file

        The forcing file stores cell-centred stress components `tau_x` and
        `tau_y` in $\mathrm{N\,m^{-2}}$. The loader reads the file once, places
        the stresses on the staggered velocity grid, and returns a normal model
        forcing function. A separate envelope can vary the amplitude in time.

        The model is forced by stress rather than wind velocity. For display
        only, the code estimates a wind speed from

        $$
        |\boldsymbol{\tau}|=\rho_{air} C_D U_{10}^2,
        $$

        using constant air density and drag coefficient. Wind-speed contours and
        small direction arrows remain visible while the surface responds.
        """),
        code("""
        with np.load(DATA_DIR / "example_wind_forcing.npz", allow_pickle=False) as data:
            tau_x_map = np.asarray(data["tau_x"])
            tau_y_map = np.asarray(data["tau_y"])

        stress_magnitude = np.hypot(tau_x_map, tau_y_map)
        rho_air, drag_coefficient = 1.225, 1.3e-3
        wind_speed = np.sqrt(stress_magnitude / (rho_air * drag_coefficient))
        direction_x = np.divide(
            tau_x_map, stress_magnitude,
            out=np.zeros_like(tau_x_map), where=stress_magnitude > 0,
        )
        direction_y = np.divide(
            tau_y_map, stress_magnitude,
            out=np.zeros_like(tau_y_map), where=stress_magnitude > 0,
        )

        wind_levels = np.linspace(1.0, max(2.0, float(wind_speed.max())), 6)
        arrow_slice = (slice(1, None, 3), slice(4, None, 12))

        fig, ax = plt.subplots(figsize=(9, 3.4))
        speed_field = ax.contourf(
            map_grid.x_c/1e3, map_grid.y_c/1e3, wind_speed,
            levels=18, cmap="YlGnBu",
        )
        speed_lines = ax.contour(
            map_grid.x_c/1e3, map_grid.y_c/1e3, wind_speed,
            levels=wind_levels, colors="0.2", linewidths=0.65,
        )
        ax.clabel(speed_lines, fmt="%.1f m/s", fontsize=7)
        ax.quiver(
            map_grid.x_c[arrow_slice[1]]/1e3,
            map_grid.y_c[arrow_slice[0]]/1e3,
            direction_x[arrow_slice], direction_y[arrow_slice],
            color="0.15", scale=28, width=0.0022, pivot="middle",
        )
        ax.set(xlabel="x [km]", ylabel="y [km]", title="Mapped wind pattern")
        fig.colorbar(speed_field, ax=ax, label="estimated wind speed [m/s]")
        plt.show()
        """),
        code("""
        wind_map_forcing = make_wind_forcing_from_file(
            DATA_DIR / "example_wind_forcing.npz",
            map_grid,
            envelope=lambda t: 1.0,
        )
        wind_map_params = ModelParams(
            H=400.0, g=9.81, f0=0.0, beta=0.0,
            r=1/(2*86400), linear=True,
        )
        wind_map_dt = compute_dt_cfl(map_grid, wind_map_params, cfl=0.42)
        wind_map_out = run_model(
            tmax=8*3600, dt=wind_map_dt,
            grid=map_grid, params=wind_map_params,
            forcing_fn=wind_map_forcing, ic_fn=rest_state,
            save_every=8, out_vars=("eta",),
        )


        def draw_wind_pattern(ax):
            contours = ax.contour(
                map_grid.x_c, map_grid.y_c, wind_speed,
                levels=wind_levels, colors="0.15", linewidths=0.65,
                alpha=0.75,
            )
            ax.clabel(contours, fmt="%.1f m/s", fontsize=6)
            ax.quiver(
                map_grid.x_c[arrow_slice[1]],
                map_grid.y_c[arrow_slice[0]],
                direction_x[arrow_slice], direction_y[arrow_slice],
                color="0.12", scale=28, width=0.0022, pivot="middle",
            )


        wind_map_animation = animate_with_overlay(
            wind_map_out, map_grid, "part_c_toolbox_wind.gif",
            title="Surface response to mapped wind forcing",
            draw_overlay=draw_wind_pattern,
        )
        wind_map_animation
        """),
        markdown(r"""
        ## 6. Turning the toolbox into a project

        The demonstrations above are starting points, not prescribed projects.
        A group might change one map amplitude, shelf depth, forcing duration,
        domain dimension, initial-state scale, or Coriolis parameter. A compact
        design looks like this:

        | Element | Example |
        |---|---|
        | Question | How does eastward-wind duration affect eastern-wall setup? |
        | Prediction | Longer forcing initially produces larger setup. |
        | Baseline | 6-hour wind event; all other controls fixed. |
        | Variations | Identical 3-hour and 9-hour events. |
        | Diagnostic | Maximum mean $\eta$ in the four easternmost cells. |
        | Limitation | A rigid wet wall is not a beach or inundation model. |

        The report notebook already contains a reusable `run_case(...)`
        function and the required report headings. Groups should copy only the
        toolbox code relevant to their question, keep the report readable, and
        submit the executed report notebook—not this demonstration notebook.
        """),
    ]


def part_c_report_specs():
    return [
        markdown(r"""
        # Part C — Project report template

        **This is the notebook your group submits.** Rename it with your group
        identifier. This is a **short exploratory project**, not a comprehensive
        research study. Aim for two or three well-chosen model cases and one
        result that you can explain clearly.

        ## Assessment

        The project is graded **G (Godkänt)** or **U (Underkänt)**. There are no
        points or weighted criteria. A passing report adequately addresses:

        - a focused question, prediction, and controlled design;
        - appropriate quantitative evidence;
        - a physical interpretation of the main result;
        - awareness of relevant limitations and enough information to reproduce
          the cases;
        - a clear notebook and contribution statement.

        Keep the scope small. Use one main independent variable, and make sure
        the submitted notebook runs from top to bottom.
        """),
        markdown("""
        ## Group and research question

        **Group members:**  

        **Research question:**  
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
        `wind_off_hours`. Rotation is controlled by the constant Coriolis
        parameter `f` in $\mathrm{s^{-1}}$; use `f=0.0` for no rotation.

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
            depth=400.0, f=0.0, damping=0.0,
            initial_kind="cross_pulse", initial_amplitude=0.08,
            initial_radius=60e3, initial_x_fraction=0.25,
            initial_y_fraction=0.50,
            wind_x=0.0, wind_y=0.0, wind_ramp_hours=1.0,
            wind_off_hours=None, wind_file=None,
            tmax_hours=5.0,
        ):
            grid = make_grid(Nx, Ny, Lx, Ly)
            H = depth(grid) if callable(depth) else depth
            if isinstance(H, (str, Path)):
                H = load_bathymetry(H, grid)
            params = ModelParams(
                H=H, g=9.81, f0=f, beta=0.0,
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
                y_fraction=initial_y_fraction,
            )
            out = run_model(
                tmax=tmax_hours*3600, dt=dt, grid=grid, params=params,
                forcing_fn=forcing, ic_fn=initial, save_every=5,
                out_vars=("eta",),
            )
            print(
                f"{label}: Nx={Nx}, Ny={Ny}, Lx={Lx/1e3:.0f} km, "
                f"dx={grid.dx/1e3:.1f} km, dt={dt:.1f} s, "
                f"f={f:.2e} s^-1, frames={len(out['time'])}"
            )
            return {"label": label, "grid": grid, "params": params, "out": out}
        """),
        markdown("""
        ## Experiment

        Run one baseline and at least one controlled variation. Change one main
        parameter and keep the other relevant settings fixed. Add your own code
        cells below; the choice of plots and diagnostic is part of the project.

        Briefly state which parameter you changed and which controls you kept
        fixed.
        """),
        markdown("""
        ## Results and interpretation

        Show the evidence needed to answer the question. Include at least one
        quantitative result—not only an animation—and explain the main physical
        pattern you observe. Label figures with units. Briefly mention the single
        most relevant model, resolution, or experimental caveat for your result.
        """),
        markdown("""
        ## Conclusion

        Answer the research question directly in a short paragraph. State
        whether the prediction was supported and cite the main numerical result.
        """),
        markdown("""
        ## Contributions and submission check

        **Member contributions:**  

        Before submission:

        - [ ] All group members are named.
        - [ ] The evidence needed for the conclusion is visible.
        - [ ] The kernel was restarted and all cells ran in order without error.
        - [ ] The notebook is reproducible and contains no unexplained local path.
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
    nbf.write(
        _render(part_c_project_description_specs(), solution=False, prefix="ct"),
        NOTEBOOK_DIR / "part_c_project_description.ipynb",
    )
    nbf.write(
        _render(part_c_report_specs(), solution=False, prefix="cr"),
        NOTEBOOK_DIR / "part_c_project_report_template.ipynb",
    )
    build_example_data()
    print(f"Wrote notebooks to {NOTEBOOK_DIR}")
    print(f"Wrote example data to {DATA_DIR}")


if __name__ == "__main__":
    main()
