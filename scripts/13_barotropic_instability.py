#!/usr/bin/env python3
"""Wind-driven channel jets: weak/strong forcing and momentum advection on/off.

Run from any directory. --quick reduces resolution; --smoke checks the workflow.
The jet is spun up from rest by steady sech^2 stress, then given a tiny balanced
periodic seed. Forcing continues unchanged. No prescribed background state is
held fixed. All cases retain nonlinear mass transport and local wind acceleration.
"""
from dataclasses import asdict
import numpy as np
import matplotlib.pyplot as plt
from _experiment_tools import (
    arguments, integrate, last_state, balanced_patch, diagnostics, save_diagnostics,
    animate_fields, channel_streamfunction, DAY,
)
from shallowwater import ModelParams, make_grid, make_wind_forcing


def run_experiment(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    nx, ny = (24, 16) if args.smoke else ((80, 48) if args.quick else (160, 96))
    grid = make_grid(nx, ny, 2_000e3, 1_200e3, periodic_x=True)
    spin_days, follow_days = (0.1, 0.2) if args.smoke else (25.0, 100.0)
    width = 100e3
    results = {}
    cases = [("strong", 0.08, True), ("strong_no_advection", 0.08, False)]
    if getattr(args, "include_weak", False):
        cases.insert(0, ("weak", 0.005, True))
    for name, tau0, advection in cases:
        print(f"13 {name}: spin up {spin_days:g} days, then evolve {follow_days:g} days", flush=True)
        params = ModelParams(H=200.0, f0=1e-4, beta=1e-11, y0=grid.Ly/2,
                             r=1/(100*DAY), Ah=100.0, linear=False,
                             momentum_advection=advection, Hmin_frac=0.0)
        stress = np.broadcast_to(tau0 / np.cosh((grid.y_c[:, None]-grid.Ly/2)/width)**2,
                                 (ny, nx)).copy()
        forcing = make_wind_forcing(stress, np.zeros_like(stress), grid)
        spin = integrate(grid, params, forcing, days=spin_days, frames=min(args.frames, 40))
        initial = balanced_patch(grid, params, last_state(spin), amplitude=0.002,
                                 radius=width, seed_mode=4)
        out = integrate(grid, params, forcing, initial, days=follow_days, frames=args.frames)
        data = diagnostics(out, grid, params)
        data["psi"] = channel_streamfunction(data["zeta"], data["mean_u"], grid)
        spin_data = diagnostics(spin, grid, params)
        # Keep numerical diagnostics for both the build-up and the seeded stage.
        save_diagnostics(args.output_dir / f"experiment_13_{name}",
                         {**data,
                          **{f"spin_{k}": v for k, v in spin_data.items() if k != "zeta"},
                          "x": grid.x_c, "y": grid.y_c, "stress": stress},
                         dict(case=name, parameters=asdict(params), periodic_x=True,
                              Nx=nx, Ny=ny, spin_days=spin_days, follow_days=follow_days,
                              seed_amplitude_m=0.002, jet_width_m=width,
                              note="Advection-off case retains nonlinear mass and wind-depth terms."))
        print(f"  max speed {data['max_speed'][-1]:.3f} m/s; "
              f"eddy KE {data['eddy_ke'][0]:.3g} -> {data['eddy_ke'][-1]:.3g} m²/s²", flush=True)
        results[name] = (out, data, spin_data)

    times = results["strong"][0]["time"]
    names = ("strong", "strong_no_advection")
    psi_fields = [results[name][1]["psi"]/1e6 for name in names]
    animate_fields(args.output_dir / f"eta_13_barotropic_instability.{args.format}", grid, times,
                   psi_fields + [results[name][1]["zeta"]*1e5 for name in names],
                   ["Advection on: streamfunction", "Advection off: streamfunction",
                    "Advection on: relative vorticity", "Advection off: relative vorticity"],
                   labels=["ψ [10⁶ m² s⁻¹]"]*2 + ["ζ [10⁻⁵ s⁻¹]"]*2,
                   time_label="Time since balanced seed", scale_groups=[0, 0, 1, 1],
                   ncols=2, contour_panels=(0, 1))
    # Keep the older, uninformative spin-up animation out of the main gallery.
    old_spinup = args.output_dir / f"eta_13_jet_spinup.{args.format}"
    if old_spinup.exists():
        archive = args.output_dir / "archive"
        archive.mkdir(exist_ok=True)
        if not (archive / old_spinup.name).exists():
            old_spinup.rename(archive / old_spinup.name)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for name, (out, data, spin_data) in results.items():
        axes[0].semilogy(data["time_days"], np.maximum(data["eddy_ke"], 1e-30), label=name)
        axes[1].plot(data["mean_u"][-1], grid.y_c/1e3, label=name)
        axes[2].plot(data["mean_q_gradient"][-1]*1e13, grid.y_c/1e3, label=name)
    axes[0].set(xlabel="Time since seed [days]", ylabel="Eddy kinetic energy [m²/s²]")
    axes[1].set(xlabel="Final zonal-mean u [m/s]", ylabel="y [km]")
    axes[2].set(xlabel="Final mean PV gradient [10⁻¹³ m⁻² s⁻¹]", ylabel="y [km]")
    axes[2].axvline(0, color="k", linewidth=0.6)
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.savefig(args.output_dir / "experiment_13_diagnostics.png", dpi=150)
    plt.close(fig)
    return results


if __name__ == "__main__":
    parser = arguments(__doc__)
    parser.add_argument("--include-weak", action="store_true",
                        help="Also run the weak-wind comparison for numerical diagnostics.")
    run_experiment(parser.parse_args())
