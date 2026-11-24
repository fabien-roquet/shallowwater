#!/usr/bin/env python3
"""Long-lived balanced patches in opposite wind-driven gyres, and wind adjustment.

Both wind signs are spun up independently from rest. Each patch is compared
with its own evolving control. A separate two-panel experiment ramps wind from rest in linear and fully
nonlinear models. Hovmoller diagrams show propagation of the balanced patches.
No fixed reference circulation is imposed on the dynamics.
"""
from dataclasses import asdict
import json
import numpy as np
import matplotlib.pyplot as plt
from _experiment_tools import (
    arguments, integrate, last_state, balanced_patch, diagnostics, save_diagnostics,
    animate_fields, difference, model_fingerprint, DAY,
)
from shallowwater import ModelParams, make_grid, make_wind_forcing
from _gyre_wind_ramp import run_wind_ramp, render_wind_ramp


def spin_up(grid, params, forcing, *, days, frames, path, configuration, fresh=False):
    """Reuse an equilibrated state only when all spin-up settings match exactly."""
    signature = json.dumps(configuration, sort_keys=True)
    if path.exists() and not fresh:
        with np.load(path, allow_pickle=False) as saved:
            if str(saved['configuration']) == signature:
                print(f"  reuse matching spin-up: {path.name}", flush=True)
                initial = tuple(saved[k].copy() for k in ('eta', 'u', 'v'))
                history = {"time": saved['time'].copy(), "eta": saved['eta_history'].copy()}
                return initial, history, float(saved['relative_change']), float(saved['check_days']), True
    spin = integrate(grid, params, forcing, days=days, frames=frames)
    initial = last_state(spin)
    change = np.sqrt(sum(np.mean((spin[k][-1]-spin[k][-2])**2) for k in ('u', 'v')))
    speed = np.sqrt(sum(np.mean(spin[k][-1]**2) for k in ('u', 'v')))
    relative = float(change/max(speed, 1e-15))
    interval = (spin['time'][-1]-spin['time'][-2])/DAY
    np.savez_compressed(path, configuration=np.asarray(signature),
                        eta=initial[0], u=initial[1], v=initial[2],
                        time=np.asarray(spin['time']), eta_history=np.asarray(spin['eta']),
                        relative_change=relative, check_days=interval)
    return initial, spin, relative, interval, False


def wave_hovmollers(path, grid, times, patch, reverse_patch, *, radius):
    """Height departures averaged in a narrow band; zero is never renormalized."""
    band = abs(grid.y_c-grid.Ly/2) <= max(radius/4, grid.dy)
    fields = [patch['eta'][:, band].mean(axis=1),
              reverse_patch['eta'][:, band].mean(axis=1)]
    titles = ['Patch: original wind', 'Patch: equilibrated reversed wind']
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    common = max(float(np.max(np.abs(a))) for a in fields[:2])
    for i, (ax, field, title) in enumerate(zip(axes, fields, titles)):
        limit = max(common if i < 2 else float(np.max(np.abs(field))), 1e-12)
        image = ax.pcolormesh(grid.x_c/1e3, np.asarray(times)/DAY, field,
                              cmap='RdBu_r', vmin=-limit, vmax=limit, shading='auto')
        ax.set(xlabel='x [km]', ylabel='Time since patch / wind change [days]', title=title)
        ax.axvline(grid.Lx/2e3, color='k', linestyle=':', linewidth=0.7)
        fig.colorbar(image, ax=ax, label='height departure [m]')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return fields


def render_gyre_animations(args, grid, times, control, patch, reverse_control,
                           reverse_patch):
    """Use control-height contours instead of allocating control-only panels."""
    animate_fields(args.output_dir/f'eta_14_balanced_patch.{args.format}', grid, times,
                   [control+patch, patch, reverse_control+reverse_patch, reverse_patch],
                   ['Eastward jet: total height', 'Eastward jet: patch departure',
                    'Westward jet: total height', 'Westward jet: patch departure'],
                   labels=['η [m]', 'δη [m]', 'η [m]', 'δη [m]'],
                   time_label='Time since identical patches added',
                   subtitle='Black contours: corresponding unperturbed control height',
                   scale_groups=[0, 1, 0, 1], ncols=2,
                   contour_fields={0: control, 1: control, 2: reverse_control, 3: reverse_control})
    # Preserve retired output while removing duplicate animations from the gallery.
    for stem in ('eta_14_balanced_patch_reversed_wind', 'eta_14_patch_wind_comparison'):
        old = args.output_dir/f'{stem}.{args.format}'
        if old.exists():
            archive = args.output_dir/'archive'
            archive.mkdir(exist_ok=True)
            destination = archive/old.name
            if not destination.exists():
                old.rename(destination)


def replot_saved(args):
    """Render saved fields without repeating the integrations."""
    if args.increase_wind:
        from _gyre_wind_sweep import render_wind_sweep
        return render_wind_sweep(args)
    if args.part != 'wind-ramp':
        with np.load(args.output_dir/'experiment_14_departures.npz', allow_pickle=False) as data:
            grid = make_grid(len(data['x']), len(data['y']),
                             len(data['x'])*(data['x'][1]-data['x'][0]),
                             len(data['y'])*(data['y'][1]-data['y'][0]))
            render_gyre_animations(args, grid, data['time_days']*DAY,
                                   data['control_eta'], data['patch_eta'],
                                   data['reverse_control_eta'], data['reverse_patch_eta'])
    if args.part != 'patches':
        render_wind_ramp(args)


def run_experiment(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.increase_wind:
        from _gyre_wind_sweep import run_wind_sweep
        return run_wind_sweep(args)
    if args.part == 'wind-ramp':
        return run_wind_ramp(args)
    nx, ny = (20, 16) if args.smoke else ((64, 48) if args.quick else (128, 96))
    grid = make_grid(nx, ny, 2_000e3, 1_600e3)
    params = ModelParams(H=200., f0=1e-4, beta=args.beta, y0=grid.Ly/2,
                         r=1/(args.drag_days*DAY), Ah=args.viscosity,
                         linear=False, Hmin_frac=0.)
    spin_days, follow_days = (0.1, 0.2) if args.smoke else (args.spin_days, args.days)
    radius = args.patch_radius_km*1e3
    stress = np.broadcast_to(-args.tau0*np.cos(2*np.pi*grid.y_c[:, None]/grid.Ly), (ny, nx)).copy()
    results, histories = {}, {}
    for sign, prefix in ((1., ''), (-1., 'reverse_')):
        label = 'original' if sign > 0 else 'reversed'
        forcing = make_wind_forcing(sign*stress, np.zeros_like(stress), grid)
        config = dict(parameters=asdict(params), Nx=nx, Ny=ny, Lx=grid.Lx, Ly=grid.Ly,
                      tau0=args.tau0, wind_sign=sign, spin_days=spin_days,
                      saved_frames=min(args.frames, 50), solver_fingerprint=model_fingerprint())
        print(f"14 {label} wind: spin up for {spin_days:g} days", flush=True)
        initial, spin, steady_metric, check_days, reused = spin_up(
            grid, params, forcing, days=spin_days, frames=min(args.frames, 50),
            path=args.output_dir/f'experiment_14_spinup_{label}.npz',
            configuration=config, fresh=args.fresh_spinup)
        uc = 0.5*(initial[1][:, :-1]+initial[1][:, 1:])
        X, Y = np.meshgrid(grid.x_c, grid.y_c)
        weights = np.exp(-((X-grid.Lx/2)**2+(Y-grid.Ly/2)**2)/radius**2)
        local_u = float(np.sum(weights*uc)/np.sum(weights))
        print(f"  relative velocity change: {steady_metric:.3g} over {check_days:.2f} days; "
              f"patch-weighted U = {local_u:+.3f} m/s", flush=True)
        histories[label] = spin
        info = dict(spin_relative_velocity_change=steady_metric, spin_check_interval_days=check_days,
                    spinup_reused=reused, wind_sign=sign, patch_weighted_u_m_s=local_u)
        branches = [('control', initial, forcing, 1.),
                    ('patch', balanced_patch(grid, params, initial,
                                            amplitude=args.patch_amplitude, radius=radius), forcing, 1.)]
        for suffix, ic, branch_forcing, factor in branches:
            name = prefix+suffix
            print(f"14 {name}: evolve {follow_days:g} days", flush=True)
            out = integrate(grid, params, branch_forcing, ic, days=follow_days, frames=args.frames)
            results[name] = out
            data = diagnostics(out, grid, params)
            save_diagnostics(args.output_dir/f'experiment_14_{name}',
                             {**{k: v for k, v in data.items() if k != 'zeta'},
                              'x': grid.x_c, 'y': grid.y_c, 'stress_before': sign*stress,
                              **{f'{k}_initial': out[k][0] for k in ('eta', 'u', 'v')},
                              **{f'{k}_final': out[k][-1] for k in ('eta', 'u', 'v')}},
                             dict(case=name, parameters=asdict(params), Nx=nx, Ny=ny,
                                  spin_days=spin_days, follow_days=follow_days,
                                  wind_factor=factor, patch_amplitude_m=args.patch_amplitude,
                                  patch_radius_m=radius, **info))

    control = results['control']
    patch = difference(results['patch'], control)
    reverse_patch = difference(results['reverse_patch'], results['reverse_control'])
    times = np.asarray(control['time'])
    render_gyre_animations(args, grid, times, np.asarray(control['eta']), patch['eta'],
                           np.asarray(results['reverse_control']['eta']),
                           reverse_patch['eta'])
    hovmollers = wave_hovmollers(args.output_dir/'experiment_14_wave_propagation.png',
                               grid, times, patch, reverse_patch, radius=radius)
    np.savez_compressed(args.output_dir/'experiment_14_departures.npz', time_days=times/DAY,
                        x=grid.x_c, y=grid.y_c, patch_eta=patch['eta'],
                        reverse_patch_eta=reverse_patch['eta'],
                        control_eta=np.asarray(control['eta']),
                        reverse_control_eta=np.asarray(results['reverse_control']['eta']),
                        patch_hovmoller=hovmollers[0], reverse_patch_hovmoller=hovmollers[1],
                        spin_time_days=np.asarray(histories['original']['time'])/DAY,
                        spin_eta=np.asarray(histories['original']['eta']))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for name in ('control', 'patch', 'reverse_control', 'reverse_patch'):
        data = diagnostics(results[name], grid, params)
        axes[0].plot(times/DAY, data['mean_ke'], label=name)
    for name, delta in (('Original wind patch', patch), ('Reversed wind patch', reverse_patch)):
        axes[1].plot(times/DAY, np.sqrt(np.mean(delta['eta']**2, axis=(1, 2))), label=name)
    axes[0].set(xlabel='Time since branching [days]', ylabel='Mean kinetic energy [m²/s²]')
    axes[1].set(xlabel='Time since branching [days]', ylabel='RMS height departure from own control [m]')
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.savefig(args.output_dir/'experiment_14_diagnostics.png', dpi=150)
    plt.close(fig)
    if args.part == 'all':
        run_wind_ramp(args)
    return results


def make_parser():
    parser = arguments(__doc__)
    parser.set_defaults(frames=120)
    parser.add_argument('--part', choices=('all', 'patches', 'wind-ramp'), default='all',
                        help='Choose the patch comparison, wind-ramp comparison, or both.')
    parser.add_argument('--increase-wind', action='store_true', help='Continue both saved baseline gyres through increasing wind levels.')
    parser.add_argument('--wind-drag-days', type=float, default=None, help='Optional drag timescale for continuation; both models use the same value.')
    parser.add_argument('--wind-viscosity', type=float, default=None, help='Optional lateral viscosity for continuation [m²/s].')
    parser.add_argument('--wind-levels', nargs='+', type=float, default=[.06, .08, .10, .12, .14, .16, .20, .24, .30, .40], help='Successive wind stress targets [N/m²].')
    parser.add_argument('--wind-stage-days', type=float, default=60., help='Ramp duration for each continuation stage.')
    parser.add_argument('--wind-hold-days', type=float, default=60., help='Fixed-wind duration after each continuation ramp.')
    parser.add_argument('--ramp-case', choices=('both', 'linear', 'nonlinear'), default='both',
                        help='Compute one ramp case for a separate job, or compare both.')
    parser.add_argument('--ramp-wind-profile', choices=('double-gyre', 'jet'), default='double-gyre',
                        help='Broad gyre wind, or the narrow-jet sensitivity case.')
    parser.add_argument('--ramp-tau-max', type=float, default=.04, help='Final peak wind stress [N/m²].')
    parser.add_argument('--ramp-days', type=float, default=120., help='Time to reach maximum wind stress.')
    parser.add_argument('--ramp-hold-days', type=float, default=120., help='Hold maximum wind after the ramp.')
    parser.add_argument('--ramp-drag-days', type=float, default=40., help='Rayleigh damping time for wind-ramp cases.')
    parser.add_argument('--ramp-viscosity', type=float, default=1000., help='Wind-ramp lateral viscosity [m²/s].')
    parser.add_argument('--ramp-boundary-viscosity', type=float, default=1000., help='Viscosity at the coast [m²/s]; set equal to ramp-viscosity for uniform mixing.')
    parser.add_argument('--ramp-boundary-width-km', type=float, default=150., help='Width of enhanced coastal mixing [km].')
    parser.add_argument('--ramp-beta', type=float, default=2e-11, help='Wind-ramp beta [m⁻¹ s⁻¹].')
    parser.add_argument('--ramp-length-km', type=float, default=2000., help='Length of the wind-ramp basin [km].')
    parser.add_argument('--ramp-wind-width-km', type=float, default=75., help='Width of the central wind band [km].')
    parser.add_argument('--ramp-field', choices=('eta', 'vorticity'), default='eta', help='Quantity colored in the two ramp panels.')
    parser.add_argument('--drag-days', type=float, default=20., help='Rayleigh momentum damping time [days].')
    parser.add_argument('--viscosity', type=float, default=2000., help='Lateral viscosity [m²/s].')
    parser.add_argument('--beta', type=float, default=5e-12, help='Planetary vorticity gradient [m⁻¹ s⁻¹].')
    parser.add_argument('--tau0', type=float, default=0.015, help='Peak wind stress magnitude [N/m²].')
    parser.add_argument('--patch-amplitude', type=float, default=0.08, help='Gaussian height coefficient [m].')
    parser.add_argument('--patch-radius-km', type=float, default=180., help='Gaussian e-folding radius [km].')
    parser.add_argument('--spin-days', type=float, default=240., help='Independent spin-up duration for each wind sign.')
    parser.add_argument('--days', type=float, default=45., help='Continuation duration after branching.')
    parser.add_argument('--replot', action='store_true', help='Redraw saved fields without running the model.')
    parser.add_argument('--fresh-spinup', action='store_true', help='Recompute even a matching cached spin-up.')
    return parser


if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    if (min(args.drag_days, args.patch_radius_km, args.spin_days, args.days, args.ramp_days,
            args.ramp_drag_days, args.ramp_tau_max, args.ramp_length_km, args.ramp_wind_width_km) <= 0
            or min(args.viscosity, args.ramp_viscosity, args.ramp_hold_days) < 0
            or args.ramp_boundary_width_km <= 0
            or args.ramp_boundary_viscosity < args.ramp_viscosity):
        parser.error('Times and patch radius must be positive; viscosity must be non-negative.')
    if args.replot:
        replot_saved(args)
    else:
        run_experiment(args)
