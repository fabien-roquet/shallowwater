#!/usr/bin/env python3
"""Geostrophic adjustment with coastal momentum friction to absorb returning waves.

The central basin is undamped; the boundary sponge removes wave energy without
relaxing height. The unforced f-plane circulation approaches a near-steady eddy.
"""
from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT/'src') not in sys.path:
    sys.path.insert(0, str(ROOT/'src'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from shallowwater import ModelParams, make_grid, setup_initial_state, zero_forcing
from shallowwater import run_model, compute_dt_cfl, geostrophic_velocities_from_eta, sponge_mask_eta
from shallowwater import operators
from shallowwater.visualize import animate_eta
from _experiment_tools import DAY, model_fingerprint


def ic_balanced_fraction(grid, params, amp=.5, R=2e5, x0=None, y0=None,
                         degree_of_balance=.8, alpha=.95, sponge=6):
    eta = setup_initial_state(grid, params, mode='gaussian_bump', amp=amp, R=R,
                               x0=grid.Lx/2 if x0 is None else x0,
                               y0=grid.Ly/2 if y0 is None else y0)[0]
    u, v = geostrophic_velocities_from_eta(eta, grid, params,
                degree_of_balance=degree_of_balance, alpha=alpha, sponge=sponge)
    return eta, u, v


def momentum_sponge(grid, *, width, tau):
    """Precompute positive Rayleigh rates near all walls, leaving height untouched."""
    if width <= 0 or tau <= 0 or 2*width >= min(grid.Lx, grid.Ly):
        raise ValueError('Use positive sponge width/time and leave an undamped interior.')
    mask = sponge_mask_eta(grid, width=width, sides=('west', 'east', 'south', 'north'), power=2.)
    rate_u = operators.avg_center_to_u(mask)/tau
    rate_v = operators.avg_center_to_v(mask)/tau
    def hook(state, time, grid, params):
        return None, -rate_u*state['u'], -rate_v*state['v']
    return hook, mask


def setup_experiment(args):
    n = 24 if args.smoke else (80 if args.quick else 160)
    grid = make_grid(n, n, 4e6, 4e6)
    # Uniform drag would continue spinning down the balanced central circulation.
    params = ModelParams(H=1000., g=9.81, rho=1025., f0=1e-4, beta=0.,
                         y0=grid.Ly/2, r=0., linear=True)
    period = 2*np.pi/abs(params.f0)
    duration = (.05 if args.smoke else args.periods)*period
    frames = min(5, args.frames) if args.smoke else args.frames
    bound = min(compute_dt_cfl(grid, params, cfl=.4), .5*args.sponge_hours*3600)
    steps = max(1, int(np.ceil(duration/bound)))
    dt = duration/steps
    stride = max(1, int(np.ceil(steps/(frames-1))))
    return grid, params, duration, period, dt, stride


def diagnostic_series(data, grid, params, mask, period):
    eta, u, v = (data[key] for key in ('eta', 'u', 'v'))
    interior = mask == 0
    uc, vc = .5*(u[..., :-1]+u[..., 1:]), .5*(v[:, :-1]+v[:, 1:])
    anomaly = eta-eta.mean(axis=(1, 2), keepdims=True)
    # Constant-depth C-grid energy: the average of squared face velocities.
    speed_squared = .5*(u[..., :-1]**2+u[..., 1:]**2+v[:, :-1]**2+v[:, 1:]**2)
    energy = .5*params.H*np.mean(speed_squared, axis=(1, 2))+.5*params.g*np.mean(anomaly**2, axis=(1, 2))
    misfit, change_rate = [], []
    for a, b, c, U, V in zip(eta, u, v, uc, vc):
        du = params.f0*operators.v_on_u(c)-params.g*operators.grad_x_on_u(a, grid.dx)
        dv = -params.f0*operators.u_on_v(b)-params.g*operators.grad_y_on_v(a, grid.dy)
        residual_u = .5*(du[:, :-1]+du[:, 1:])
        residual_v = .5*(dv[:-1]+dv[1:])
        acceleration = np.sqrt(np.mean((residual_u**2+residual_v**2)[interior]))
        coriolis = abs(params.f0)*np.sqrt(np.mean((U**2+V**2)[interior]))
        misfit.append(acceleration/max(coriolis, 1e-30))
        tendency = -params.H*operators.divergence(b, c, grid.dx, grid.dy)
        height_rms = np.sqrt(np.mean((a-a.mean())[interior]**2))
        change_rate.append(period*np.sqrt(np.mean(tendency[interior]**2))/max(height_rms, 1e-30))
    late = data['time'] >= data['time'][-1]-period
    late_height = anomaly[late][:, interior]
    variability = np.sqrt(np.mean((late_height-late_height.mean(axis=0))**2))
    amplitude = np.sqrt(np.mean(late_height**2))
    summary = dict(final_time_days=float(data['time'][-1]/DAY),
                   final_peak_height_m=float(eta[-1].max()),
                   final_interior_force_misfit=float(misfit[-1]),
                   late_interior_height_variability_fraction=float(variability/max(amplitude, 1e-30)),
                   maximum_mean_height_drift=float(np.max(np.abs(eta.mean(axis=(1, 2))-eta[0].mean()))),
                   final_energy_fraction=float(energy[-1]/energy[0]),
                   note='Interior means zero sponge strength; variability is evaluated over the last inertial period. The steady state is not prescribed or frozen.')
    return dict(energy=energy, force_misfit=np.asarray(misfit),
                relative_height_change_per_period=np.asarray(change_rate)), summary


def run_case(args):
    grid, params, duration, period, dt, stride = setup_experiment(args)
    sponge, mask = momentum_sponge(grid, width=args.sponge_width_km*1e3, tau=args.sponge_hours*3600)
    source = '\n'.join(inspect.getsource(fn) for fn in
                       (setup_experiment, ic_balanced_fraction, momentum_sponge, sponge_mask_eta))
    config = dict(Nx=grid.Nx, Ny=grid.Ny, Lx=grid.Lx, Ly=grid.Ly,
                  parameters=asdict(params), duration=duration, dt=dt, save_every=stride,
                  sponge_width_m=args.sponge_width_km*1e3, sponge_tau_seconds=args.sponge_hours*3600,
                  sponge_power=2., damp_height=False, amplitude=.5, radius=2e5,
                  initial_balance_fraction=.8, initial_alpha=.95, initial_taper_cells=6,
                  solver_fingerprint=model_fingerprint(),
                  experiment_fingerprint=hashlib.sha256(source.encode()).hexdigest())
    path = args.output_dir/'experiment_10_adjustment.npz'
    meta = path.with_suffix('.json')
    matching = path.exists() and meta.exists() and json.loads(meta.read_text())['configuration'] == config
    if args.replot and not matching:
        raise ValueError('No matching saved experiment 10; run without --replot first.')
    if matching:
        with np.load(path, allow_pickle=False) as saved:
            data = {key: saved[key] for key in ('time', 'eta', 'u', 'v')}
        print('10: reuse matching saved simulation', flush=True)
    else:
        print(f'10: {duration/DAY:.2f} days, coastal drag time {args.sponge_hours:g} h, width {args.sponge_width_km:g} km', flush=True)
        out = run_model(np.nextafter(duration, 0.), dt, grid, params, zero_forcing,
                        ic_balanced_fraction, save_every=stride, hooks=[sponge])
        data = {key: np.asarray(out[key]) for key in ('time', 'eta', 'u', 'v')}
        if not all(np.isfinite(a).all() for a in data.values()):
            raise RuntimeError('Non-finite adjustment solution.')
        np.savez_compressed(path, **data, x=grid.x_c, y=grid.y_c)
    series, summary = diagnostic_series(data, grid, params, mask, period)
    meta.write_text(json.dumps(dict(configuration=config, diagnostics=summary), indent=2)+'\n')
    print(json.dumps(summary), flush=True)
    return grid, params, data, mask, series, summary


def render(args, grid, params, data, mask, series):
    animation = animate_eta(data, grid, interval=100, contours=True, contour_levels=21,
                            title='Geostrophic adjustment\nCoastal momentum friction')
    animation.save(str(args.output_dir/'eta_10.gif'), fps=10)
    plt.close(animation.figure)
    period = 2*np.pi/abs(params.f0)
    time = data['time']/period
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), constrained_layout=True)
    axes[0].plot(time, series['energy']/series['energy'][0])
    axes[0].set(xlabel='Time / inertial period', ylabel='Energy / initial energy', title='Wave energy removed; circulation remains')
    axes[1].semilogy(time, np.maximum(series['force_misfit'], 1e-12))
    axes[1].set(xlabel='Time / inertial period', ylabel='Interior acceleration / Coriolis scale', title='Approach to geostrophic balance')
    axes[2].semilogy(time, np.maximum(series['relative_height_change_per_period'], 1e-12))
    axes[2].set(xlabel='Time / inertial period', ylabel='Tf × RMS(η_t) / RMS(η anomaly)', title='Approach to steady height')
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(args.output_dir/'experiment_10_adjustment_diagnostics.png', dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    im = ax.imshow(mask/(args.sponge_hours*3600), origin='lower',
                   extent=(0, grid.Lx/1e3, 0, grid.Ly/1e3), vmin=0)
    fig.colorbar(im, ax=ax, label='Momentum damping rate [s⁻¹]')
    ax.set(xlabel='x [km]', ylabel='y [km]', title='Coastal friction; undamped basin centre')
    fig.savefig(args.output_dir/'experiment_10_friction.png', dpi=150)
    plt.close(fig)


def make_parser():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--periods', type=float, default=8.)
    parser.add_argument('--sponge-hours', type=float, default=1.)
    parser.add_argument('--sponge-width-km', type=float, default=800.)
    parser.add_argument('--frames', type=int, default=241)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'animations')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--replot', action='store_true')
    return parser


def run_experiment(args):
    if min(args.periods, args.sponge_hours, args.sponge_width_km) <= 0 or args.frames < 2:
        raise ValueError('Use positive durations/width and at least two frames.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    grid, params, data, mask, series, summary = run_case(args)
    render(args, grid, params, data, mask, series)
    return data, summary


if __name__ == '__main__':
    run_experiment(make_parser().parse_args())
