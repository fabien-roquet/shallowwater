#!/usr/bin/env python3
"""Tsunami shoaling in a square ocean with weak bottom drag and outer sponges."""
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
from matplotlib.animation import FuncAnimation
from matplotlib.colors import BoundaryNorm
import numpy as np

from shallowwater import (ModelParams, make_grid, setup_initial_state, zero_forcing,
                          run_model, compute_dt_cfl, shelf_bathymetry, sponge_mask_eta)
from shallowwater.operators import avg_center_to_u, avg_center_to_v
from _experiment_tools import model_fingerprint


def setup_experiment(args):
    n = 32 if args.smoke else (160 if args.quick else 320)
    grid = make_grid(n, n, 4.8e6, 4.8e6)
    depth = shelf_bathymetry(grid, H_deep=4000., H_coast=80.,
                            shelf_width=900e3, coast='east', power=1.4)
    # Linear Rayleigh bottom drag damps momentum, not surface height directly.
    params = ModelParams(H=depth, f0=0., beta=0.,
                         r=0. if args.drag_hours == 0 else 1/(args.drag_hours*3600), linear=True)
    duration = (.1 if args.smoke else args.hours)*3600
    steps = int(np.ceil(duration/compute_dt_cfl(grid, params, cfl=.45)))
    dt = duration/steps
    frames = min(args.frames, 5) if args.smoke else args.frames
    stride = max(1, int(np.ceil(steps/(frames-1))))
    return grid, params, duration, dt, stride


def tsunami_initial_condition(grid, params):
    return setup_initial_state(grid, params, mode='gaussian_bump', amp=.20,
                               R=120e3, x0=1600e3, y0=.5*grid.Ly)


def absorbing_boundaries(grid):
    """Precompute absorption on the three outer edges; leave the east coast alone."""
    mask = sponge_mask_eta(grid, width=600e3, sides=('west', 'south', 'north'), power=2.)
    sigma = mask/900.
    sigma_u, sigma_v = avg_center_to_u(sigma), avg_center_to_v(sigma)
    def hook(state, time, grid, params):
        return -sigma*state['eta'], -sigma_u*state['u'], -sigma_v*state['v']
    return hook, mask


def run_case(args):
    grid, params, duration, dt, stride = setup_experiment(args)
    sponge, mask = absorbing_boundaries(grid)
    source = '\n'.join(inspect.getsource(fn) for fn in
                       (setup_experiment, tsunami_initial_condition, absorbing_boundaries))
    parameters = asdict(params)
    parameters.pop('H')
    config = dict(Nx=grid.Nx, Ny=grid.Ny, Lx=grid.Lx, Ly=grid.Ly,
                  parameters=parameters, depth_sha256=hashlib.sha256(params.H.tobytes()).hexdigest(),
                  H_deep=4000., H_coast=80., shelf_width_m=900e3, shelf_power=1.4,
                  amplitude_m=.20, radius_m=120e3, source_x_m=1600e3, source_y_m=grid.Ly/2,
                  sponge_sides=['west', 'south', 'north'], sponge_width_m=600e3,
                  sponge_tau_seconds=900., sponge_power=2., duration=duration, dt=dt,
                  save_every=stride, solver_fingerprint=model_fingerprint(),
                  experiment_fingerprint=hashlib.sha256(source.encode()).hexdigest())
    path = args.output_dir/'experiment_03_tsunami.npz'
    meta = path.with_suffix('.json')
    matching = path.exists() and meta.exists() and json.loads(meta.read_text())['configuration'] == config
    if args.replot and not matching:
        raise ValueError('No matching tsunami history; run without --replot first.')
    if matching:
        with np.load(path, allow_pickle=False) as saved:
            data = {key: saved[key] for key in ('eta', 'time')}
        print('03: reuse saved simulation', flush=True)
    else:
        print(f'03: {grid.Nx} × {grid.Ny}, {duration/3600:g} h, bottom-drag time {args.drag_hours:g} h', flush=True)
        out = run_model(np.nextafter(duration, 0.), dt, grid, params, zero_forcing,
                        tsunami_initial_condition, save_every=stride, hooks=[sponge], out_vars=('eta',))
        data = {key: np.asarray(out[key]) for key in ('time', 'eta')}
        if not all(np.isfinite(a).all() for a in data.values()):
            raise RuntimeError('Non-finite tsunami solution.')
        np.savez_compressed(path, **data, H=params.H, x=grid.x_c, y=grid.y_c)
    meta.write_text(json.dumps(dict(configuration=config), indent=2)+'\n')
    return grid, params, data, mask


def render(args, grid, params, data, mask):
    levels = np.linspace(-.04, .04, 17)
    fig, ax = plt.subplots(figsize=(8, 7.5), constrained_layout=True)
    extent = (0, grid.Lx/1e3, 0, grid.Ly/1e3)
    norm = BoundaryNorm(levels, plt.get_cmap('RdBu_r').N)
    image = ax.imshow(data['eta'][0], origin='lower', extent=extent,
                      cmap='RdBu_r', norm=norm, aspect='equal')
    fig.colorbar(image, ax=ax, label='Surface height η [m]', extend='both',
                 ticks=np.linspace(-.04, .04, 5), shrink=.9)
    X, Y = np.meshgrid(grid.x_c/1e3, grid.y_c/1e3)
    bathy = ax.contour(X, Y, params.H, levels=[100, 500, 2000], colors='0.35',
                       linestyles='--', linewidths=.65, alpha=.7)
    ax.clabel(bathy, fmt=lambda h: f'{h:.0f} m', fontsize=8)
    ax.contour(X, Y, mask, levels=[.001], colors='0.5', linewidths=.6, linestyles=':')
    ax.set(xlabel='x [km]', ylabel='y [km]')
    contours = [None]
    def update(i):
        image.set_data(data['eta'][i])
        if contours[0] is not None:
            contours[0].remove()
        contours[0] = ax.contour(X, Y, data['eta'][i], levels=np.linspace(-.04, .04, 9),
                                  colors='k', linewidths=.35, alpha=.3)
        ax.set_title(f'Tsunami shoaling toward the eastern coast\n'
                     f't = {data["time"][i]/3600:.2f} h; '+
                     (f'bottom-drag time = {args.drag_hours:g} h' if args.drag_hours else 'no bottom drag'))
    animation = FuncAnimation(fig, update, frames=len(data['time']), interval=1000/args.fps, blit=False)
    animation.save(str(args.output_dir/'eta_03_tsunami_shoaling_sponge.gif'),
                    writer='pillow', fps=args.fps, dpi=120)
    update(int(np.argmin(abs(data['time']-5.5*3600))))
    fig.savefig(args.output_dir/'experiment_03_shoaling_frame.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    render_diagnostics(args, grid, params, data)


def render_diagnostics(args, grid, params, data):
    # A maximum over time includes coast reflection; it is not a pure incident-wave gain.
    peak = np.max(np.abs(data['eta'][:, grid.Ny//2, :]), axis=0)
    travel = np.cumsum(grid.dx/np.sqrt(params.g*params.H[grid.Ny//2]))
    source = np.argmin(abs(grid.x_c-1600e3))
    arrival = (travel-travel[source])/3600
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), constrained_layout=True)
    axes[0].plot(grid.x_c/1e3, peak, label='Maximum |η| on the centreline')
    axes[0].set(xlabel='x [km]', ylabel='Surface height [m]', title='Propagation, spreading and coastal amplification')
    axes[0].axvline(3900, color='.5', linestyle=':', label='Shelf begins')
    axes[0].legend(fontsize=9)
    axdepth = axes[0].twinx()
    axdepth.plot(grid.x_c/1e3, params.H[grid.Ny//2], color='.5', linestyle='--')
    axdepth.set(ylabel='Depth [m]', ylim=(4200, 0))
    limit = .04
    axes[1].pcolormesh(grid.x_c/1e3, data['time']/3600,
                       data['eta'][:, grid.Ny//2], cmap='RdBu_r', vmin=-limit, vmax=limit, shading='auto')
    east = grid.x_c >= 1600e3
    axes[1].plot(grid.x_c[east]/1e3, arrival[east], 'k--', linewidth=.8, label='Direct-wave travel time ∫ dx/√(gH)')
    axes[1].set(xlabel='x [km]', ylabel='Time [h]', ylim=(0, data['time'][-1]/3600))
    axes[1].legend(fontsize=8)
    fig.savefig(args.output_dir/'experiment_03_shoaling_diagnostics.png', dpi=150)
    plt.close(fig)


def make_parser():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hours', type=float, default=7.)
    parser.add_argument('--drag-hours', type=float, default=24., help='Momentum damping time; 0 disables drag.')
    parser.add_argument('--frames', type=int, default=141)
    parser.add_argument('--fps', type=float, default=10.)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'animations')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--replot', action='store_true')
    return parser


def run_experiment(args):
    if min(args.hours, args.fps) <= 0 or args.drag_hours < 0 or args.frames < 2:
        raise ValueError('Use positive duration/fps, nonnegative drag time and at least two frames.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    grid, params, data, mask = run_case(args)
    render(args, grid, params, data, mask)
    return grid, params, data


if __name__ == '__main__':
    run_experiment(make_parser().parse_args())
