#!/usr/bin/env python3
"""Compare the same equilibrium-tide forcing with and without rotation."""
from dataclasses import asdict, replace
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
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm
import numpy as np

from shallowwater import (ModelParams, make_grid, setup_initial_state,
                          tidal_potential_forcing, run_model, compute_dt_cfl)
from _experiment_tools import DAY, model_fingerprint


def setup_experiment(args):
    n = 16 if args.smoke else (64 if args.quick else 128)
    grid = make_grid(n, n, 4e6, 4e6)
    params = ModelParams(H=4000., g=9.81, rho=1025., f0=1e-4, beta=0.,
                         y0=grid.Ly/2, r=1/(15*DAY), linear=True)
    duration = (.05 if args.smoke else args.days)*DAY
    return grid, params, duration, compute_dt_cfl(grid, params, cfl=.5)


def initial_state(grid, params):
    return setup_initial_state(grid, params, mode='rest')


def forcing(time, grid, params):
    return tidal_potential_forcing(time, grid, params, amp_eta_eq=.20,
             omega=None, phase=0., kx=2*np.pi/20000e3, ky=0.)


def run_case(args, name):
    grid, base, duration, dt = setup_experiment(args)
    params = replace(base, f0=1e-4 if name == 'rotating' else 0.)
    source = '\n'.join(inspect.getsource(fn) for fn in
                       (setup_experiment, initial_state, forcing, run_case))
    config = dict(Nx=grid.Nx, Ny=grid.Ny, Lx=grid.Lx, Ly=grid.Ly,
                  parameters=asdict(params), duration=duration, dt=dt, save_every=30,
                  equilibrium_height_m=.20, forcing_omega='default M2',
                  forcing_kx=2*np.pi/20000e3, forcing_ky=0., forcing_phase=0.,
                  solver_fingerprint=model_fingerprint(),
                  experiment_fingerprint=hashlib.sha256(source.encode()).hexdigest())
    stem = '04' if name == 'rotating' else '04b_no_rotation'
    path = args.output_dir/f'experiment_{stem}.npz'
    meta = path.with_suffix('.json')
    matching = path.exists() and meta.exists() and json.loads(meta.read_text())['configuration'] == config
    if args.replot and not matching:
        raise ValueError(f'No matching {name} tide history; run without --replot first.')
    if matching:
        with np.load(path, allow_pickle=False) as saved:
            data = {key: saved[key] for key in ('time', 'eta')}
        print(f'04 {name}: reuse saved simulation', flush=True)
    else:
        print(f'04 {name}: integrate {duration/DAY:g} days', flush=True)
        out = run_model(duration, dt, grid, params, forcing, initial_state,
                        save_every=30, out_vars=('eta',))
        data = {key: np.asarray(out[key]) for key in ('time', 'eta')}
        if not all(np.isfinite(a).all() for a in data.values()):
            raise RuntimeError(f'Non-finite {name} tide simulation.')
        np.savez_compressed(path, **data, x=grid.x_c, y=grid.y_c)
        meta.write_text(json.dumps(dict(configuration=config), indent=2)+'\n')
    return grid, data


def archive_separate_movies(folder):
    if not (folder/'eta_04b_no_rotation.gif').exists():
        return
    archive = folder/'archive'
    archive.mkdir(exist_ok=True)
    for source, basename in [(folder/'eta_04.gif', 'eta_04_previous.gif'),
                              (folder/'eta_04b_no_rotation.gif', 'eta_04b_no_rotation.gif')]:
        if not source.exists():
            continue
        destination = archive/basename
        index = 1
        while destination.exists():
            destination = archive/f'{Path(basename).stem}_{index}.gif'
            index += 1
        source.rename(destination)


def render_comparison(args, grid, results):
    times = results['rotating']['time']
    np.testing.assert_array_equal(times, results['nonrotating']['time'])
    names = ('rotating', 'nonrotating')
    fields = [results[name]['eta']-results[name]['eta'].mean(axis=(1, 2), keepdims=True)
              for name in names]
    bound = max(float(np.percentile(np.abs(field), 98.)) for field in fields)
    unit = 10.**np.floor(np.log10(max(bound, 1e-12)))
    limit = next(a*unit for a in (1., 1.5, 2., 2.5, 3., 4., 5., 7.5, 10.) if a*unit >= bound)
    levels = np.linspace(-limit, limit, 21)
    contour_levels = np.linspace(-limit, limit, 13)
    norm = BoundaryNorm(levels, plt.get_cmap('RdBu_r').N)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    X, Y = np.meshgrid(grid.x_c/1e3, grid.y_c/1e3)
    # Fill the outer half-cell with the nearest cell height for plotting.
    Xf, Yf = np.meshgrid(np.r_[0., grid.x_c/1e3, grid.Lx/1e3],
                         np.r_[0., grid.y_c/1e3, grid.Ly/1e3])
    def filled_height(ax, field):
        return ax.contourf(Xf, Yf, np.pad(field, 1, mode='edge'), levels=levels,
                           cmap='RdBu_r', norm=norm, extend='both')
    fills, contours = [], [None, None]
    for ax, field, title in zip(axes, fields,
                ('With rotation: f = 10⁻⁴ s⁻¹', 'Without rotation: f = 0')):
        fills.append(filled_height(ax, field[0]))
        ax.set_aspect('equal')
        ax.set(xlabel='x [km]', ylabel='y [km]', title=title)
    fig.colorbar(ScalarMappable(norm=norm, cmap='RdBu_r'), ax=list(axes),
                 boundaries=levels, label='η [m]', extend='both',
                 ticks=np.linspace(-limit, limit, 5), shrink=.9)
    heading = fig.suptitle('', fontsize=12)
    def update(i):
        for j, (ax, field) in enumerate(zip(axes, fields)):
            fills[j].remove()
            fills[j] = filled_height(ax, field[i])
            if contours[j] is not None:
                contours[j].remove()
            if np.ptp(field[i]) > 1e-12:
                contours[j] = ax.contour(X, Y, field[i], levels=contour_levels,
                                         colors='k', linewidths=.4, alpha=.4)
            else:
                contours[j] = None
        heading.set_text(f'Tides: rotation versus no rotation\n'
                         f't = {times[i]/3600:.1f} hours; identical M2 forcing')
    animation = FuncAnimation(fig, update, frames=len(times), interval=1000/args.fps, blit=False)
    pending = args.output_dir/'eta_04_comparison_pending.gif'
    animation.save(str(pending), writer='pillow', fps=args.fps, dpi=120)
    archive_separate_movies(args.output_dir)
    pending.replace(args.output_dir/'eta_04.gif')
    fig.savefig(args.output_dir/'experiment_04_comparison_last_frame.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    (args.output_dir/'experiment_04_comparison.json').write_text(json.dumps(dict(
        panels=list(names), color_limits=[-limit, limit], renderer='contourf',
        filled_contour_levels=levels.tolist(), contour_levels=contour_levels.tolist(),
        fps=args.fps, frame_count=len(times), playback_seconds=len(times)/args.fps,
        time_seconds=times.tolist()), indent=2)+'\n')


def make_parser():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--days', type=float, default=3.)
    parser.add_argument('--fps', type=float, default=10.)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'animations')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--replot', action='store_true')
    return parser


def run_experiment(args):
    if min(args.days, args.fps) <= 0:
        raise ValueError('Duration and playback rate must be positive.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in ('rotating', 'nonrotating'):
        grid, results[name] = run_case(args, name)
    render_comparison(args, grid, results)
    return grid, results


if __name__ == '__main__':
    run_experiment(make_parser().parse_args())
