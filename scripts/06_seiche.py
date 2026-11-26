#!/usr/bin/env python3
"""Compare rectangular-basin seiche modes, with (1,0) on the left.

The constant-height mode is retained in the display rather than demeaned away.
Use --modes 0,0 1,0 0,1 1,1 to include the lowest oscillating modes as well.
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
from matplotlib.animation import FuncAnimation
from matplotlib.colors import BoundaryNorm
import numpy as np

from shallowwater import ModelParams, make_grid, run_model, zero_forcing, compute_dt_cfl
from _experiment_tools import model_fingerprint

AMPLITUDE = .10


def mode_frequency(grid, params, m, n, *, discrete=False):
    k, ell = m*np.pi/grid.Lx, n*np.pi/grid.Ly
    if discrete:
        k = 2*np.sin(k*grid.dx/2)/grid.dx
        ell = 2*np.sin(ell*grid.dy/2)/grid.dy
    return float(np.sqrt(params.g*params.H*(k*k+ell*ell)))


def seiche_ic(grid, params, A=AMPLITUDE, m=1, n=0):
    X, Y = np.meshgrid(grid.x_c, grid.y_c)
    eta = A*np.cos(m*np.pi*X/grid.Lx)*np.cos(n*np.pi*Y/grid.Ly)
    return eta, np.zeros((grid.Ny, grid.Nx+1)), np.zeros((grid.Ny+1, grid.Nx))


def setup_experiment(args):
    n = 16 if args.smoke else (32 if args.quick else 64)
    grid = make_grid(n, n, 2e6, 2e6)
    params = ModelParams(H=1000., g=9.81, rho=1025., f0=0., beta=0.,
                         y0=grid.Ly/2, r=0., linear=True)
    if any(m >= grid.Nx or n >= grid.Ny for m, n in args.modes):
        raise ValueError('Mode indices must be below the grid resolution.')
    frequencies = [mode_frequency(grid, params, m, n) for m, n in args.modes]
    positive = [omega for omega in frequencies if omega > 0]
    reference = min(positive) if positive else mode_frequency(grid, params, 1, 0)
    duration = (.05 if args.smoke else args.periods)*2*np.pi/reference
    return grid, params, duration, compute_dt_cfl(grid, params, cfl=.5)


def run_mode(args, grid, params, duration, dt, m, n):
    source = '\n'.join(inspect.getsource(fn) for fn in
                       (setup_experiment, mode_frequency, seiche_ic, run_mode))
    omega = mode_frequency(grid, params, m, n)
    config = dict(Nx=grid.Nx, Ny=grid.Ny, Lx=grid.Lx, Ly=grid.Ly,
                  parameters=asdict(params), mode=[m, n], amplitude=AMPLITUDE,
                  duration=duration, dt=dt, save_every=10,
                  solver_fingerprint=model_fingerprint(),
                  experiment_fingerprint=hashlib.sha256(source.encode()).hexdigest())
    path = args.output_dir/f'experiment_06_m{m}_n{n}.npz'
    meta = path.with_suffix('.json')
    matching = path.exists() and meta.exists() and json.loads(meta.read_text())['configuration'] == config
    if args.replot and not matching:
        raise ValueError(f'No matching saved mode ({m},{n}); run without --replot first.')
    if matching:
        with np.load(path, allow_pickle=False) as saved:
            data = {key: saved[key] for key in saved.files}
        print(f'06 mode ({m},{n}): reuse saved simulation', flush=True)
    else:
        print(f'06 mode ({m},{n}): '+('stationary uniform offset' if omega == 0 else f'period {2*np.pi/omega/3600:.2f} h'), flush=True)
        out = run_model(duration, dt, grid, params, zero_forcing,
                        lambda g, p: seiche_ic(g, p, m=m, n=n),
                        save_every=10, out_vars=('eta', 'u', 'v'))
        data = {key: np.asarray(out[key]) for key in ('time', 'eta', 'u', 'v')}
        if not all(np.isfinite(a).all() for a in data.values()):
            raise RuntimeError(f'Non-finite mode ({m},{n}).')
        np.savez_compressed(path, **data, x=grid.x_c, y=grid.y_c)
    pattern = seiche_ic(grid, params, A=1., m=m, n=n)[0]
    projection = np.mean(data['eta']*pattern, axis=(1, 2))/np.mean(pattern**2)
    discrete_omega = mode_frequency(grid, params, m, n, discrete=True)
    error = np.max(np.abs(projection-AMPLITUDE*np.cos(discrete_omega*data['time'])))
    meta.write_text(json.dumps(dict(configuration=config,
          frequency_rad_s=omega, period_seconds=2*np.pi/omega if omega > 0 else None,
          diagnostics=dict(maximum_modal_height_error=float(error),
                           maximum_mean_height_drift=float(np.max(np.abs(data['eta'].mean(axis=(1, 2))-data['eta'][0].mean()))))), indent=2)+'\n')
    return data


def render(args, grid, params, results):
    modes = list(results)
    columns = len(modes) if len(modes) <= 3 else 2
    rows = int(np.ceil(len(modes)/columns))
    fig, axes = plt.subplots(rows, columns, figsize=(5.5*columns+1, 4.8*rows+1),
                             constrained_layout=True, squeeze=False)
    axes = axes.ravel()
    levels = np.linspace(-AMPLITUDE, AMPLITUDE, 21)
    contours = np.linspace(-AMPLITUDE, AMPLITUDE, 15)
    norm = BoundaryNorm(levels, plt.get_cmap('RdBu_r').N)
    images = []
    times = results[modes[0]]['time']
    for ax, mode in zip(axes, modes):
        data = results[mode]
        np.testing.assert_array_equal(times, data['time'])
        # A constant offset is a legitimate zero-frequency mode. Do not remove means.
        images.append(ax.imshow(data['eta'][0], origin='lower',
                      extent=(0, grid.Lx/1e3, 0, grid.Ly/1e3),
                      cmap='RdBu_r', norm=norm, aspect='equal'))
        omega = mode_frequency(grid, params, *mode)
        subtitle = 'Uniform offset: stationary' if omega == 0 else f'T = {2*np.pi/omega/3600:.2f} hours'
        ax.set(xlabel='x [km]', ylabel='y [km]', title=f'm = {mode[0]}, n = {mode[1]}\n{subtitle}')
    for ax in axes[len(modes):]:
        ax.set_visible(False)
    fig.colorbar(images[0], ax=list(axes[:len(modes)]), label='Height η [m]',
                 ticks=np.linspace(-AMPLITUDE, AMPLITUDE, 5), shrink=.85)
    X, Y = np.meshgrid(grid.x_c/1e3, grid.y_c/1e3)
    contour_sets = [None]*len(modes)
    heading = fig.suptitle('', fontsize=12)
    def update(i):
        for j, (ax, image, mode) in enumerate(zip(axes, images, modes)):
            field = results[mode]['eta'][i]
            image.set_data(field)
            if contour_sets[j] is not None:
                contour_sets[j].remove(); contour_sets[j] = None
            if np.ptp(field) > 1e-12:
                contour_sets[j] = ax.contour(X, Y, field, levels=contours,
                                            colors='k', linewidths=.45, alpha=.6)
        heading.set_text(f'Rectangular-basin seiche modes\nt = {times[i]/3600:.2f} hours')
    animation = FuncAnimation(fig, update, frames=len(times), interval=1000/args.fps, blit=False)
    animation.save(str(args.output_dir/'eta_06.gif'), writer='pillow', fps=args.fps, dpi=120)
    fig.savefig(args.output_dir/'experiment_06_comparison_last_frame.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    (args.output_dir/'experiment_06_comparison.json').write_text(json.dumps(dict(
         modes=[list(mode) for mode in modes], remove_mean=False,
         color_limits=[-AMPLITUDE, AMPLITUDE], time_seconds=times.tolist(), fps=args.fps,
         note='Mode (0,0) is a stationary uniform height offset, not an oscillating seiche.'), indent=2)+'\n')


def make_parser():
    import argparse
    def mode_pair(value):
        try:
            pair = tuple(int(x) for x in value.split(','))
        except ValueError:
            raise argparse.ArgumentTypeError('Use m,n with nonnegative integer indices.')
        if len(pair) != 2 or min(pair) < 0:
            raise argparse.ArgumentTypeError('Use m,n with nonnegative integer indices.')
        return pair
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--modes', nargs='+', type=mode_pair, default=[(1, 0), (1, 1)])
    parser.add_argument('--periods', type=float, default=3., help='Periods of the slowest selected oscillating mode.')
    parser.add_argument('--fps', type=float, default=10.)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'animations')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--replot', action='store_true')
    return parser


def run_experiment(args):
    if min(args.periods, args.fps) <= 0 or not args.modes:
        raise ValueError('Use positive duration/playback rate and at least one mode.')
    if len(set(args.modes)) != len(args.modes):
        raise ValueError('Select each mode only once.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    grid, params, duration, dt = setup_experiment(args)
    results = {mode: run_mode(args, grid, params, duration, dt, *mode) for mode in args.modes}
    render(args, grid, params, results)
    return results


if __name__ == '__main__':
    run_experiment(make_parser().parse_args())
