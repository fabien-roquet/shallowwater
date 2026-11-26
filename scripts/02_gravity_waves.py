#!/usr/bin/env python3
"""Gravity waves with and without rotation, synchronized in one animation.

eta_02.gif covers 1.5 days at 10 fps, saving twice as often to show motion
at half speed within the original movie length. Arrows show direction only.
"""
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
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable
import numpy as np

from shallowwater import ModelParams, make_grid, setup_initial_state, run_model, compute_dt_cfl, zero_forcing
from _experiment_tools import DAY, model_fingerprint


def setup_experiment(args):
    n = 16 if args.smoke else (32 if args.quick else 64)
    grid = make_grid(n, n, 2e6, 2e6)
    params = ModelParams(H=500., g=9.81, rho=1025., f0=1e-4, beta=0.,
                         y0=grid.Ly/2, r=1/(2*DAY), linear=True)
    duration = (.05 if args.smoke else args.days)*DAY
    dt = compute_dt_cfl(grid, params, cfl=.5)
    return grid, params, duration, dt


def initial_state(grid, params):
    return setup_initial_state(grid, params, mode='gaussian_bump', amp=.02,
                               R=2.5e5, x0=.5*grid.Lx, y0=grid.Ly)


def run_case(args, name):
    grid, base, duration, dt = setup_experiment(args)
    params = replace(base, f0=1e-4 if name == 'rotating' else 0.)
    stem = '02' if name == 'rotating' else '02b_no_rotation'
    path = args.output_dir/f'experiment_{stem}.npz'
    meta = path.with_suffix('.json')
    source = '\n'.join(inspect.getsource(fn) for fn in (setup_experiment, initial_state, run_case))
    config = dict(Nx=grid.Nx, Ny=grid.Ny, Lx=grid.Lx, Ly=grid.Ly,
                  parameters=asdict(params), duration=duration, dt=dt, save_every=2,
                  solver_fingerprint=model_fingerprint(),
                  experiment_fingerprint=hashlib.sha256(source.encode()).hexdigest())
    matching = path.exists() and meta.exists() and json.loads(meta.read_text())['configuration'] == config
    if args.replot and not matching:
        raise ValueError(f'No matching {name} history; run without --replot first.')
    if matching:
        with np.load(path, allow_pickle=False) as saved:
            data = {key: saved[key] for key in saved.files}
        print(f'02 {name}: reuse saved simulation', flush=True)
        return grid, data
    print(f'02 {name}: integrate {duration/DAY:g} days', flush=True)
    out = run_model(duration, dt, grid, params, zero_forcing, initial_state,
                    save_every=2, out_vars=('eta', 'u', 'v'))
    data = {key: np.asarray(out[key]) for key in ('time', 'eta', 'u', 'v')}
    if not all(np.isfinite(a).all() for a in data.values()):
        raise RuntimeError(f'Non-finite {name} simulation.')
    data.update(x=grid.x_c, y=grid.y_c)
    np.savez_compressed(path, **data)
    meta.write_text(json.dumps(dict(configuration=config), indent=2)+'\n')
    return grid, data


def archive_separate_movies(folder):
    separate = folder/'eta_02b_no_rotation.gif'
    if not separate.exists():
        return
    archive = folder/'archive'
    archive.mkdir(exist_ok=True)
    for source, basename in [(folder/'eta_02.gif', 'eta_02_previous.gif'), (separate, separate.name)]:
        if not source.exists():
            continue
        destination = archive/basename
        index = 1
        while destination.exists():
            destination = archive/f'{Path(basename).stem}_{index}.gif'
            index += 1
        source.rename(destination)


def direction_vectors(u, v):
    """Normalize local currents; hide rest and negligible numerical tails."""
    speed = np.hypot(u, v)
    threshold = np.maximum(1e-12, 1e-6*np.max(speed, axis=(-2, -1), keepdims=True))
    moving = speed > threshold
    return (np.ma.array(np.divide(u, speed, out=np.zeros_like(u), where=moving), mask=~moving),
            np.ma.array(np.divide(v, speed, out=np.zeros_like(v), where=moving), mask=~moving))


def render_comparison(args, grid, results):
    times = results['rotating']['time']
    np.testing.assert_array_equal(times, results['nonrotating']['time'])
    fields = [results[name]['eta']-results[name]['eta'].mean(axis=(1, 2), keepdims=True)
              for name in ('rotating', 'nonrotating')]
    bound = max(float(np.percentile(np.abs(field), 98.)) for field in fields)
    # Round outward while retaining the former saturated height coloring.
    unit = 10.**np.floor(np.log10(max(bound, 1e-12)))
    limit = next(a*unit for a in (1., 1.5, 2., 2.5, 3., 4., 5., 7.5, 10.) if a*unit >= bound)
    levels = np.linspace(-limit, limit, 21)
    contour_levels = np.linspace(-limit, limit, 13)
    norm = BoundaryNorm(levels, plt.get_cmap('RdBu_r').N)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    rows = np.arange(args.arrow_stride//2, grid.Ny, args.arrow_stride)
    cols = np.arange(args.arrow_stride//2, grid.Nx, args.arrow_stride)
    X, Y = np.meshgrid(grid.x_c/1e3, grid.y_c/1e3)
    # Extend centre values to the walls so contourf fills the whole basin.
    # Use the nearest cell height over the outer half-cell for plotting only.
    Xf, Yf = np.meshgrid(np.r_[0., grid.x_c/1e3, grid.Lx/1e3],
                         np.r_[0., grid.y_c/1e3, grid.Ly/1e3])
    def filled_height(ax, field):
        return ax.contourf(Xf, Yf, np.pad(field, 1, mode='edge'),
                           levels=levels, cmap='RdBu_r', norm=norm, extend='both')
    Xq, Yq = np.meshgrid(grid.x_c[cols]/1e3, grid.y_c[rows]/1e3)
    images, arrows, velocities = [], [], []
    for ax, field, name, title in zip(axes, fields, ('rotating', 'nonrotating'),
                    ('With rotation: f = 10⁻⁴ s⁻¹', 'Without rotation: f = 0')):
        data = results[name]
        u = .5*(data['u'][..., :-1]+data['u'][..., 1:])
        v = .5*(data['v'][:, :-1]+data['v'][:, 1:])
        u, v = direction_vectors(u, v)
        velocities.append((u[:, rows][:, :, cols], v[:, rows][:, :, cols]))
        images.append(filled_height(ax, field[0]))
        ax.set_aspect('equal')
        ax.set(xlabel='x [km]', ylabel='y [km]')
        ax.set_title(title+'\nArrows: current direction only', fontsize=11, pad=10)
        arrows.append(ax.quiver(Xq, Yq, velocities[-1][0][0], velocities[-1][1][0],
                       angles='uv', scale_units='width', scale=args.arrow_scale,
                       width=.0025, headwidth=4.5, headlength=6, headaxislength=5,
                       minlength=0, color='0.05', zorder=3))
    fig.colorbar(ScalarMappable(norm=norm, cmap='RdBu_r'), ax=list(axes),
                 label='η [m]', boundaries=levels, extend='both',
                 ticks=np.linspace(-limit, limit, 7), shrink=.9)
    heading = fig.suptitle('', fontsize=12)
    contours = [None, None]
    def update(i):
        for j, (ax, field, quiver, velocity) in enumerate(zip(axes, fields, arrows, velocities)):
            images[j].remove()
            images[j] = filled_height(ax, field[i])
            quiver.set_UVC(velocity[0][i], velocity[1][i])
            if contours[j] is not None:
                contours[j].remove()
            contours[j] = ax.contour(X, Y, field[i], levels=contour_levels,
                                     colors='k', linewidths=.4, alpha=.35)
        heading.set_text(f'Gravity waves: rotation versus no rotation\nt = {times[i]/3600:.1f} hours ({times[i]/DAY:.2f} days)')
    archive_separate_movies(args.output_dir)
    animation = FuncAnimation(fig, update, frames=len(times), interval=1000/args.fps, blit=False)
    animation.save(str(args.output_dir/'eta_02.gif'), writer='pillow', fps=args.fps, dpi=120)
    fig.savefig(args.output_dir/'experiment_02_comparison_last_frame.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    (args.output_dir/'experiment_02_comparison.json').write_text(json.dumps(dict(
        panels=['rotating', 'nonrotating'], color_limits=[-limit, limit],
        contour_levels=contour_levels.tolist(), filled_contour_levels=levels.tolist(),
        renderer='contourf', arrow_normalized=True,
        arrow_mask_absolute_m_s=1e-12, arrow_mask_relative_to_frame_max=1e-6,
        arrow_scale=args.arrow_scale, arrow_stride=args.arrow_stride, fps=args.fps,
        frame_count=len(times), playback_seconds=len(times)/args.fps,
        time_seconds=times.tolist()), indent=2)+'\n')


def make_parser():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('both', 'rotating', 'nonrotating'), default='both')
    parser.add_argument('--days', type=float, default=1.5)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'animations')
    parser.add_argument('--fps', type=float, default=10., help='Original frame rate; physical sampling is twice as dense.')
    parser.add_argument('--arrow-scale', type=float, default=24., help='Inverse length of normalized arrows in panel widths.')
    parser.add_argument('--arrow-stride', type=int, default=4)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--replot', action='store_true')
    return parser


def run_experiment(args):
    if min(args.days, args.fps, args.arrow_scale, args.arrow_stride) <= 0:
        raise ValueError('Duration, playback rate, arrow scale and spacing must be positive.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    grid, *_ = setup_experiment(args)
    if args.arrow_stride > min(grid.Nx, grid.Ny):
        raise ValueError('Arrow spacing must fit within the grid.')
    results = {}
    for name in (('rotating', 'nonrotating') if args.case == 'both' else (args.case,)):
        grid, results[name] = run_case(args, name)
    if args.case == 'both':
        render_comparison(args, grid, results)
    return results


if __name__ == '__main__':
    run_experiment(make_parser().parse_args())
