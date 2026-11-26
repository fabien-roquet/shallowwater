#!/usr/bin/env python3
"""Wind-driven reduced-gravity gyre: linear SW left, nonlinear SW right.

Run from the repository root or scripts/. Matched histories are cached in
animations/; --replot redraws them without repeating the integrations.
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
import numpy as np

from shallowwater import ModelParams, make_grid, setup_initial_state
from shallowwater import wind_gyre_forcing, run_model, compute_dt_cfl
from _experiment_tools import DAY, model_fingerprint


def save_every_for_target_frames(tmax, dt, n_frames, include_initial=True):
    """Compatibility helper for the original notebook's approximate cadence."""
    intervals = max(1, n_frames-int(include_initial))
    return max(1, int(round(np.ceil(tmax/dt)/intervals)))


def setup_experiment(args):
    n = 20 if args.smoke else (64 if args.quick else 128)
    grid = make_grid(n, n, 3e6, 3e6)
    params = ModelParams(H=500., g=.2, rho=1025., f0=1e-4, beta=2e-11,
                         y0=grid.Ly/2, r=3e-7, linear=True, Ah=500.,
                         Hmin_frac=.02, qmax=5e-4)
    duration = (.1 if args.smoke else args.days)*DAY
    frames = min(args.frames, 5) if args.smoke else args.frames
    # Use the original nonlinear case's conservative CFL for both branches.
    dt_max = compute_dt_cfl(grid, params, cfl=.25)
    stride = max(1, int(np.ceil(duration/dt_max/(frames-1))))
    steps = stride*(frames-1)
    return grid, params, duration, duration/steps, stride, frames


def run_case(args, name):
    grid, base, duration, dt, stride, frames = setup_experiment(args)
    params = replace(base, linear=name == 'linear')
    source = '\n'.join(inspect.getsource(fn) for fn in (setup_experiment, run_case))
    config = dict(Nx=grid.Nx, Ny=grid.Ny, Lx=grid.Lx, Ly=grid.Ly,
                  parameters=asdict(params), tau0=.1, duration=duration,
                  dt=dt, save_every=stride, frames=frames,
                  solver_fingerprint=model_fingerprint(),
                  experiment_fingerprint=hashlib.sha256(source.encode()).hexdigest())
    path = args.output_dir/f'experiment_12_{name}.npz'
    meta_path = path.with_suffix('.json')
    matching = path.exists() and meta_path.exists() and json.loads(meta_path.read_text())['configuration'] == config
    if args.replot and not matching:
        raise ValueError(f'No matching {name} history; run without --replot first.')
    if matching:
        with np.load(path, allow_pickle=False) as saved:
            data = {key: saved[key] for key in saved.files}
        print(f'12 {name}: reuse saved simulation', flush=True)
        return grid, data
    print(f'12 {name}: {duration/DAY:g} days, {grid.Nx} × {grid.Ny}, {frames} snapshots', flush=True)
    out = run_model(np.nextafter(duration, 0.), dt, grid, params,
                    lambda t, g, p: wind_gyre_forcing(t, g, p, tau0=.1),
                    lambda g, p: setup_initial_state(g, p, mode='rest'),
                    save_every=stride, show_progress=args.progress,
                    progress_kwargs={'desc': f'{name} wind gyre', 'unit': 'step'})
    if len(out['time']) != frames:
        raise RuntimeError('Unexpected output cadence.')
    if not all(np.isfinite(np.asarray(out[key])).all() for key in ('eta', 'u', 'v')):
        raise RuntimeError(f'Non-finite {name} simulation.')
    eta, u, v = (np.asarray(out[key]) for key in ('eta', 'u', 'v'))
    if np.min(params.H+eta) <= 0:
        raise RuntimeError('Non-positive layer thickness.')
    uc, vc = .5*(u[..., :-1]+u[..., 1:]), .5*(v[:, :-1]+v[:, 1:])
    data = dict(eta=eta, time=np.asarray(out['time']), x=grid.x_c, y=grid.y_c,
                mean_ke=.5*np.mean(uc**2+vc**2, axis=(1, 2)))
    stats = dict(maximum_speed=float(np.max(np.hypot(uc, vc))),
                 minimum_thickness=float(np.min(params.H+eta)),
                 maximum_mean_height_drift=float(np.max(np.abs(eta.mean(axis=(1, 2))))))
    np.savez_compressed(path, **data)
    meta_path.write_text(json.dumps(dict(configuration=config, diagnostics=stats), indent=2)+'\n')
    print(f'12 {name}: '+json.dumps(stats), flush=True)
    return grid, data


def archive_separate_movies(folder):
    """Retire the separate outputs once a combined comparison can be rendered."""
    nonlinear = folder/'eta_12b_nonlinear.gif'
    if not nonlinear.exists():
        return
    archive = folder/'archive'
    archive.mkdir(exist_ok=True)
    for source, basename in [(folder/'eta_12.gif', 'eta_12_previous.gif'),
                             (nonlinear, nonlinear.name)]:
        if not source.exists():
            continue
        destination = archive/basename
        index = 1
        while destination.exists():
            destination = archive/f'{Path(basename).stem}_{index}.gif'
            index += 1
        source.rename(destination)


def shared_limit(fields):
    """A fixed, common robust limit, rounded outward to a readable scale."""
    value = max(float(np.percentile(np.abs(a), 98.)) for a in fields)
    value = max(value, 1e-8)
    unit = 10.**np.floor(np.log10(value))
    return next(x*unit for x in (1., 2., 2.5, 5., 10.) if x*unit >= value)


def render_comparison(args, grid, results):
    times = results['linear']['time']
    np.testing.assert_allclose(times, results['nonlinear']['time'], rtol=0., atol=1e-7)
    fields = [results[name]['eta']-results[name]['eta'].mean(axis=(1, 2), keepdims=True)
              for name in ('linear', 'nonlinear')]
    limit = shared_limit(fields)
    levels = np.linspace(-limit, limit, 21)
    norm = BoundaryNorm(levels, plt.get_cmap('RdBu_r').N)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
    images = []
    for ax, field, title in zip(axes, fields, ['Linear SW', 'Nonlinear SW']):
        images.append(ax.imshow(field[0], origin='lower', extent=(0, grid.Lx/1e3, 0, grid.Ly/1e3),
                                 cmap='RdBu_r', norm=norm, aspect='equal'))
        ax.set(xlabel='x [km]', ylabel='y [km]', title=title)
    fig.colorbar(images[0], ax=list(axes), label='η [m]', extend='both',
                 ticks=np.linspace(-limit, limit, 9), shrink=.9)
    X, Y = np.meshgrid(grid.x_c/1e3, grid.y_c/1e3)
    contour_sets = [None, None]
    heading = fig.suptitle('', fontsize=12)
    def update(i):
        for j, (ax, im, field) in enumerate(zip(axes, images, fields)):
            im.set_data(field[i])
            if contour_sets[j] is not None:
                contour_sets[j].remove()
            contour_sets[j] = ax.contour(X, Y, field[i], levels=levels, colors='k',
                                        linewidths=.45, alpha=.65)
        heading.set_text(f'Wind-driven reduced-gravity gyre: identical wind and initial rest state\nt = {times[i]/DAY:.1f} days')
    archive_separate_movies(args.output_dir)
    animation = FuncAnimation(fig, update, frames=len(times), interval=100, blit=False)
    animation.save(str(args.output_dir/'eta_12.gif'), writer='pillow', fps=10, dpi=120)
    fig.savefig(args.output_dir/'experiment_12_comparison_last_frame.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    (args.output_dir/'experiment_12_comparison.json').write_text(json.dumps(dict(
        panels=['linear', 'nonlinear'], color_limits=[-limit, limit], contour_levels=levels.tolist(),
        time_days=(times/DAY).tolist(), note='One common fixed scale and identical physical times; height is demeaned for display.'), indent=2)+'\n')
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for name, data in results.items():
        ax.plot(data['time']/DAY, data['mean_ke'], label=name)
    ax.set(xlabel='Time [days]', ylabel='Mean kinetic energy [m²/s²]', title='Reduced-gravity wind gyre')
    ax.grid(alpha=.2); ax.legend()
    fig.savefig(args.output_dir/'experiment_12_comparison_diagnostics.png', dpi=150)
    plt.close(fig)


def make_parser():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('both', 'linear', 'nonlinear'), default='both')
    parser.add_argument('--days', type=float, default=730.)
    parser.add_argument('--frames', type=int, default=121)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'animations')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--replot', action='store_true')
    parser.add_argument('--progress', action='store_true')
    return parser


def run_experiment(args):
    if args.days <= 0 or args.frames < 2:
        raise ValueError('Use positive days and at least two frames.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names = ('linear', 'nonlinear') if args.case == 'both' else (args.case,)
    results = {}
    for name in names:
        grid, results[name] = run_case(args, name)
    if args.case == 'both':
        render_comparison(args, grid, results)
    return results


if __name__ == '__main__':
    run_experiment(make_parser().parse_args())
