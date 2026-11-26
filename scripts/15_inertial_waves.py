#!/usr/bin/env python3
"""Inertial rotation and propagating inertia–gravity waves in shallow water.

All animations integrate the existing linear shallow-water equations. The
channel is periodic in x and closed in y; no uniform cross-wall flow is used.
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

from shallowwater import ModelParams, make_grid, run_model, compute_dt_cfl, zero_forcing
from _experiment_tools import model_fingerprint

FREF = 1e-4
TF = 2*np.pi/FREF
AMPLITUDE = .15


def standing_initial(grid, params, amplitude=AMPLITUDE):
    """A zero-PV standing Poincaré mode; normal velocity vanishes at the walls."""
    eta = np.zeros((grid.Ny, grid.Nx))
    u = np.zeros((grid.Ny, grid.Nx+1))
    v = np.broadcast_to(amplitude*np.sin(np.pi*grid.y_v[:, None]/grid.Ly),
                        (grid.Ny+1, grid.Nx)).copy()
    v[0] = v[-1] = 0.
    return eta, u, v


def standing_frequency(grid, params, *, discrete=False):
    ell = np.pi/grid.Ly
    coriolis = params.f0
    if discrete:
        coriolis *= np.cos(ell*grid.dy/2)
        ell = 2*np.sin(ell*grid.dy/2)/grid.dy
    return np.sqrt(coriolis**2+params.g*params.H*ell**2)


def standing_solution(grid, params, time, amplitude=AMPLITUDE):
    """Exact solution of the spatially discretized, continuous-time C-grid mode."""
    ell = np.pi/grid.Ly
    ktilde = 2*np.sin(ell*grid.dy/2)/grid.dy
    ftilde = params.f0*np.cos(ell*grid.dy/2)
    omega = standing_frequency(grid, params, discrete=True)
    eta = np.broadcast_to(-params.H*ktilde*amplitude/omega*np.cos(ell*grid.y_c[:, None])
                          *np.sin(omega*time), (grid.Ny, grid.Nx)).copy()
    u = np.broadcast_to(ftilde*amplitude/omega*np.sin(ell*grid.y_c[:, None])
                        *np.sin(omega*time), (grid.Ny, grid.Nx+1)).copy()
    v = np.broadcast_to(amplitude*np.sin(ell*grid.y_v[:, None])*np.cos(omega*time),
                        (grid.Ny+1, grid.Nx)).copy()
    v[0] = v[-1] = 0.
    return eta, u, v


def packet_initial(grid, params, amplitude=AMPLITUDE):
    """An initial meridional velocity packet, with zero height, u and linear PV."""
    ld = np.sqrt(params.g*params.H)/FREF
    distance = grid.y_v-grid.Ly/2
    profile = (amplitude*np.exp(-(distance/(2*ld))**2)*np.cos(distance/ld)
               *np.sin(np.pi*grid.y_v/grid.Ly))
    v = np.broadcast_to(profile[:, None], (grid.Ny+1, grid.Nx)).copy()
    v[0] = v[-1] = 0.
    return np.zeros((grid.Ny, grid.Nx)), np.zeros((grid.Ny, grid.Nx+1)), v


def wave_energy(data, params):
    """Discrete energy per density and basin area; count the periodic seam once."""
    eta, u, v = (np.asarray(data[k]) for k in ('eta', 'u', 'v'))
    cells = eta.shape[-2]*eta.shape[-1]
    kinetic = .5*params.H*(np.sum(u[..., :-1]**2, axis=(-2, -1))
                           +np.sum(v[..., 1:-1, :]**2, axis=(-2, -1)))/cells
    return kinetic+.5*params.g*np.mean(eta**2, axis=(-2, -1))


def integrate_case(grid, params, initial, duration, frames):
    """Use exact common output times; no drag, forcing or background circulation."""
    dt_bound = compute_dt_cfl(grid, params, umax=AMPLITUDE, cfl=.4)
    # The gravity-wave CFL alone does not resolve rotation on very coarse grids.
    dt_bound = min(dt_bound, .06/max(FREF, abs(params.f0)))
    stride = max(1, int(np.ceil(duration/dt_bound/(frames-1))))
    steps = stride*(frames-1)
    dt = duration/steps
    out = run_model(np.nextafter(duration, 0.), dt, grid, params, zero_forcing,
                    lambda g, p: tuple(a.copy() for a in initial), save_every=stride)
    data = {k: np.asarray(out[k]) for k in ('time', 'eta', 'u', 'v')}
    if not all(np.isfinite(a).all() for a in data.values()):
        raise RuntimeError('Non-finite wave solution.')
    if len(data['time']) != frames:
        raise RuntimeError('Unexpected output cadence.')
    data.update(x=grid.x_c, y=grid.y_c)
    data['energy'] = wave_energy(data, params)
    return data


def load_case(args, name, *, a=.2, f=FREF, packet=False, periods=2.):
    ld = np.sqrt(9.81*100.)/FREF
    ny = 32 if args.smoke else (64 if args.quick else (384 if packet else 96))
    ly = 24*ld if packet else np.pi*ld/a
    grid = make_grid(12, ny, 4*ly/3, ly, periodic_x=True)
    params = ModelParams(H=100., f0=f, beta=0., y0=ly/2, r=0., Ah=0., linear=True)
    frames = 9 if args.smoke else args.frames
    duration = (.05 if args.smoke else periods)*TF
    initial_fn = packet_initial if packet else standing_initial
    digest = hashlib.sha256('\n'.join(inspect.getsource(fn) for fn in
                            (initial_fn, integrate_case)).encode()).hexdigest()
    config = dict(Nx=grid.Nx, Ny=grid.Ny, Lx=grid.Lx, Ly=grid.Ly,
                  parameters=asdict(params), duration=duration, frames=frames,
                  amplitude=AMPLITUDE, initial='packet' if packet else 'standing',
                  solver_fingerprint=model_fingerprint(), experiment_fingerprint=digest)
    path = args.output_dir/f'experiment_15_{name}.npz'
    meta = path.with_suffix('.json')
    matching = path.exists() and meta.exists() and json.loads(meta.read_text())['configuration'] == config
    if args.replot and not matching:
        raise ValueError(f'No matching saved run for {name}; use its original resolution and frame settings.')
    if matching:
        with np.load(path, allow_pickle=False) as saved:
            data = {k: saved[k] for k in saved.files}
        print(f'15 {name}: reuse saved run', flush=True)
    else:
        print(f'15 {name}: integrate {duration/3600:.1f} hours', flush=True)
        data = integrate_case(grid, params, initial_fn(grid, params), duration, frames)
        np.savez_compressed(path, **data)
        stats = dict(maximum_relative_energy_change=float(np.max(np.abs(data['energy']/data['energy'][0]-1))),
                     maximum_mean_height_drift=float(np.max(np.abs(data['eta'].mean(axis=(1, 2))))))
        if not packet:
            stats.update(omega_over_fref=float(standing_frequency(grid, params)/FREF),
                         maximum_eta_error=float(max(np.max(np.abs(data['eta'][i]-standing_solution(grid, params, t)[0]))
                                                     for i, t in enumerate(data['time']))))
        meta.write_text(json.dumps(dict(configuration=config, diagnostics=stats), indent=2)+'\n')
        print(json.dumps(stats), flush=True)
    return grid, params, data


def centered(data):
    return (.5*(data['u'][..., :-1]+data['u'][..., 1:]),
            .5*(data['v'][..., :-1, :]+data['v'][..., 1:, :]))


def map_panel(fig, ax, case, title, limit):
    grid, params, data = case
    levels = np.linspace(-limit, limit, 21)
    norm = BoundaryNorm(levels, plt.get_cmap('RdBu_r').N)
    im = ax.imshow(data['eta'][0], origin='lower', extent=(0, grid.Lx/1e3, 0, grid.Ly/1e3),
                   cmap='RdBu_r', norm=norm, aspect='equal')
    ax.set(xlabel='x [km] (periodic)', ylabel='y [km]')
    ax.set_title(title, fontsize=10, pad=22)
    fig.colorbar(im, ax=ax, label='Height η [m]', extend='both',
                 ticks=np.linspace(-limit, limit, 5), shrink=.8)
    rows = np.unique(np.linspace(3, grid.Ny-4, 9).astype(int))
    cols = np.arange(0, grid.Nx, 2)
    X, Y = np.meshgrid(grid.x_c[cols]/1e3, grid.y_c[rows]/1e3)
    uc, vc = centered(data)
    quiver = ax.quiver(X, Y, uc[0][np.ix_(rows, cols)], vc[0][np.ix_(rows, cols)],
                       angles='uv', scale_units='width', scale=2., width=.0045, color='0.15')
    ax.quiverkey(quiver, .93, 1.02, AMPLITUDE, '0.15 m/s', labelpos='W', coordinates='axes', fontproperties={'size': 8})
    def update(i):
        im.set_data(data['eta'][i])
        quiver.set_UVC(uc[i][np.ix_(rows, cols)], vc[i][np.ix_(rows, cols)])
    return update


def save_animation(args, name, fig, times, updates, caption):
    heading = fig.suptitle('', fontsize=12)
    def update(i):
        for callback in updates:
            callback(i)
        heading.set_text(f'{caption}\nt = {times[i]/3600:.1f} h = {times[i]/TF:.2f} reference inertial periods')
    animation = FuncAnimation(fig, update, frames=len(times), interval=100, blit=False)
    animation.save(str(args.output_dir/f'{name}.gif'), writer='pillow', fps=12, dpi=110)
    fig.savefig(args.output_dir/f'{name}_last_frame.png', dpi=150)
    plt.close(fig)


def rotation(args):
    cases = [load_case(args, 'rotation_north', f=FREF), load_case(args, 'rotation_south', f=-FREF)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    updates = []
    for col, (case, label) in enumerate(zip(cases, ['Northern hemisphere: clockwise', 'Southern hemisphere: anticlockwise'])):
        updates.append(map_panel(fig, axes[0, col], case, label, .1))
        grid, params, data = case
        uc, vc = centered(data)
        u = uc[:, grid.Ny//2, 0]; v = vc[:, grid.Ny//2, 0]
        ax = axes[1, col]
        ax.plot(u, v, color='0.65', lw=1.5, label='SW trajectory in velocity space')
        phase = np.linspace(0, 2*np.pi, 241)
        ax.plot(AMPLITUDE*np.sin(phase), AMPLITUDE*np.cos(phase), 'k:', lw=1,
                label='Pure inertial circle (reference)')
        point, = ax.plot([], [], 'o', color='tab:red', ms=7)
        needle = ax.quiver([0], [0], [u[0]], [v[0]], angles='xy', scale_units='xy', scale=1, color='tab:red')
        ax.set(xlim=(-.18, .18), ylim=(-.18, .18), xlabel='Eastward velocity u [m/s]',
               ylabel='Northward velocity v [m/s]', aspect='equal', title='Velocity at the channel centre')
        ax.axhline(0, color='0.85', lw=.7); ax.axvline(0, color='0.85', lw=.7)
        ax.legend(loc='lower left', fontsize=7)
        def update(i, point=point, needle=needle, u=u, v=v):
            point.set_data([u[i]], [v[i]]); needle.set_UVC([u[i]], [v[i]])
        updates.append(update)
    save_animation(args, 'velocity_15_inertial_rotation', fig, cases[0][2]['time'], updates,
                   'Near-inertial motion: K Ld = 0.2, ω/|f| ≈ 1.02; Tf = 17.45 h')


def regimes(args):
    values = [.2, 1., 5.]
    cases = [load_case(args, f'regime_{a:g}', a=a, periods=3.) for a in values]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    updates = []
    for ax, case, a in zip(axes, cases, values):
        ratio = np.sqrt(1+a*a)
        updates.append(map_panel(fig, ax, case, f'K Ld = {a:g}; ω/f = {ratio:.2f}\nT/Tf = {1/ratio:.2f}', .5))
    save_animation(args, 'eta_15_poincare_regimes', fig, cases[0][2]['time'], updates,
                   'Standing inertia–gravity waves: near-inertial → mixed → gravity dominated')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    a = np.linspace(0, 6, 400)
    axes[0].plot(a, np.sqrt(1+a*a), label='ω/|f| = √(1 + (K Ld)²)')
    axes[0].axhline(1, color='0.5', ls=':', label='Inertial frequency')
    axes[0].scatter(values, np.sqrt(1+np.array(values)**2), color='tab:red')
    axes[0].set(xlabel='K Ld', ylabel='ω / |f|', title='Dispersion relation')
    for a, case in zip(values, cases):
        data = case[2]
        axes[1].plot(data['time']/TF, data['v'][:, case[0].Ny//2, 0]/AMPLITUDE, label=f'K Ld = {a:g}')
    axes[1].set(xlabel='Time / Tf', ylabel='Central v / initial amplitude', title='Different frequencies, same initial velocity')
    for ax in axes:
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.savefig(args.output_dir/'experiment_15_dispersion.png', dpi=150)
    plt.close(fig)


def packet(args):
    cases = [load_case(args, 'packet_rotating', packet=True, periods=1.25),
             load_case(args, 'packet_nonrotating', f=0., packet=True, periods=1.25)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    updates = []
    for col, (case, label) in enumerate(zip(cases, ['Rotating: f = 10⁻⁴ s⁻¹', 'Nonrotating: f = 0'])):
        updates.append(map_panel(fig, axes[0, col], case, label, .25))
        grid, params, data = case
        ax = axes[1, col]
        levels = np.linspace(-.25, .25, 21)
        im = ax.pcolormesh(grid.y_c/1e3, data['time']/3600, data['eta'][:, :, 0],
                           cmap='RdBu_r', norm=BoundaryNorm(levels, plt.get_cmap('RdBu_r').N), shading='auto')
        fig.colorbar(im, ax=ax, label='Height η [m]', extend='both')
        ax.set(xlabel='y [km]', ylabel='Time [h]', title='Height along the channel')
        cursor = ax.axhline(0, color='k', lw=1)
        def update(i, cursor=cursor, times=data['time']):
            cursor.set_ydata([times[i]/3600]*2)
        updates.append(update)
    save_animation(args, 'eta_15_wave_packet', fig, cases[0][2]['time'], updates,
                   'An initial velocity packet splits and radiates waves; β = 0, no forcing or drag')


def make_parser():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('all', 'rotation', 'regimes', 'packet'), default='all')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'animations')
    parser.add_argument('--frames', type=int, default=181)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--replot', action='store_true')
    return parser


def run_experiment(args):
    if args.frames < 2:
        raise ValueError('At least two frames are required.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, function in [('rotation', rotation), ('regimes', regimes), ('packet', packet)]:
        if args.case in ('all', name):
            function(args)


if __name__ == '__main__':
    run_experiment(make_parser().parse_args())
