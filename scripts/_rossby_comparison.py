"""Matched linear/full-SW Gaussian experiments for the eta_11 family."""
from dataclasses import asdict, replace
import hashlib
import inspect
import json
import os
from pathlib import Path

import numpy as np

from shallowwater import ModelParams, make_grid, setup_initial_state, geostrophic_velocities_from_eta
from shallowwater import compute_dt_cfl, run_model, zero_forcing
from shallowwater.dynamics import enforce_bcs
from _experiment_tools import DAY, model_fingerprint
from _rossby_plotting import render_eta_animation
from _rossby_small_scales import initial_height

CASES = {
    'large': ('large_eddy', 3., '3 Ld', 'Large anticyclonic eddy'),
    'medium': ('medium_eddy', 1., 'Ld', 'Medium anticyclonic eddy'),
    'small': ('small_eddy', 1/3, 'Ld/3', 'Small anticyclonic eddy'),
    'tiny': ('tiny_eddy', .1, 'Ld/10', 'Tiny anticyclonic eddy'),
    'tiny-cyclonic': ('tiny_eddy_cyclonic', .1, 'Ld/10', 'Tiny cyclonic eddy'),
}


def fingerprint(state):
    digest = hashlib.sha256()
    for a in state:
        digest.update(np.ascontiguousarray(a).tobytes())
    return digest.hexdigest()


def case_setup(name, args):
    stem, factor, radius_label, label = CASES[name]
    tiny = name.startswith('tiny')
    nx, ny = ((24, 18) if args.smoke else
              ((192, 144) if args.quick else (256, 192))) if tiny else (
              (24, 18) if args.smoke else ((96, 64) if args.quick else (192, 128)))
    lx, ly = (800e3, 600e3) if tiny else (4e6, 3e6)
    grid = make_grid(nx, ny, lx, ly)
    params = ModelParams(H=100., f0=1e-4, beta=2e-11, y0=ly/2,
                         r=1/((120 if tiny else 15)*DAY), linear=True, Ah=0.)
    ld = np.sqrt(params.g*params.H)/params.f0
    if tiny:
        eta = initial_height(grid, ld)
    else:
        eta = setup_initial_state(grid, params, mode='gaussian_bump', amp=.15,
                                   R=factor*ld, x0=lx/2, y0=ly/2)[0]
        eta -= eta.mean()
    if name == 'tiny-cyclonic':
        eta = -eta
    u, v = geostrophic_velocities_from_eta(eta, grid, params,
                                          alpha=1. if tiny else .95, sponge=0 if tiny else 6)
    enforce_bcs(u, v)
    days = .05 if args.smoke else (args.days if tiny else 20.)
    if tiny:
        bound = compute_dt_cfl(grid, params, umax=2., cfl=.4)
        steps = int(np.ceil(days*DAY/bound))
        dt = days*DAY/steps
        stride = max(1, int(np.ceil(steps/(args.frames-1))))
    else:
        dt = compute_dt_cfl(grid, params, cfl=.5)
        steps = int(np.ceil(days*DAY/dt))
        stride = max(1, int(round(steps/199)))
    indices = np.unique(np.r_[np.arange(0, steps+1, stride), steps]).astype(int)
    if tiny and not args.smoke:
        # Resolve the first few inertial periods without storing every velocity frame.
        early_stride = max(1, int(round(.04*DAY/dt)))
        early = np.arange(0, min(steps, int(3*DAY/dt))+1, early_stride)
        indices = np.unique(np.r_[indices, early]).astype(int)
    return grid, params, (eta, u, v), dt, indices, (stem, factor*ld, radius_label, label)


def sampled_integration(grid, params, initial, dt, indices, *, existing=None):
    """Continue the autonomous model at integer steps; save only desired frames.

    A validated previous linear history can supply late frames. Only its missing
    early times need reintegration. No interpolation of fields is performed.
    """
    reusable = {}
    if existing is not None:
        old_steps = np.rint(existing['time']/dt).astype(int)
        if np.allclose(old_steps*dt, existing['time'], rtol=0., atol=1e-6):
            reusable = {int(n): i for i, n in enumerate(old_steps)}
    missing = [int(n) for n in indices if int(n) not in reusable]
    last_needed = max(missing, default=0)
    state = tuple(a.copy() for a in initial)
    heights = [state[0].copy()]
    current = 0
    max_speed = float(np.max(np.hypot(.5*(state[1][:, :-1]+state[1][:, 1:]),
                                       .5*(state[2][:-1]+state[2][1:]))))
    min_depth = float(np.min(params.H+state[0]))
    for n in indices[1:]:
        n = int(n)
        if n > last_needed and n in reusable:
            heights.append(existing['eta'][reusable[n]])
            continue
        count = n-current
        out = run_model(np.nextafter(count*dt, 0.), dt, grid, params, zero_forcing,
                        lambda g, p: tuple(a.copy() for a in state), save_every=count)
        if len(out['time']) != 2 or not np.isclose(out['time'][-1], count*dt):
            raise RuntimeError('Unexpected number of steps in a continuation segment.')
        state = tuple(np.asarray(out[k][-1]) for k in ('eta', 'u', 'v'))
        if not all(np.isfinite(a).all() for a in state):
            raise RuntimeError(f'Non-finite solution at {n*dt/DAY:g} days.')
        min_depth = min(min_depth, float(np.min(params.H+state[0])))
        max_speed = max(max_speed, float(np.max(np.hypot(.5*(state[1][:, :-1]+state[1][:, 1:]),
                                                       .5*(state[2][:-1]+state[2][1:])))))
        if min_depth <= 0:
            raise RuntimeError('Non-positive thickness in a Rossby comparison.')
        heights.append(state[0].copy())
        if (current*dt)//(10*DAY) < (n*dt)//(10*DAY):
            print(f'  {"linear" if params.linear else "nonlinear"}: day {n*dt/DAY:.1f}', flush=True)
        current = n
    eta = np.asarray(heights)
    data = dict(eta=eta, time=indices*dt, x=grid.x_c, y=grid.y_c,
                initial_u=initial[1], initial_v=initial[2])
    diagnostics = dict(maximum_newly_integrated_speed=max_speed, minimum_sampled_thickness=min_depth,
                       maximum_mean_height_drift=float(np.max(np.abs(eta.mean(axis=(1, 2))-eta[0].mean()))),
                       reused_linear_frames=sum(int(n) in reusable and int(n) > last_needed for n in indices[1:]))
    return data, diagnostics


def atomic_save(path, data, metadata):
    temporary = path.with_name(f'{path.stem}.{os.getpid()}.tmp.npz')
    np.savez_compressed(temporary, **data)
    temporary.replace(path)
    meta = path.with_suffix('.json')
    temporary = meta.with_name(f'{meta.stem}.{os.getpid()}.tmp.json')
    temporary.write_text(json.dumps(metadata, indent=2)+'\n')
    temporary.replace(meta)


def previous_linear(folder, stem, grid, params, initial, dt, experiment_hash):
    """Reuse a validated reference, including exact sign reversal for the cyclone."""
    sign = -1. if stem == 'tiny_eddy_cyclonic' else 1.
    base = 'tiny_eddy' if stem == 'tiny_eddy_cyclonic' else stem
    candidates = [folder/f'experiment_11_{base}_linear.npz', folder/f'experiment_11_{base}.npz']
    for path in candidates:
        meta = path.with_suffix('.json')
        if not path.exists() or not meta.exists():
            continue
        config = json.loads(meta.read_text()).get('configuration', {})
        if config.get('solver_fingerprint') != model_fingerprint() or config.get('parameters') != asdict(params):
            continue
        if path == candidates[0] and (config.get('experiment_fingerprint') != experiment_hash
                                      or config.get('dt') != dt):
            continue
        with np.load(path, allow_pickle=False) as saved:
            eta = saved['eta']
            if eta.shape[1:] != (grid.Ny, grid.Nx) or not np.allclose(sign*eta[0], initial[0], rtol=0., atol=1e-14):
                continue
            if not (np.allclose(saved['x'], grid.x_c) and np.allclose(saved['y'], grid.y_c)):
                continue
            if 'initial_u' in saved and not all(np.array_equal(sign*saved[key], field)
                                                for key, field in zip(('initial_u', 'initial_v'), initial[1:])):
                continue
            return dict(eta=sign*eta, time=saved['time'].copy())
    return None


def comparison_diagnostics(results, folder, stem, *, polarity=1):
    """Describe amplitude changes and differences without calling them instabilities."""
    import matplotlib.pyplot as plt
    fields = {name: data['eta']-data['eta'].mean(axis=(1, 2), keepdims=True)
              for name, data in results.items()}
    time = results['linear']['time']/DAY
    rms = np.sqrt(np.mean((fields['nonlinear']-fields['linear'])**2, axis=(1, 2)))
    relative = rms/np.maximum(np.sqrt(np.mean(fields['linear']**2, axis=(1, 2))), 1e-15)
    peak = {name: np.max(polarity*eta, axis=(1, 2)) for name, eta in fields.items()}
    summary = dict(time_days=time.tolist(), height_rms_difference=rms.tolist(),
                   relative_height_rms_difference=relative.tolist(),
                   linear_peak_signed_height=peak['linear'].tolist(),
                   nonlinear_peak_signed_height=peak['nonlinear'].tolist(),
                   note='Heights are demeaned; signed height means eta for anticyclones and minus eta for cyclones. Peak height is a domain maximum, not a tracked material vortex.')
    (folder/f'experiment_11_{stem}_comparison.json').write_text(json.dumps(summary, indent=2)+'\n')
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), constrained_layout=True)
    for name in ('linear', 'nonlinear'):
        axes[0].plot(time, peak[name], label=name)
        early = time <= min(3., time[-1])
        axes[1].plot(time[early], peak[name][early], label=name)
    axes[0].set(title='Peak signed height', xlabel='Time [days]', ylabel='Height [m]')
    axes[1].set(title='Initial adjustment', xlabel='Time [days]', ylabel='Peak signed height [m]')
    axes[2].semilogy(time, np.maximum(relative, 1e-12))
    axes[2].set(title='Nonlinear / linear difference', xlabel='Time [days]', ylabel='RMS difference / linear RMS')
    for ax in axes:
        ax.grid(alpha=.2)
    axes[0].legend(); axes[1].legend()
    fig.savefig(folder/f'experiment_11_{stem}_comparison.png', dpi=150)
    plt.close(fig)
    return summary


def run_comparison(args, name):
    folder = Path(args.output_dir); folder.mkdir(parents=True, exist_ok=True)
    grid, params, initial, dt, indices, details = case_setup(name, args)
    stem, radius, radius_label, label = details
    results = {}
    source = '\n'.join(inspect.getsource(fn) for fn in
                       (case_setup, sampled_integration, initial_height))
    experiment_hash = hashlib.sha256(source.encode()).hexdigest()
    for kind, linear in [('linear', True), ('nonlinear', False)]:
        p = replace(params, linear=linear)
        config = dict(parameters=asdict(p), Nx=grid.Nx, Ny=grid.Ny, Lx=grid.Lx, Ly=grid.Ly,
                      radius=radius, polarity=-1 if name == 'tiny-cyclonic' else 1,
                      dt=dt, saved_step_indices=indices.tolist(), initial_fingerprint=fingerprint(initial),
                      solver_fingerprint=model_fingerprint(), experiment_fingerprint=experiment_hash)
        path = folder/f'experiment_11_{stem}_{kind}.npz'
        meta = path.with_suffix('.json')
        matching = path.exists() and meta.exists() and json.loads(meta.read_text())['configuration'] == config
        if args.replot and not matching:
            raise ValueError(f'No matching {kind} comparison cache for {stem}; run without --replot first.')
        if matching:
            with np.load(path, allow_pickle=False) as saved:
                data = {key: saved[key] for key in saved.files}
            print(f'11 {stem}: reuse {kind}', flush=True)
        else:
            existing = previous_linear(folder, stem, grid, params, initial, dt, experiment_hash) if linear else None
            print(f'11 {stem}: {kind}, {indices[-1]*dt/DAY:.1f} days'+
                  (' (reuse late linear history)' if existing is not None else ''), flush=True)
            data, diagnostics = sampled_integration(grid, p, initial, dt, indices, existing=existing)
            atomic_save(path, data, dict(configuration=config, diagnostics=diagnostics,
                        note='The initial fields are identical in both branches; full nonlinear SW retains momentum and thickness advection.'))
            print(json.dumps(diagnostics), flush=True)
        results[kind] = data
    np.testing.assert_allclose(results['linear']['time'], results['nonlinear']['time'], rtol=0., atol=1e-7)
    np.testing.assert_array_equal(results['linear']['eta'][0], results['nonlinear']['eta'][0])
    render_eta_animation(results['linear'], grid, folder, stem, label=label,
                         radius=radius, radius_label=radius_label, nonlinear=results['nonlinear'],
                         early_sampling=name.startswith('tiny') and not args.smoke)
    comparison_diagnostics(results, folder, stem, polarity=-1 if name == 'tiny-cyclonic' else 1)
    return results


def run_comparisons(args):
    selections = list(CASES) if args.case == 'all' else (
                 ['large', 'medium', 'small'] if args.case == 'existing' else [args.case])
    return {name: run_comparison(args, name) for name in selections}
