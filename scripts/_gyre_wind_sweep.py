"""Continue a validated gyre while increasing only its wind amplitude."""
from dataclasses import asdict, replace
import hashlib
import json
import numpy as np

from _experiment_tools import DAY, model_fingerprint
from _gyre_wind_ramp import (
    wind_stress, ramp_fraction, ramp_diagnostics, forcing_mixing_fingerprint,
    render_wind_ramp,
)
from shallowwater import ModelParams, make_grid, make_wind_forcing, compute_dt_cfl, run_model


def state_fingerprint(state):
    digest = hashlib.sha256()
    for field in state:
        digest.update(np.ascontiguousarray(field).tobytes())
    return digest.hexdigest()


def stage_fraction(time, previous, target, ramp_days):
    if not np.all(np.isfinite([previous, target, ramp_days])) or target < previous or previous < 0 or target <= 0 or ramp_days <= 0:
        raise ValueError('Use positive duration and nondecreasing positive wind levels.')
    return previous/target+(1-previous/target)*ramp_fraction(time, ramp_days)


def integrate_stage(initial, grid, params, previous, target, *, ramp_days, hold_days, frames=121):
    """Identical fixed time steps for the two models throughout a wind stage."""
    if hold_days <= 0 or ramp_days <= 0:
        raise ValueError('Ramp and hold durations must be positive.')
    stress = wind_stress(grid, target, profile='double-gyre')
    envelope = lambda t: float(stage_fraction(t, previous, target, ramp_days))
    forcing = make_wind_forcing(stress, np.zeros_like(stress), grid, envelope=envelope)
    duration = (ramp_days+hold_days)*DAY
    dt0 = compute_dt_cfl(grid, params, umax=8., cfl=.4)
    dt0 = min(dt0, .2/(params.Ah*(grid.dx**-2+grid.dy**-2))) if params.Ah else dt0
    steps = int(np.ceil(duration/dt0)); dt = duration/steps
    out = run_model(np.nextafter(duration, 0.), dt, grid, params, forcing,
                    lambda g, p: tuple(a.copy() for a in initial),
                    save_every=max(1, int(np.ceil(steps/max(1, frames-1)))))
    for key in ('eta', 'u', 'v'):
        if not np.isfinite(np.asarray(out[key])).all():
            raise RuntimeError(f'Non-finite {key} at wind {target:g}.')
    if np.min(params.H+np.asarray(out['eta'])) <= 0:
        raise RuntimeError('Non-positive layer thickness during wind sweep.')
    fractions = stage_fraction(np.asarray(out['time']), previous, target, ramp_days)
    return ramp_diagnostics(out, grid, params, stress, ramp_days, wind_fractions=fractions)


def instability_indicators(data, ramp_days, hold_days):
    """Screen an interior part of the hold, avoiding temporal-filter endpoints.

    This selects candidates for a fixed-wind confirmation and visual inspection;
    it is not an eigenvalue estimate or an exact critical-stress calculation.
    """
    t = np.asarray(data['time_days'])
    hold = (t >= ramp_days+.25*hold_days) & (t <= ramp_days+.85*hold_days)
    if not np.any(hold):
        raise ValueError('The hold contains too few saved times.')
    variance = float(np.mean(data['temporal_eddy_ke'][hold]))
    energy = float(np.mean(data['mean_ke'][hold]))
    production = float(np.mean(data['shear_production'][hold]))
    west = np.sqrt(np.mean(data['western_current_rms'][hold]**2))
    east = np.sqrt(np.mean(data['eastern_current_rms'][hold]**2))
    ratio = float(west/max(east, 1e-30))
    indices = np.flatnonzero(hold)
    width = max(1, len(indices)//3)
    growth = float(np.mean(data['temporal_eddy_ke'][indices[-width:]]) /
                   max(float(np.mean(data['temporal_eddy_ke'][indices[:width]])), 1e-30))
    return dict(hold_variance_growth=growth, eddy_energy_fraction=variance/max(energy, 1e-30),
                temporal_eddy_ke=variance, mean_ke=energy, shear_production=production,
                west_east_ratio=ratio, maximum_speed=float(np.max(data['max_speed'])),
                candidate=bool((variance/max(energy, 1e-30) > 1e-3 or
                                (growth > 3. and variance/max(energy, 1e-30) > 1e-6))
                               and production > 0 and ratio > 1.))


def join_histories(chunks):
    """Join stage-local clocks and discard duplicate checkpoint frames."""
    first = chunks[0]
    result = {'x': first['x'], 'y': first['y']}
    dynamic = [key for key, value in first.items() if key not in result
               and key not in ('u', 'v') and np.ndim(value) and len(value) == len(first['time_days'])]
    for key in dynamic:
        pieces = []; offset = 0.
        for index, chunk in enumerate(chunks):
            values = chunk[key]
            if key == 'time_days':
                values = values+offset
            pieces.append(values if index == 0 else values[1:])
            offset += float(chunk['time_days'][-1])
        result[key] = np.concatenate(pieces, axis=0)
    # Filtering is done within stages; preserve each stage's actual window size.
    result['temporal_mean_window_days'] = np.concatenate([
        np.full(len(c['time_days'])-(i > 0), float(c['temporal_mean_window_days']))
        for i, c in enumerate(chunks)])
    return result


def load_baseline(folder):
    metadata = json.loads((folder/'experiment_14_wind_ramp.json').read_text())
    config = metadata['configuration']
    if config['solver_fingerprint'] != model_fingerprint():
        raise ValueError('The baseline uses a different solver; regenerate it before continuing.')
    if config['wind_profile'] != 'double-gyre' or config['boundary_viscosity'] != config['viscosity']:
        raise ValueError('Continue the broad-wind, uniform-viscosity baseline.')
    grid = make_grid(config['Nx'], config['Ny'], config['Lx'], config['Ly'])
    baseline = {}
    for name in ('linear', 'nonlinear'):
        with np.load(folder/f'experiment_14_wind_ramp_{name}.npz', allow_pickle=False) as saved:
            baseline[name] = {k: saved[k].copy() for k in saved.files}
    return metadata, grid, baseline


def sweep_parameters(metadata, name, args):
    params = ModelParams(**metadata['parameters'][name])
    if args.wind_drag_days is not None:
        if args.wind_drag_days <= 0:
            raise ValueError('The drag timescale must be positive.')
        params = replace(params, r=1/(args.wind_drag_days*DAY))
    if args.wind_viscosity is not None:
        if args.wind_viscosity < 0:
            raise ValueError('Viscosity must be nonnegative.')
        params = replace(params, Ah=args.wind_viscosity)
    return params


def cached_stage(folder, name, initial, grid, params, previous, target, ramp_days, hold_days, *, confirmation=False):
    token = f'{target:.3f}'.replace('.', 'p')+('_confirmation' if confirmation else '')
    path = folder/f'experiment_14_wind_sweep_{name}_{token}.npz'
    config = dict(previous=previous, target=target, ramp_days=ramp_days,
                  hold_days=hold_days, initial=state_fingerprint(initial),
                  parameters=asdict(params), solver_fingerprint=model_fingerprint(),
                  forcing_mixing_fingerprint=forcing_mixing_fingerprint())
    meta_path = path.with_suffix('.json')
    if path.exists() and meta_path.exists() and json.loads(meta_path.read_text())['configuration'] == config:
        with np.load(path, allow_pickle=False) as saved:
            data = {k: saved[k].copy() for k in saved.files}
        print(f'14 {name} wind sweep: reuse {token}', flush=True)
    else:
        print(f'14 {name} wind sweep: {previous:g} → {target:g} N/m²'+
              (' (fixed-wind confirmation)' if confirmation else ''), flush=True)
        data = integrate_stage(initial, grid, params, previous, target,
                               ramp_days=ramp_days, hold_days=hold_days)
        np.savez_compressed(path, **data)
        meta_path.write_text(json.dumps(dict(configuration=config), indent=2)+'\n')
    return data


def run_wind_sweep(args):
    folder = args.output_dir
    metadata, grid, base = load_baseline(folder)
    previous = float(metadata['configuration']['peak_stress'])
    initial = tuple(base['nonlinear'][k][-1] for k in ('eta', 'u', 'v'))
    params = sweep_parameters(metadata, 'nonlinear', args)
    cache_folder = folder
    if args.wind_drag_days is not None or args.wind_viscosity is not None:
        drag = 1/(params.r*DAY) if params.r else float('inf')
        cache_folder = folder/f'wind_sweep_drag{drag:g}_visc{params.Ah:g}'
        cache_folder.mkdir(exist_ok=True)
    chosen, stages = [], []
    rejected = None
    confirmed = False
    for target in args.wind_levels:
        if target < previous or (target == previous and chosen):
            raise ValueError('Wind levels must increase from the baseline maximum.')
        data = cached_stage(cache_folder, 'nonlinear', initial, grid, params, previous, target,
                            args.wind_stage_days, args.wind_hold_days)
        report = instability_indicators(data, args.wind_stage_days, args.wind_hold_days)
        print(json.dumps(dict(target=target, **report)), flush=True)
        if report['west_east_ratio'] < 1.:
            rejected = dict(target=target, **report)
            print('Western intensification lost; exclude this stage from the animation.', flush=True)
            break
        stages.append(dict(previous=previous, target=target, ramp_days=args.wind_stage_days,
                           hold_days=args.wind_hold_days, confirmation=False, **report))
        chosen.append(data)
        initial = tuple(data[k][-1] for k in ('eta', 'u', 'v'))
        previous = target
        if report['candidate']:
            check = cached_stage(cache_folder, 'nonlinear', initial, grid, params, target, target,
                                 30., 90., confirmation=True)
            result = instability_indicators(check, 30., 90.)
            print('Fixed-wind confirmation: '+json.dumps(result), flush=True)
            stages.append(dict(previous=target, target=target, ramp_days=30., hold_days=90.,
                               confirmation=True, **result))
            chosen.append(check)
            initial = tuple(check[k][-1] for k in ('eta', 'u', 'v'))
            confirmed = result['candidate']
            if confirmed:
                break
    histories = {'nonlinear': [base['nonlinear']]+chosen, 'linear': [base['linear']]}
    initial = tuple(base['linear'][k][-1] for k in ('eta', 'u', 'v'))
    params = sweep_parameters(metadata, 'linear', args)
    for stage in stages:
        data = cached_stage(cache_folder, 'linear', initial, grid, params, stage['previous'],
                            stage['target'], stage['ramp_days'], stage['hold_days'],
                            confirmation=stage['confirmation'])
        histories['linear'].append(data)
        initial = tuple(data[k][-1] for k in ('eta', 'u', 'v'))
    joined = {name: join_histories(chunks) for name, chunks in histories.items()}
    np.testing.assert_allclose(joined['linear']['time_days'], joined['nonlinear']['time_days'])
    common = ('time_days', 'x', 'y', 'wind_stress_peak')
    np.savez_compressed(folder/'experiment_14_wind_sweep.npz',
                        **{k: joined['linear'][k] for k in common},
                        **{name+'_'+k: value for name, data in joined.items()
                           for k, value in data.items() if k not in common})
    (folder/'experiment_14_wind_sweep.json').write_text(json.dumps(dict(
        baseline=metadata, stages=stages, rejected_stage=rejected,
        fixed_wind_variability_confirmed=confirmed,
        continuation_parameters={name: asdict(sweep_parameters(metadata, name, args))
                                 for name in ('linear', 'nonlinear')},
        note='Screening and fixed-wind confirmation do not replace visual inspection or convergence checks.'), indent=2)+'\n')
    archive = folder/'archive'
    archive.mkdir(exist_ok=True)
    movie = folder/f'eta_14_wind_step.{args.format}'
    baseline_movie = archive/f'eta_14_wind_step_baseline_0p040.{args.format}'
    if movie.exists() and not baseline_movie.exists():
        import shutil
        shutil.copy2(movie, baseline_movie)
    render_wind_sweep(args)


def render_wind_sweep(args):
    # Use the common two-panel renderer while preserving the baseline archive.
    render_wind_ramp(args, archive_name='experiment_14_wind_sweep')
