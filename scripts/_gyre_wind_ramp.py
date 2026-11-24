"""Wind ramp from rest: genuinely linear versus fully nonlinear basin dynamics."""
from dataclasses import asdict, replace
import json
import hashlib
import inspect
import numpy as np
import matplotlib.pyplot as plt

from _experiment_tools import (
    DAY, integrate, balanced_patch, diagnostics, velocities, animate_fields,
    model_fingerprint,
)
from shallowwater import ModelParams, make_grid, make_wind_forcing, setup_initial_state, zero_forcing
from shallowwater.dynamics import tendencies
from shallowwater.bathymetry import depth_on_u, depth_on_v


def ramp_fraction(t, ramp_days):
    return np.clip(np.asarray(t)/(ramp_days*DAY), 0., 1.)


def wind_stress(grid, peak, width=100e3, seed_fraction=.001, profile='double-gyre'):
    """Broad double-gyre wind by default; retain the narrow-jet comparison.

    The same 0.1-percent deterministic seed is applied to both dynamical models.
    """
    X, Y = np.meshgrid(grid.x_c, grid.y_c)
    if profile == 'double-gyre':
        basic = -np.cos(2*np.pi*Y/grid.Ly)
        seed = seed_fraction*np.sin(2*np.pi*Y/grid.Ly)*np.cos(2*np.pi*X/grid.Lx)
        return peak*(basic+seed)
    if profile != 'jet':
        raise ValueError('Wind profile must be double-gyre or jet.')
    y = (Y-grid.Ly/2)/width
    band = 1/np.cosh(y)**2
    basic = band-band.mean(axis=0, keepdims=True)
    basic /= np.max(np.abs(basic))
    seed = seed_fraction*band*np.tanh(y)*(np.cos(2*np.pi*X/(8*width))
              +.5*np.cos(2*np.pi*X/(12*width))+.25*np.cos(2*np.pi*X/(6*width)))
    return peak*(basic+seed)


def boundary_current_diagnostics(out, grid):
    """Compare the actual meridional currents, not the height-color extrema."""
    v = np.asarray(out['v'])
    middle = (grid.y_v >= .1*grid.Ly) & (grid.y_v <= .9*grid.Ly)
    west = grid.x_c < .15*grid.Lx
    east = grid.x_c > .85*grid.Lx
    west_rms = np.sqrt(np.mean(v[:, middle][:, :, west]**2, axis=(1, 2)))
    east_rms = np.sqrt(np.mean(v[:, middle][:, :, east]**2, axis=(1, 2)))
    return dict(western_current_rms=west_rms, eastern_current_rms=east_rms,
                west_east_current_ratio=np.divide(west_rms, east_rms,
                    out=np.ones_like(west_rms), where=east_rms > 1e-15))


def variable_viscosity_tendency(a, viscosity, dx, dy):
    """Conservative div(nu grad(a)); zero flux through the outer stencil edge."""
    out = np.zeros_like(a)
    flux_x = .5*(viscosity[:, 1:]+viscosity[:, :-1])*np.diff(a, axis=1)/dx
    out[:, :-1] += flux_x/dx
    out[:, 1:] -= flux_x/dx
    flux_y = .5*(viscosity[1:]+viscosity[:-1])*np.diff(a, axis=0)/dy
    out[:-1] += flux_y/dy
    out[1:] -= flux_y/dy
    return out


def coastal_viscosity_hook(grid, interior, boundary, width):
    """Additional viscosity near closed walls; do not damp surface height."""
    if boundary < interior or width <= 0 or grid.periodic_x:
        raise ValueError('Use a closed basin, positive width, and boundary viscosity >= interior.')
    def extra(x, y):
        X, Y = np.meshgrid(x, y)
        distance = np.minimum.reduce((X, grid.Lx-X, Y, grid.Ly-Y))
        return (boundary-interior)*np.exp(-(distance/width)**2)
    nu_u = extra(grid.x_u, grid.y_u)
    nu_v = extra(grid.x_v, grid.y_v)
    def hook(state, t, g, params):
        return (None, variable_viscosity_tendency(state['u'], nu_u, g.dx, g.dy),
                variable_viscosity_tendency(state['v'], nu_v, g.dx, g.dy))
    return hook


def forcing_mixing_fingerprint():
    sources = (ramp_fraction, wind_stress, coastal_viscosity_hook,
               variable_viscosity_tendency, integrate, balanced_patch)
    return hashlib.sha256(''.join(inspect.getsource(fn) for fn in sources).encode()).hexdigest()


def regime_summary(data, ramp_days):
    times = np.asarray(data['time_days'])
    late = times >= max(ramp_days, times[-1]-.25*times[-1])
    west = np.sqrt(np.mean(data['western_current_rms'][late]**2))
    east = np.sqrt(np.mean(data['eastern_current_rms'][late]**2))
    wind = np.sqrt(np.mean(data['wind_work'][late]**2))
    ratio = float(west/max(east, 1e-30))
    return dict(west_east_current_ratio=ratio, western_intensified=ratio > 1.,
                maximum_speed_m_s=float(np.max(data['max_speed'])),
                late_mean_kinetic_energy_m2_s2=float(np.mean(data['mean_ke'][late])),
                max_temporal_eddy_ke_m2_s2=float(np.max(data['temporal_eddy_ke'][late])),
                numerical_energy_defect_relative_to_wind_work=float(
                    np.sqrt(np.mean(data['inviscid_energy_defect'][late]**2))/max(wind, 1e-30)))


def temporal_mean(values, window):
    """Centred box average; endpoint estimates use edge padding."""
    half = window//2
    padded = np.pad(values, ((half, half), (0, 0), (0, 0)), mode='edge')
    accumulated = np.concatenate((np.zeros_like(padded[:1]), np.cumsum(padded, axis=0)))
    return (accumulated[window:]-accumulated[:-window])/window


def ramp_diagnostics(out, grid, params, stress, ramp_days, wind_fractions=None):
    data = diagnostics(out, grid, params)
    data.update(boundary_current_diagnostics(out, grid))
    u, v = velocities(out)
    times = np.asarray(out['time'])
    spacing = max(float(np.median(np.diff(times)))/DAY, 1e-10)
    window = min(len(times) if len(times)%2 else len(times)-1, max(1, int(round(20/spacing)) | 1))
    mean_u, mean_v = temporal_mean(u, window), temporal_mean(v, window)
    up, vp = u-mean_u, v-mean_v
    data['temporal_eddy_ke'] = .5*np.mean(up**2+vp**2, axis=(1, 2))
    production = []
    for U, V, a, b in zip(mean_u, mean_v, up, vp):
        Uy, Ux = np.gradient(U, grid.dy, grid.dx)
        Vy, Vx = np.gradient(V, grid.dy, grid.dx)
        transfer = -(a*a*Ux+a*b*(Uy+Vx)+b*b*Vy)
        production.append(np.mean(transfer[2:-2, 2:-2]))
    data['shear_production'] = np.asarray(production)
    data['temporal_mean_window_days'] = window*spacing
    f = params.f0+params.beta*(grid.y_c[:, None]-params.y0)
    data['min_absolute_vorticity'] = np.min(data['zeta']+f, axis=(1, 2))
    fractions = ramp_fraction(times, ramp_days) if wind_fractions is None else np.asarray(wind_fractions)
    data['wind_stress_peak'] = np.max(np.abs(stress))*fractions
    data['eta'] = np.asarray(out['eta'])
    # Retain face velocities so later perturbation/growth checks need no spin-up.
    data['u'], data['v'] = np.asarray(out['u']), np.asarray(out['v'])
    data['x'], data['y'] = grid.x_c, grid.y_c
    energy, defect, work = [], [], []
    inviscid = replace(params, r=0., Ah=0.)
    normalizer = params.H*grid.Nx*grid.Ny
    for i, (eta, uf, vf) in enumerate(zip(out['eta'], out['u'], out['v'])):
        h = params.H+eta if not params.linear else params.H
        hu, hv = depth_on_u(grid, h), depth_on_v(grid, h)
        kinetic = .5*(np.sum(hu*uf**2)+np.sum(hv*vf**2))/normalizer
        energy.append(kinetic+.5*params.g*np.sum(eta**2)/normalizer)
        de, du, dv = tendencies(dict(eta=eta, u=uf, v=vf), times[i], grid, inviscid, zero_forcing)
        gradient = params.g*eta
        if not params.linear:
            gradient = gradient+.25*(uf[:, :-1]**2+uf[:, 1:]**2+vf[:-1]**2+vf[1:]**2)
        defect.append((np.sum(hu*uf*du)+np.sum(hv*vf*dv)+np.sum(gradient*de))/normalizer)
        work.append(np.mean(stress*u[i])*fractions[i]/(params.rho*params.H))
    data['total_energy'] = np.asarray(energy)
    data['inviscid_energy_defect'] = np.asarray(defect)
    data['wind_work'] = np.asarray(work)
    return data


def render_wind_ramp(args, archive_name="experiment_14_wind_ramp"):
    legacy = args.output_dir/'experiment_14_wind_step.json'
    if legacy.exists():
        archive = args.output_dir/'archive'
        archive.mkdir(exist_ok=True)
        for old in (legacy, legacy.with_suffix('.npz'),
                    args.output_dir/'eta_14_wind_step.gif', args.output_dir/'eta_14_wind_step.mp4'):
            if old.exists():
                target = archive/old.name
                index = 1
                while target.exists():
                    target = archive/f'{old.stem}_{index}{old.suffix}'
                    index += 1
                old.rename(target)
    with np.load(args.output_dir/(archive_name+'.npz'), allow_pickle=False) as saved:
        grid = make_grid(len(saved['x']), len(saved['y']),
                         len(saved['x'])*(saved['x'][1]-saved['x'][0]),
                         len(saved['y'])*(saved['y'][1]-saved['y'][0]))
        times = saved['time_days']*DAY
        field = args.ramp_field
        key, label = ('eta', 'surface height η [m]') if field == 'eta' else ('zeta', 'relative vorticity [s⁻¹]')
        stress = saved['wind_stress_peak']
        selected = np.unique(np.linspace(0, len(times)-1, min(241, len(times))).astype(int))
        sweep_meta_path = args.output_dir/(archive_name+'.json')
        sweep_meta = json.loads(sweep_meta_path.read_text()) if archive_name.endswith('sweep') else None
        def subtitle(i):
            frame = selected[i]
            line = f'Identical wind: peak stress = {stress[frame]:.3f} N m⁻²'
            if sweep_meta:
                cutoff = sweep_meta['baseline']['configuration']['total_days']*DAY
                params = (sweep_meta['baseline']['parameters']['nonlinear'] if times[frame] <= cutoff
                          else sweep_meta['continuation_parameters']['nonlinear'])
                drag = 1/(params['r']*DAY) if params['r'] else float('inf')
                line += f"; drag {drag:g} days, viscosity {params['Ah']:g} m² s⁻¹"
            return line
        animate_fields(args.output_dir/f'eta_14_wind_step.{args.format}', grid, times[selected],
                       [saved['linear_'+key][selected], saved['nonlinear_'+key][selected]],
                       ['Linear approximation', 'Full shallow-water equations'],
                       labels=[label, label], scale_groups=[0, 0], contour_panels=(0, 1),
                       time_label='Time since wind ramp started',
                       subtitle=subtitle)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
        for name in ('linear', 'nonlinear'):
            axes[0].plot(times/DAY, saved[name+'_mean_ke'], label=name)
            axes[1].semilogy(times/DAY, np.maximum(saved[name+'_temporal_eddy_ke'], 1e-20), label=name)
        for name in ('linear', 'nonlinear'):
            axes[2].plot(times/DAY, saved[name+'_west_east_current_ratio'], label=name)
        axes[2].axhline(1., color='k', linestyle=':', linewidth=.8)
        axes[0].set(xlabel='Time [days]', ylabel='Mean kinetic energy [m²/s²]')
        axes[1].set(xlabel='Time [days]', ylabel='Variability about temporal mean [m²/s²]')
        axes[2].set(xlabel='Time [days]', ylabel='Western/eastern boundary-current RMS ratio')
        axes[2].legend()
        for ax in axes:
            ax.grid(alpha=.2)
        axes[0].legend(); axes[1].legend()
        if sweep_meta:
            wind_axis = axes[0].twinx()
            wind_axis.plot(times/DAY, stress, color='0.4', linestyle='--', linewidth=1.)
            wind_axis.set_ylabel('Peak wind stress [N/m²]', color='0.4')
            if sweep_meta['stages'] and sweep_meta['stages'][-1]['confirmation']:
                last = sweep_meta['stages'][-1]
                start = times[-1]/DAY-last['ramp_days']-last['hold_days']
                for ax in axes:
                    ax.axvspan(start, times[-1]/DAY, color='0.5', alpha=.12)
        fig.savefig(args.output_dir/(archive_name+'_diagnostics.png'), dpi=150)
        plt.close(fig)


def wind_ramp_configuration(args, grid, ramp_days, total_days, frames):
    return dict(Nx=grid.Nx, Ny=grid.Ny, Lx=grid.Lx, Ly=grid.Ly, H=200., g=9.81, f0=1e-4,
                  beta=args.ramp_beta, drag_days=args.ramp_drag_days, viscosity=args.ramp_viscosity,
                  ramp_days=ramp_days, total_days=total_days, peak_stress=args.ramp_tau_max,
                  frames=frames, seed_amplitude=1e-4,
                  boundary_viscosity=args.ramp_boundary_viscosity, boundary_width_m=args.ramp_boundary_width_km*1e3, wind_profile=args.ramp_wind_profile, wind_width_m=args.ramp_wind_width_km*1e3,
                  wind_seed_fraction=.001, diagnostics_revision=3,
                  forcing_mixing_fingerprint=forcing_mixing_fingerprint(), solver_fingerprint=model_fingerprint())


def run_wind_ramp(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    nx, ny = (20, 16) if args.smoke else ((96, 64) if args.quick else (128, 96))
    grid = make_grid(nx, ny, args.ramp_length_km*1e3, 1.6e6)
    ramp_days, total_days = (.1, .2) if args.smoke else (args.ramp_days, args.ramp_days+args.ramp_hold_days)
    frames = args.frames if args.smoke else max(args.frames, 161)
    config = wind_ramp_configuration(args, grid, ramp_days, total_days, frames)
    path = args.output_dir/'experiment_14_wind_ramp.npz'
    meta_path = path.with_suffix('.json')
    if args.ramp_case == 'both' and path.exists() and meta_path.exists() and json.loads(meta_path.read_text())['configuration'] == config:
        print('14 wind ramp: reuse matching saved simulations', flush=True)
        render_wind_ramp(args)
        return
    stress = wind_stress(grid, args.ramp_tau_max, args.ramp_wind_width_km*1e3,
                         profile=args.ramp_wind_profile)
    forcing = make_wind_forcing(stress, np.zeros_like(stress), grid,
                               envelope=lambda t: ramp_fraction(t, ramp_days))
    results = {}
    parameter_sets = {}
    hooks = []
    if args.ramp_boundary_viscosity > args.ramp_viscosity:
        hooks = [coastal_viscosity_hook(grid, args.ramp_viscosity, args.ramp_boundary_viscosity,
                                        args.ramp_boundary_width_km*1e3)]
    for name, linear in (('linear', True), ('nonlinear', False)):
        if args.ramp_case != 'both' and name != args.ramp_case:
            continue
        params = ModelParams(H=200., f0=1e-4, beta=args.ramp_beta, y0=grid.Ly/2,
                             r=1/(args.ramp_drag_days*DAY), Ah=args.ramp_viscosity,
                             linear=linear, Hmin_frac=0.)
        parameter_sets[name] = asdict(params)
        case_path = args.output_dir/f'experiment_14_wind_ramp_{name}.npz'
        case_meta = case_path.with_suffix('.json')
        if (case_path.exists() and case_meta.exists()
                and json.loads(case_meta.read_text())['configuration'] == config):
            with np.load(case_path, allow_pickle=False) as saved:
                results[name] = {key: saved[key].copy() for key in saved.files}
            print(f'14 wind ramp {name}: reuse matching saved simulation', flush=True)
            continue
        initial = balanced_patch(grid, params, setup_initial_state(grid, params),
                                 amplitude=1e-4, radius=180e3, seed_mode=3)
        print(f'14 wind ramp {name}: {total_days:g} days, stress 0 → {args.ramp_tau_max:g} N/m²', flush=True)
        out = integrate(grid, params, forcing, initial, days=total_days, frames=frames,
                        hooks=hooks, diffusivity_bound=args.ramp_boundary_viscosity)
        results[name] = ramp_diagnostics(out, grid, params, stress, ramp_days)
        np.savez_compressed(case_path, **results[name])
        case_meta.write_text(json.dumps(dict(configuration=config, parameters=asdict(params),
                            regime=regime_summary(results[name], ramp_days)), indent=2)+'\n')
        print(f"  final maximum speed {results[name]['max_speed'][-1]:.3f} m/s", flush=True)
    if args.ramp_case != 'both':
        return
    np.testing.assert_allclose(results['linear']['time_days'], results['nonlinear']['time_days'])
    common = ('time_days', 'x', 'y', 'wind_stress_peak')
    if meta_path.exists() and json.loads(meta_path.read_text())['configuration'] != config:
        archive = args.output_dir/'archive'
        archive.mkdir(exist_ok=True)
        for old in (path, meta_path, args.output_dir/f'eta_14_wind_step.{args.format}',
                    args.output_dir/'experiment_14_wind_ramp_diagnostics.png'):
            if old.exists():
                destination = archive/f'{old.stem}_previous_regime{old.suffix}'
                index = 1
                while destination.exists():
                    destination = archive/f'{old.stem}_previous_regime_{index}{old.suffix}'
                    index += 1
                old.rename(destination)
    np.savez_compressed(path, **{key: results['linear'][key] for key in common},
                        **{name+'_'+key: value for name, data in results.items()
                           for key, value in data.items() if key not in common})
    meta_path.write_text(json.dumps(dict(configuration=config, parameters=parameter_sets,
                         regime={name: regime_summary(data, ramp_days) for name, data in results.items()},
                         note='Temporal variability and shear production are diagnostics, not a formal eigenmode threshold.'), indent=2)+'\n')
    render_wind_ramp(args)
