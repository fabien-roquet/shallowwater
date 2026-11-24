"""Resolved, sub-deformation-scale Rossby disturbances for experiment 11."""
from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _experiment_tools import DAY, integrate, model_fingerprint, velocities
from _rossby_plotting import render_eta_animation
from shallowwater import ModelParams, make_grid, geostrophic_velocities_from_eta, zero_forcing
from shallowwater.dynamics import enforce_bcs


def odd_extension(field):
    """Sine extension across the four walls, on the cell-centred grid."""
    row = np.concatenate((field, -field[:, ::-1]), axis=1)
    return np.concatenate((row, -row[::-1, :]), axis=0)


def wavenumbers(grid):
    k = 2*np.pi*np.fft.fftfreq(2*grid.Nx, grid.dx)
    ell = 2*np.pi*np.fft.fftfreq(2*grid.Ny, grid.dy)
    return np.hypot(k[None, :], ell[:, None])


def initial_height(grid, ld, *, filtered=False, amplitude=.15):
    X, Y = np.meshgrid(grid.x_c, grid.y_c)
    radius = ld/10
    eta = np.exp(-((X-grid.Lx/2)**2+(Y-grid.Ly/2)**2)/radius**2)
    if filtered:
        # No initial sine component with wavelength >= 2*pi*Ld/8 (~0.79 Ld).
        # A smooth transition reduces ringing; some rings remain unavoidable.
        scaled = wavenumbers(grid)*ld
        transition = np.clip((scaled-8.)/2., 0., 1.)
        window = .5-.5*np.cos(np.pi*transition)
        extended = np.fft.ifft2(np.fft.fft2(odd_extension(eta))*window).real
        eta = extended[:grid.Ny, :grid.Nx]
    return amplitude*eta/np.max(eta)


def spectral_diagnostics(eta, grid, ld):
    power = np.abs(np.fft.fft2(odd_extension(eta)))**2
    scaled = wavenumbers(grid)*ld
    return dict(initial_fraction_KLd_below_1=float(power[scaled < 1].sum()/power.sum()),
                initial_fraction_wavelength_above_Ld=float(power[scaled < 2*np.pi].sum()/power.sum()),
                initial_fraction_KLd_below_8=float(power[scaled < 8].sum()/power.sum()))


def configuration(grid, params, days, frames, filtered):
    source = '\n'.join(inspect.getsource(fn) for fn in
                       (initial_height, odd_extension, wavenumbers))
    return dict(Nx=grid.Nx, Ny=grid.Ny, Lx=grid.Lx, Ly=grid.Ly,
                parameters=asdict(params), days=days, frames=frames, filtered=filtered,
                amplitude=.15, radius_over_Ld=.1, minimum_KLd=8. if filtered else None,
                initial_condition_fingerprint=hashlib.sha256(source.encode()).hexdigest(),
                solver_fingerprint=model_fingerprint())


def render(data, grid, ld, folder, stem, filtered):
    label = 'Initially short wavelengths only' if filtered else 'Tiny Gaussian eddy'
    render_eta_animation(data, grid, folder, stem, label=label,
                         radius=ld/10, radius_label='Ld/10')
    eta = data['eta']-data['eta'].mean(axis=(1, 2), keepdims=True)
    t = data['time']/DAY
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    im = axes[0].pcolormesh(grid.x_c/1e3, t, eta[:, grid.Ny//2, :],
                            cmap='RdBu_r', vmin=-.08, vmax=.08, shading='auto')
    axes[0].set(xlabel='x [km]', ylabel='Time [days]', title='Height at the central latitude')
    fig.colorbar(im, ax=axes[0], label='η [m]')
    # These are wavelengths of the closed-basin sine expansion at t=0.
    power = np.abs(np.fft.fft2(odd_extension(data['eta'][0])))**2
    kl = wavenumbers(grid)*ld
    edges = np.linspace(0, 50, 101)
    weights, _ = np.histogram(kl, edges, weights=power)
    axes[1].plot(.5*(edges[1:]+edges[:-1]), weights/power.sum())
    axes[1].axvline(1., color='0.5', linestyle=':', label='K Ld = 1')
    axes[1].axvline(2*np.pi, color='k', linestyle='--', label='wavelength = Ld')
    axes[1].axvline(8., color='tab:red', linestyle=':', label='short-wave cutoff')
    axes[1].set(xlabel='Total wavenumber K × Ld', ylabel='Fraction of initial squared height per bin',
                title='Initial height spectrum', xlim=(0, 50))
    axes[1].legend(fontsize=8)
    fig.savefig(folder/f'experiment_11_{stem}_diagnostics.png', dpi=150)
    plt.close(fig)


def run_small_scales(args):
    folder = Path(args.output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    nx, ny = (24, 18) if args.smoke else ((192, 144) if args.quick else (256, 192))
    grid = make_grid(nx, ny, 800e3, 600e3)
    params = ModelParams(H=100., f0=1e-4, beta=2e-11, y0=grid.Ly/2,
                         r=1/(120*DAY), linear=True, Ah=0.)
    ld = np.sqrt(params.g*params.H)/params.f0
    days = .05 if args.smoke else args.days
    for filtered, stem in ((False, 'tiny_eddy'), (True, 'short_waves')):
        if args.variant != 'both' and (args.variant == 'short-waves') != filtered:
            continue
        path = folder/f'experiment_11_{stem}.npz'
        meta_path = path.with_suffix('.json')
        config = configuration(grid, params, days, args.frames, filtered)
        if args.replot or (path.exists() and meta_path.exists()
                           and json.loads(meta_path.read_text())['configuration'] == config):
            if not meta_path.exists() or json.loads(meta_path.read_text())['configuration'] != config:
                raise ValueError('Saved configuration differs; replot with the original run settings.')
            with np.load(path, allow_pickle=False) as saved:
                data = {key: saved[key] for key in saved.files}
            print(f'11 {stem}: reuse saved run', flush=True)
        else:
            eta = initial_height(grid, ld, filtered=filtered)
            u, v = geostrophic_velocities_from_eta(eta, grid, params, alpha=1., sponge=0)
            enforce_bcs(u, v)
            print(f'11 {stem}: R={ld/10/1e3:.2f} km, dx={grid.dx/1e3:.2f} km; {days:g} days', flush=True)
            out = integrate(grid, params, zero_forcing, (eta, u, v), days=days, frames=args.frames)
            uc, vc = velocities(out)
            data = dict(eta=np.asarray(out['eta']), time=np.asarray(out['time']),
                        x=grid.x_c, y=grid.y_c, mean_ke=.5*np.mean(uc**2+vc**2, axis=(1, 2)))
            stats = spectral_diagnostics(eta, grid, ld)
            stats['maximum_mean_height_drift'] = float(np.max(np.abs(data['eta'].mean(axis=(1, 2))-eta.mean())))
            np.savez_compressed(path, **data)
            meta_path.write_text(json.dumps(dict(configuration=config, Ld_m=ld,
                radius_m=ld/10, diagnostics=stats,
                note='Cutoff applies to initial height in the sine expansion. Varying f, boundaries and SW adjustment can subsequently introduce other scales. The stationary spatial mean is removed for display.'), indent=2)+'\n')
            print(json.dumps(stats), flush=True)
        render(data, grid, ld, folder, stem, filtered)
