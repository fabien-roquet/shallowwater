"""Shared integration, perturbation and plotting helpers for experiments 13–14."""
from pathlib import Path
import argparse
import json
import hashlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from shallowwater import compute_dt_cfl, run_model, setup_initial_state
from shallowwater import geostrophic_velocities_from_eta, operators
from shallowwater.dynamics import enforce_bcs
from shallowwater.channel import curl_on_center

DAY = 86400.0


def model_fingerprint():
    """Invalidate numerical checkpoints when the solver source changes."""
    digest = hashlib.sha256()
    for name in ("dynamics", "operators", "operators_numba", "channel", "runner",
                 "bathymetry", "forcing", "grid", "params", "initial"):
        path = ROOT / "src" / "shallowwater" / (name + ".py")
        digest.update(name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def arguments(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--quick", action="store_true", help="Lower resolution demonstration; use defaults for analysis.")
    parser.add_argument("--smoke", action="store_true", help="Very short integration to check the complete workflow.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "animations")
    parser.add_argument("--format", choices=("gif", "mp4"), default="gif")
    def frame_count(value):
        count = int(value)
        if count < 2:
            raise argparse.ArgumentTypeError("at least two frames are required")
        return count
    parser.add_argument("--frames", type=frame_count, default=100)
    return parser


def integrate(grid, params, forcing, initial=None, *, days, frames, hooks=None, diffusivity_bound=None):
    if initial is None:
        initial = setup_initial_state(grid, params)
    speed = max(float(np.max(np.abs(initial[1]))), float(np.max(np.abs(initial[2]))), 2.0)
    dt = compute_dt_cfl(grid, params, umax=speed, cfl=0.4)
    viscosity = max(params.Ah, diffusivity_bound or 0.)
    if viscosity:
        dt = min(dt, 0.2 / (viscosity * (grid.dx**-2 + grid.dy**-2)))
    # Land exactly on the requested final time; continuations then share a time axis.
    steps = max(1, int(np.ceil(days * DAY / dt)))
    dt = days * DAY / steps
    # Avoid a floating-point quotient just above an integer adding an extra step.
    stride = max(1, int(np.ceil(steps / max(1, frames - 1))))
    out = run_model(np.nextafter(days * DAY, 0.0), dt, grid, params, forcing,
                    lambda g, p: tuple(a.copy() for a in initial),
                    save_every=stride, hooks=hooks)
    for key in ("eta", "u", "v"):
        values = np.asarray(out[key])
        if not np.isfinite(values).all():
            raise RuntimeError(f"Non-finite {key}: reduce dt or forcing, or increase resolution/viscosity.")
    if np.min(params.H + np.asarray(out["eta"])) <= 0:
        raise RuntimeError("Non-positive layer thickness.")
    return out


def last_state(out):
    return tuple(out[key][-1].copy() for key in ("eta", "u", "v"))


def balanced_patch(grid, params, initial, *, amplitude, radius, seed_mode=None):
    X, Y = np.meshgrid(grid.x_c, grid.y_c)
    if seed_mode is None:
        eta = amplitude * np.exp(-((X - grid.Lx/2)**2 + (Y - grid.Ly/2)**2) / radius**2)
    else:
        # Periodic disturbance localized to the jet, seeded with reproducible modes.
        eta = amplitude * np.exp(-((Y-grid.Ly/2)/radius)**2) * (
            np.cos(2*np.pi*seed_mode*X/grid.Lx)
            + 0.5*np.cos(2*np.pi*(seed_mode+1)*X/grid.Lx + 0.7))
    eta -= eta.mean()
    u, v = geostrophic_velocities_from_eta(eta, grid, params, alpha=1.0)
    state = tuple(a+b for a, b in zip(initial, (eta, u, v)))
    enforce_bcs(state[1], state[2], grid.periodic_x)
    return state


def velocities(out):
    u, v = np.asarray(out["u"]), np.asarray(out["v"])
    return 0.5*(u[:, :, :-1]+u[:, :, 1:]), 0.5*(v[:, :-1]+v[:, 1:])


def eddy_energy(out):
    u, v = velocities(out)
    return 0.5*np.mean((u-u.mean(axis=2, keepdims=True))**2
                       + (v-v.mean(axis=2, keepdims=True))**2, axis=(1, 2))


def difference(a, b):
    return {"time": np.asarray(a["time"]), **{
        key: np.asarray(a[key])-np.asarray(b[key]) for key in ("eta", "u", "v")}}


def diagnostics(out, grid, params):
    u, v = velocities(out)
    eta = np.asarray(out["eta"])
    curl = curl_on_center if grid.periodic_x else operators.curl_on_center
    zeta = np.asarray([curl(uu, vv, grid.dx, grid.dy)
                       for uu, vv in zip(out["u"], out["v"])])
    h = params.H + eta
    f = params.f0 + params.beta*(grid.y_c[:, None]-params.y0)
    q = (zeta+f)/h
    return dict(time_days=np.asarray(out["time"])/DAY,
                mean_u=u.mean(axis=2), mean_q=q.mean(axis=2),
                mean_q_gradient=np.gradient(q.mean(axis=2), grid.dy, axis=1),
                eddy_ke=eddy_energy(out),
                mean_ke=0.5*np.mean(u*u+v*v, axis=(1, 2)),
                mean_eta=eta.mean(axis=(1, 2)),
                max_speed=np.max(np.hypot(u, v), axis=(1, 2)),
                min_thickness=np.min(h, axis=(1, 2)), zeta=zeta)


def channel_streamfunction(zeta, mean_u, grid):
    """Rotational velocity streamfunction, u_psi=-psi_y and v_psi=psi_x.

    Solve laplacian(psi)=zeta on cell centres with periodic x and constant
    psi on each wall. Their difference retains the instantaneous zonal velocity
    integral: psi_north-psi_south=-integral(mean_u dy).
    This is a nondivergent velocity diagnostic, not a geostrophic proxy g*eta/f.
    A single spatial mean is removed from each frame (the arbitrary gauge).
    """
    if not grid.periodic_x:
        raise ValueError("The streamfunction solver requires an x-periodic channel.")
    zeta = np.asarray(zeta)
    mean_u = np.asarray(mean_u)
    single = zeta.ndim == 2
    if single:
        zeta, mean_u = zeta[None], mean_u[None]
    if zeta.shape[1:] != (grid.Ny, grid.Nx) or mean_u.shape != zeta.shape[:2]:
        raise ValueError("Expected zeta (time, Ny, Nx) and mean_u (time, Ny).")
    # Fourier transform in x; solve all modes/times together by Thomas elimination.
    rhs = np.fft.rfft(zeta, axis=2)
    north = -grid.dy*np.sum(mean_u, axis=1)
    rhs[:, -1, 0] -= 2*north*grid.Nx/grid.dy**2
    eigen_x = -4*np.sin(np.pi*np.arange(grid.Nx//2+1)/grid.Nx)**2/grid.dx**2
    diagonal = np.broadcast_to(eigen_x-2/grid.dy**2, rhs.shape[1:]).copy()
    diagonal[[0, -1]] -= 1/grid.dy**2
    offdiag = 1/grid.dy**2
    for j in range(1, grid.Ny):
        multiplier = offdiag/diagonal[j-1]
        diagonal[j] -= multiplier*offdiag
        rhs[:, j] -= multiplier*rhs[:, j-1]
    rhs[:, -1] /= diagonal[-1]
    for j in range(grid.Ny-2, -1, -1):
        rhs[:, j] = (rhs[:, j]-offdiag*rhs[:, j+1])/diagonal[j]
    psi = np.fft.irfft(rhs, n=grid.Nx, axis=2)
    psi -= psi.mean(axis=(1, 2), keepdims=True)
    return psi[0] if single else psi


def save_diagnostics(path, data, metadata):
    np.savez_compressed(path.with_suffix(".npz"), **data)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


def animate_fields(path, grid, times, fields, titles, *, labels=None, symmetric=True,
                   time_label="Time since perturbation / wind change", scale_groups=None,
                   ncols=None, contour_panels=(), contour_fields=None, subtitle=None):
    """Fixed color scale for each panel over all frames."""
    count = len(fields)
    ncols = count if ncols is None else ncols
    nrows = int(np.ceil(count/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), constrained_layout=True,
                             squeeze=False)
    axes = axes.ravel()
    for ax in axes[count:]:
        ax.set_visible(False)
    images = []
    if scale_groups is None:
        scale_groups = list(range(count))
    limits = {}
    for field, group in zip(fields, scale_groups):
        limits[group] = max(limits.get(group, 1e-12), float(np.nanpercentile(np.abs(field), 99)))
    for i, (ax, field, title) in enumerate(zip(axes, fields, titles)):
        limit = limits[scale_groups[i]]
        im = ax.imshow(field[0], origin="lower", extent=(0, grid.Lx/1e3, 0, grid.Ly/1e3),
                       vmin=-limit if symmetric else 0, vmax=limit, cmap="RdBu_r", aspect="auto")
        ax.set(xlabel="x [km]", ylabel="y [km]", title=title)
        fig.colorbar(im, ax=ax, label=labels[i] if labels else "surface displacement [m]")
        images.append(im)
    title_artist = fig.suptitle("")
    contour_sets = {}
    X, Y = np.meshgrid(grid.x_c/1e3, grid.y_c/1e3)
    contour_levels = {i: np.linspace(-limits[scale_groups[i]], limits[scale_groups[i]], 15)
                      for i in contour_panels}
    contour_fields = {} if contour_fields is None else contour_fields
    if contour_fields:
        control_limit = max(1e-12, *(float(np.nanpercentile(np.abs(a), 99))
                                     for a in contour_fields.values()))
        for panel in contour_fields:
            contour_levels[panel] = np.linspace(-control_limit, control_limit, 15)
    all_contour_panels = sorted(set(contour_panels) | set(contour_fields))


    def update(i):
        for im, field in zip(images, fields):
            im.set_data(field[i])
        for panel in all_contour_panels:
            if panel in contour_sets:
                contour_sets[panel].remove()
            source = contour_fields.get(panel, fields[panel])
            field = source[i] if np.ndim(source) == 3 else source
            contour_sets[panel] = axes[panel].contour(
                X, Y, field, levels=contour_levels[panel],
                colors="k", linewidths=0.4, alpha=0.55)
        heading = f"{time_label}: {times[i]/DAY:.1f} days"
        if subtitle:
            heading += "\n" + (subtitle(i) if callable(subtitle) else subtitle)
        title_artist.set_text(heading)
        return images

    anim = FuncAnimation(fig, update, frames=len(times), interval=100, blit=False)
    anim.save(str(path), writer="pillow" if path.suffix == ".gif" else "ffmpeg", fps=10, dpi=100)
    plt.close(fig)
