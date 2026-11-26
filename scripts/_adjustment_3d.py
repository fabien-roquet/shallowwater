"""3D surface rendering of experiment 10, using its saved physical heights."""
import json
from dataclasses import asdict, replace
import hashlib
import inspect

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LightSource, TwoSlopeNorm
import numpy as np

from shallowwater import run_model, zero_forcing
from _experiment_tools import model_fingerprint


def run_3d_case(args, setup, initial, sponge_factory, *, f0=None):
    """Save every two minutes, sharing the rotating initial state in both cases."""
    grid, reference, _, period, bound, _ = setup(args)
    params = reference if f0 is None else replace(reference, f0=f0)
    state = initial(grid, reference)
    duration = min(args.hours*3600, .05*period) if args.smoke else args.hours*3600
    intervals = max(1, int(np.ceil(duration/120.)))
    stride = max(1, int(np.ceil((duration/intervals)/bound)))
    dt = duration/(intervals*stride)
    source = '\n'.join(inspect.getsource(fn) for fn in
                       (setup, initial, sponge_factory, run_3d_case))
    config = dict(parameters=asdict(params), Nx=grid.Nx, Ny=grid.Ny,
                  Lx=grid.Lx, Ly=grid.Ly, dt=dt, duration=duration, save_every=stride,
                  initial_state_sha256=hashlib.sha256(b''.join(a.tobytes() for a in state)).hexdigest(),
                  sponge_width_m=args.sponge_width_km*1000, sponge_tau_seconds=args.sponge_hours*3600,
                  solver_fingerprint=model_fingerprint(),
                  experiment_fingerprint=hashlib.sha256(source.encode()).hexdigest())
    stem = 'experiment_10_3d_adjustment' if params.f0 != 0 else 'experiment_10_3d_no_rotation'
    path = args.output_dir/(stem+'.npz')
    meta = path.with_suffix('.json')
    matching = path.exists() and meta.exists() and json.loads(meta.read_text())['configuration'] == config
    if args.replot and not matching:
        raise ValueError('No matching uniformly sampled 3D history; run without --replot first.')
    if matching:
        with np.load(path, allow_pickle=False) as saved:
            data = {key: saved[key] for key in ('eta', 'time')}
        print(f'10 3D f={params.f0:g}: reuse saved simulation', flush=True)
    else:
        print(f'10 3D f={params.f0:g}: {duration/3600:g} h, {intervals+1} frames, '
              f'uniform spacing {stride*dt/60:.2f} min', flush=True)
        sponge, _ = sponge_factory(grid, width=args.sponge_width_km*1000, tau=args.sponge_hours*3600)
        out = run_model(np.nextafter(duration, 0.), dt, grid, params, zero_forcing,
                        lambda g, p: state, save_every=stride, hooks=[sponge], out_vars=('eta',))
        data = {key: np.asarray(out[key]) for key in ('eta', 'time')}
        if not np.isfinite(data['eta']).all():
            raise RuntimeError('Non-finite 3D adjustment solution.')
        np.savez_compressed(path, **data, x=grid.x_c, y=grid.y_c)
        meta.write_text(json.dumps(dict(configuration=config), indent=2)+'\n')
    return grid, params, data


def render_3d(args, grid, params, data, nonrotating):
    # Keep all saved times; reduce only surface tessellation for GIF rendering.
    stride = max(1, grid.Nx//160)
    rows = np.flatnonzero((grid.y_c >= 1e6) & (grid.y_c <= 3e6))[::stride]
    cols = np.flatnonzero((grid.x_c >= 1e6) & (grid.x_c <= 3e6))[::stride]
    # A display mesh of at most 56 points per axis keeps the long GIF practical.
    rows = rows[np.unique(np.linspace(0, len(rows)-1, min(56, len(rows))).astype(int))]
    cols = cols[np.unique(np.linspace(0, len(cols)-1, min(56, len(cols))).astype(int))]
    X, Y = np.meshgrid(grid.x_c[cols]/1e3, grid.y_c[rows]/1e3)
    np.testing.assert_array_equal(data['time'], nonrotating['time'])
    heights = [d['eta'][:, rows][:, :, cols]*100 for d in (data, nonrotating)]
    low, high, floor = -20., 50., -25.
    norm = TwoSlopeNorm(vmin=low, vcenter=0., vmax=high)
    cmap = plt.get_cmap('RdBu_r')
    levels = np.r_[np.linspace(low, 0., 9), np.linspace(0., high, 21)[1:]]
    background, foreground = '#081827', '#dcebf5'
    fig = plt.figure(figsize=(15, 7.5), facecolor=background)
    axes = [fig.add_subplot(1, 2, i+1, projection='3d', facecolor=background) for i in range(2)]
    fig.subplots_adjust(left=0., right=.91, bottom=.08, top=.83, wspace=.03)
    for ax, title in zip(axes, ('With rotation: f = 10⁻⁴ s⁻¹', 'Without rotation: f = 0')):
        ax.set(xlim=(1000, 3000), ylim=(1000, 3000), zlim=(floor, 55),
               xlabel='x [km]', ylabel='y [km]', zlabel='Sea-level displacement [cm]')
        ax.set_box_aspect((1., 1., .48))
        ax.view_init(elev=28, azim=-55)
        ax.set_proj_type('ortho')
        ax.set_xticks([1000, 1500, 2000, 2500, 3000])
        ax.set_yticks([1500, 2000, 2500, 3000])
        ax.set_zticks([-20, -10, 0, 10, 20, 30, 40, 50])
        ax.tick_params(colors=foreground, labelsize=8, pad=1)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.label.set_color(foreground)
            axis.set_pane_color((.04, .10, .15, 1.))
            axis.line.set_color('#526879')
            axis._axinfo['grid'].update(color=(.45, .6, .7, .16), linewidth=.5)
        # A dashed perimeter marks zero displacement above the projected map.
        ax.plot([1000, 3000, 3000, 1000, 1000],
                [1000, 1000, 3000, 3000, 1000], [0]*5,
                color='#8ca8b9', linewidth=.65, linestyle='--', alpha=.7)
        ax.set_title(title, color=foreground, fontsize=12, pad=2)
    fig.text(.07, .94, 'ADJUSTMENT WITH AND WITHOUT ROTATION', color=foreground,
              fontsize=19, weight='bold')
    fig.text(.07, .905, f'The first {args.hours:g} hours · identical initial height and currents · uniform playback',
              color='#91b4c9', fontsize=11)
    clock = fig.text(.07, .06, '', color=foreground, fontsize=11)
    fig.text(.07, .025, 'Vertical relief exaggerated • height and colour scales fixed • contours projected below',
              color='#91b4c9', fontsize=9)
    color_ax = fig.add_axes([.94, .25, .012, .42])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=color_ax,
                      ticks=[-20, -10, 0, 10, 20, 30, 40, 50], extend='both')
    cb.set_label('η [cm]', color=foreground, labelpad=10)
    cb.ax.tick_params(colors=foreground, labelsize=9)
    cb.outline.set_edgecolor('#526879')
    light = LightSource(azdeg=305, altdeg=45)
    artists = []
    def update(i):
        for artist in artists:
            artist.remove()
        artists.clear()
        for ax, history in zip(axes, heights):
            height = history[i]
            colors = light.shade_rgb(cmap(norm(height))[..., :3], height,
                                     dx=float(np.mean(np.diff(X[0]))),
                                     dy=float(np.mean(np.diff(Y[:, 0]))),
                                     vert_exag=5., blend_mode='soft')
            artists.append(ax.plot_surface(X, Y, height, facecolors=colors,
                             rstride=1, cstride=1, linewidth=0, antialiased=False,
                             shade=False, rasterized=True))
            artists.append(ax.contourf(X, Y, height, levels=levels, zdir='z', offset=floor,
                                        cmap=cmap, norm=norm, extend='both', alpha=.9))
            if np.ptp(height) > 1e-10:
                artists.append(ax.contour(X, Y, height, levels=[-15, -10, -5, -2, 0, 2, 5, 10, 20, 30, 40],
                                           zdir='z', offset=floor+.02,
                                           colors='#a9cadb', linewidths=.35, alpha=.4))
        clock.set_text(f't = {data["time"][i]/3600:.2f} hours')
    update(0)
    fig.savefig(args.output_dir/'eta_10_3d_initial.png', dpi=160, facecolor=background)
    animation = FuncAnimation(fig, update, frames=len(data['time']), interval=100, blit=False)
    # Stream frames through ffmpeg rather than holding a long two-panel GIF in RAM.
    writer = FFMpegWriter(fps=10, codec='gif', extra_args=[
        '-filter_complex', 'split[a][b];[a]palettegen=stats_mode=single[p];[b][p]paletteuse=new=1',
        '-loop', '0'])
    animation.save(str(args.output_dir/'eta_10_3d.gif'), writer=writer, dpi=90,
                   progress_callback=lambda i, n: print(f'3D render: {i}/{n} frames', flush=True) if i % 100 == 0 else None)
    update(len(data['time'])-1)
    fig.savefig(args.output_dir/'eta_10_3d_final.png', dpi=160, facecolor=background)
    plt.close(fig)
    (args.output_dir/'experiment_10_3d.json').write_text(json.dumps(dict(
        sources=['experiment_10_3d_adjustment.npz', 'experiment_10_3d_no_rotation.npz'],
        panels=['rotating', 'nonrotating'], uniform_time_spacing=True, height_units='cm',
        x_limits_km=[1000, 3000], y_limits_km=[1000, 3000],
        spatial_stride=stride, camera_elevation_deg=28, camera_azimuth_deg=-55,
        vertical_relief_exaggerated=True, colour_limits_cm=[low, high],
        projected_contour_floor_cm=floor, fps=10, frame_count=len(data['time']),
        time_seconds=data['time'].tolist()), indent=2)+'\n')
