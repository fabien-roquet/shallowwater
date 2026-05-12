from pathlib import Path
import contextlib
import io
import sys

import matplotlib

matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
OUTPUT_DIR = Path(__file__).resolve().parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from shallowwater import (
    ModelParams,
    make_grid,
    zero_forcing,
    compute_dt_cfl,
    run_model,
    shelf_bathymetry,
    make_sponge_hook,
)

NX, NY = 220, 60
LX, LY = 4_200e3, 900e3
H_DEEP = 4000.0
H_COAST = 60.0
SHELF_WIDTH = 1_100e3


def tsunami_initial_condition(grid, params):
    X, Y = np.meshgrid(grid.x_c, grid.y_c)
    x0 = 650e3
    y0 = 0.5 * grid.Ly
    bredd_x = 170e3
    bredd_y = 0.42 * grid.Ly
    amp = 0.18

    eta = amp * np.exp(-((X - x0) ** 2) / (2 * bredd_x ** 2))
    eta *= np.exp(-((Y - y0) ** 2) / (2 * bredd_y ** 2))
    u = np.zeros((grid.Ny, grid.Nx + 1))
    v = np.zeros((grid.Ny + 1, grid.Nx))
    return eta, u, v


def run_tsunami_model():
    grid = make_grid(NX, NY, LX, LY)
    H = shelf_bathymetry(
        grid,
        H_deep=H_DEEP,
        H_coast=H_COAST,
        shelf_width=SHELF_WIDTH,
        coast="east",
        power=1.25,
    )
    params = ModelParams(H=H, f0=0.0, beta=0.0, r=0.0, linear=True)
    sponge = make_sponge_hook(width=450e3, tau=1200.0, sides=("west",), power=2.0)

    dt = compute_dt_cfl(grid, params, cfl=0.45)
    tmax = 7.0 * 3600.0
    save_every = max(1, int((8 * 60) / dt))

    technical_log = io.StringIO()
    with contextlib.redirect_stdout(technical_log):
        out = run_model(
            tmax,
            dt,
            grid,
            params,
            zero_forcing,
            tsunami_initial_condition,
            save_every=save_every,
            hooks=[sponge],
            show_progress=False,
        )

    return grid, H, out


def stack_eta(out):
    return np.stack(out["eta"], axis=0)


def choose_frames(out, max_count):
    count = len(out["eta"])
    return np.unique(np.linspace(0, count - 1, min(count, max_count), dtype=int))


def time_text(seconds):
    return f"{seconds / 3600:.1f} timmar"


def grid_labels(grid, H):
    x_km = grid.x_c / 1000
    y_km = grid.y_c / 1000
    X_km, Y_km = np.meshgrid(x_km, y_km)
    center_j = grid.Ny // 2
    shallow_start_km = (grid.Lx - SHELF_WIDTH) / 1000
    return x_km, y_km, X_km, Y_km, center_j, shallow_start_km


def animate_top_view(out, grid, H, max_frames=55, interval=130):
    x_km, y_km, X_km, Y_km, center_j, shallow_start_km = grid_labels(grid, H)
    eta = stack_eta(out)
    times = np.asarray(out["time"])
    frames = choose_frames(out, max_frames)
    eta_sel = eta[frames]
    vmax = float(np.nanpercentile(np.abs(eta_sel), 99))
    vmax = max(vmax, 1e-6)

    fig, ax = plt.subplots(figsize=(9, 3.4))
    im = ax.imshow(
        eta_sel[0],
        origin="lower",
        extent=[0, grid.Lx / 1000, 0, grid.Ly / 1000],
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )
    ax.contour(X_km, Y_km, H / 1000, levels=[0.1, 0.5, 1, 2, 3], colors="k", alpha=0.18, linewidths=0.7)
    ax.axvspan(shallow_start_km, grid.Lx / 1000, color="#facc15", alpha=0.12)
    ax.set_xlabel("avstånd österut [km]")
    ax.set_ylabel("avstånd nord-syd [km]")
    fig.colorbar(im, ax=ax, label="vattenytans höjd [m]")

    def update(k):
        im.set_data(eta_sel[k])
        ax.set_title(f"Vågen sedd uppifrån i 2D - tid {time_text(times[frames[k]])}")
        return (im,)

    update(0)
    return animation.FuncAnimation(fig, update, frames=len(frames), interval=interval, blit=False), fig


def animate_side_view(out, grid, H, max_frames=60, interval=130, wave_scale=8000):
    x_km, y_km, X_km, Y_km, center_j, shallow_start_km = grid_labels(grid, H)
    eta = stack_eta(out)
    times = np.asarray(out["time"])
    frames = choose_frames(out, max_frames)
    bottom = -H[center_j, :] / 1000
    floor = -4.35

    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.fill_between(x_km, bottom, floor, color="#7a6a58", alpha=0.9)
    ax.plot(x_km, bottom, color="#4a3528", linewidth=1.8)
    ax.axhline(0, color="0.35", linewidth=0.8, alpha=0.7)
    ax.axvspan(shallow_start_km, grid.Lx / 1000, color="#facc15", alpha=0.14)

    water_fill = None
    (water_line,) = ax.plot([], [], color="#0f5e9c", linewidth=2.4)

    ax.set_xlim(0, grid.Lx / 1000)
    ax.set_ylim(floor, 2.1)
    ax.set_xlabel("avstånd österut [km]")
    ax.set_ylabel("höjd i bilden [km]")
    ax.grid(True, alpha=0.25)
    ax.text(60, 1.65, f"Vågens höjd är förstorad {wave_scale} gånger", fontsize=10)

    def update(k):
        nonlocal water_fill
        if water_fill is not None:
            water_fill.remove()
        water = eta[frames[k], center_j, :] * wave_scale / 1000
        water_fill = ax.fill_between(x_km, bottom, water, color="#9bd4f0", alpha=0.42)
        water_line.set_data(x_km, water)
        ax.set_title(f"Vågen sedd från sidan i 2D - tid {time_text(times[frames[k]])}")
        return (water_line, water_fill)

    update(0)
    return animation.FuncAnimation(fig, update, frames=len(frames), interval=interval, blit=False), fig


def animate_3d(out, grid, H, max_frames=34, interval=170, wave_scale=8000, step_x=4, step_y=3):
    x_km, y_km, X_km, Y_km, center_j, shallow_start_km = grid_labels(grid, H)
    eta = stack_eta(out)
    times = np.asarray(out["time"])
    frames = choose_frames(out, max_frames)

    xs = slice(None, None, step_x)
    ys = slice(None, None, step_y)
    X3, Y3 = np.meshgrid(grid.x_c[xs] / 1000, grid.y_c[ys] / 1000)
    bottom = -H[ys, xs] / 1000
    eta_sel = eta[frames, ys, xs]
    vmax = float(np.nanpercentile(np.abs(eta_sel), 99))
    zmax = max(1.8, 1.25 * vmax * wave_scale / 1000)

    fig = plt.figure(figsize=(9, 5.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X3, Y3, bottom, color="#7a6a58", alpha=0.58, linewidth=0, antialiased=False)
    ax.set_xlim(0, grid.Lx / 1000)
    ax.set_ylim(0, grid.Ly / 1000)
    ax.set_zlim(-4.4, zmax)
    ax.set_xlabel("öst [km]")
    ax.set_ylabel("nord-syd [km]")
    ax.set_zlabel("höjd i bilden [km]")
    ax.view_init(elev=28, azim=-63)
    ax.set_box_aspect((4.2, 0.9, 0.8))
    ax.text2D(0.02, 0.95, f"Vågen är förstorad {wave_scale} gånger", transform=ax.transAxes)

    water_surface = None

    def update(k):
        nonlocal water_surface
        if water_surface is not None:
            water_surface.remove()
        water = eta_sel[k] * wave_scale / 1000
        water_surface = ax.plot_surface(
            X3,
            Y3,
            water,
            cmap="RdBu_r",
            vmin=-vmax * wave_scale / 1000,
            vmax=vmax * wave_scale / 1000,
            alpha=0.86,
            linewidth=0,
            antialiased=False,
        )
        ax.set_title(f"3D: vattenyta och sluttande botten - tid {time_text(times[frames[k]])}")
        return (water_surface,)

    update(0)
    return animation.FuncAnimation(fig, update, frames=len(frames), interval=interval, blit=False), fig


def save_gif(name, maker, grid, H, out, fps):
    path = OUTPUT_DIR / name
    print(f"Sparar {path}")
    anim, fig = maker(out, grid, H)
    anim.save(path, writer="pillow", fps=fps, dpi=130)
    plt.close(fig)


def main():
    print("Kör modellen för tsunami-labben...")
    grid, H, out = run_tsunami_model()
    save_gif("tsunami_shoaling_sv_2d_uppifran.gif", animate_top_view, grid, H, out, fps=8)
    save_gif("tsunami_shoaling_sv_2d_sidovy.gif", animate_side_view, grid, H, out, fps=8)
    save_gif("tsunami_shoaling_sv_3d_botten.gif", animate_3d, grid, H, out, fps=7)
    print("Klart.")


if __name__ == "__main__":
    main()
