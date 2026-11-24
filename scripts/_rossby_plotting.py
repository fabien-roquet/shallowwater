"""Common display settings for every experiment-11 height animation."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import BoundaryNorm
import numpy as np

DAY = 86400.


def render_eta_animation(data, grid, folder, stem, *, label, radius, radius_label):
    """Draw demeaned height with fixed limits, contours, and kilometre axes."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    eta = np.asarray(data['eta'])
    eta = eta-eta.mean(axis=(1, 2), keepdims=True)
    levels = np.linspace(-.08, .08, 17)
    contour_levels = levels
    norm = BoundaryNorm(levels, plt.get_cmap('RdBu_r').N)
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    im = ax.imshow(eta[0], origin='lower', extent=(0, grid.Lx/1e3, 0, grid.Ly/1e3),
                   cmap='RdBu_r', norm=norm, aspect='equal')
    ax.set(xlabel='x [km]', ylabel='y [km]')
    fig.colorbar(im, ax=ax, label='η [m]', ticks=np.linspace(-.08, .08, 9), extend='both')
    heading = fig.suptitle('', fontsize=11)
    X, Y = np.meshgrid(grid.x_c/1e3, grid.y_c/1e3)
    contours = None
    def update(i):
        nonlocal contours
        im.set_data(eta[i])
        if contours is not None:
            contours.remove()
        contours = ax.contour(X, Y, eta[i], levels=contour_levels,
                              colors='k', linewidths=.45, alpha=.65)
        heading.set_text(f'{label}\nSource R = {radius_label} = {radius/1e3:.1f} km; t = {data["time"][i]/DAY:.1f} days')
        return im, heading
    anim = FuncAnimation(fig, update, frames=len(eta), interval=100, blit=False)
    anim.save(str(folder/f'eta_11_{stem}.gif'), writer='pillow', fps=10, dpi=120)
    fig.savefig(folder/f'experiment_11_{stem}_last_frame.png', dpi=150)
    plt.close(fig)
