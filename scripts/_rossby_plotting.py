"""Common display settings for every experiment-11 height animation."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import BoundaryNorm
import numpy as np

DAY = 86400.


def render_eta_animation(data, grid, folder, stem, *, label, radius, radius_label,
                         nonlinear=None, early_sampling=False):
    """Use the same fixed scales for linear and nonlinear height."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    datasets = [data] if nonlinear is None else [data, nonlinear]
    fields = [np.asarray(d['eta'])-np.asarray(d['eta']).mean(axis=(1, 2), keepdims=True)
              for d in datasets]
    levels = np.linspace(-.08, .08, 17)
    norm = BoundaryNorm(levels, plt.get_cmap('RdBu_r').N)
    fig, axes = plt.subplots(1, len(fields), figsize=(7 if nonlinear is None else 12, 5),
                             constrained_layout=True, squeeze=False)
    axes = axes.ravel()
    images = []
    for i, (ax, eta) in enumerate(zip(axes, fields)):
        images.append(ax.imshow(eta[0], origin='lower', extent=(0, grid.Lx/1e3, 0, grid.Ly/1e3),
                      cmap='RdBu_r', norm=norm, aspect='equal'))
        ax.set(xlabel='x [km]', ylabel='y [km]')
        if nonlinear is not None:
            ax.set_title(['Linear SW', 'Nonlinear SW'][i], fontsize=11)
    fig.colorbar(images[0], ax=list(axes), label='η [m]',
                 ticks=np.linspace(-.08, .08, 9), extend='both', shrink=.9)
    heading = fig.suptitle('', fontsize=11)
    X, Y = np.meshgrid(grid.x_c/1e3, grid.y_c/1e3)
    contours = [None]*len(fields)
    def update(i):
        for j, (ax, im, eta) in enumerate(zip(axes, images, fields)):
            im.set_data(eta[i])
            if contours[j] is not None:
                contours[j].remove()
            contours[j] = ax.contour(X, Y, eta[i], levels=levels,
                                     colors='k', linewidths=.45, alpha=.65)
        text = f'{label}\nSource R = {radius_label} = {radius/1e3:.1f} km; t = {data["time"][i]/DAY:.2f} days'
        if early_sampling:
            text += ('  |  Early adjustment: fine time sampling' if data['time'][i] <= 3*DAY
                     else '  |  Later evolution: coarse time sampling')
        heading.set_text(text)
        return tuple(images)+(heading,)
    anim = FuncAnimation(fig, update, frames=len(fields[0]), interval=100, blit=False)
    anim.save(str(folder/f'eta_11_{stem}.gif'), writer='pillow', fps=10, dpi=120)
    fig.savefig(folder/f'experiment_11_{stem}_last_frame.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
