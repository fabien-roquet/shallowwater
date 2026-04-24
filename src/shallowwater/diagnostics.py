import numpy as np

from .bathymetry import center_depth


def compute_dt_cfl(grid, params, umax=0.0, vmax=0.0, cfl=0.5):
    """Return a stable explicit time step from the fastest shallow-water speed."""
    H = center_depth(grid, params.H)
    Hmax = float(np.nanmax(H))
    c = (params.g * Hmax) ** 0.5
    speed = c + max(abs(umax), abs(vmax))
    return cfl * min(grid.dx, grid.dy) / speed
