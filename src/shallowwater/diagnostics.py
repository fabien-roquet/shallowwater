import numpy as np
def compute_dt_cfl(grid, params, umax=0.0, vmax=0.0, cfl=0.5):
    c = (params.g * params.H) ** 0.5
    dt_x = grid.dx / (c + max(1e-12, abs(umax)))
    dt_y = grid.dy / (c + max(1e-12, abs(vmax)))
    return cfl * min(dt_x, dt_y)
