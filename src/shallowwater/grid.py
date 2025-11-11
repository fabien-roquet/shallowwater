from dataclasses import dataclass
import numpy as np

@dataclass
class Grid:
    Nx: int
    Ny: int
    Lx: float
    Ly: float
    dx: float
    dy: float
    x_c: np.ndarray
    y_c: np.ndarray
    x_u: np.ndarray
    y_u: np.ndarray
    x_v: np.ndarray
    y_v: np.ndarray

def make_grid(Nx: int, Ny: int, Lx: float, Ly: float) -> Grid:
    dx = Lx / Nx
    dy = Ly / Ny
    x_c = (np.arange(Nx) + 0.5) * dx
    y_c = (np.arange(Ny) + 0.5) * dy
    x_u = np.arange(Nx + 1) * dx
    y_u = y_c.copy()
    x_v = x_c.copy()
    y_v = np.arange(Ny + 1) * dy
    return Grid(Nx, Ny, Lx, Ly, dx, dy, x_c, y_c, x_u, y_u, x_v, y_v)
