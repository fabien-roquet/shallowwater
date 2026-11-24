"""Operators for an x-periodic C-grid with impermeable, free-slip y walls.

The two endpoint columns of u represent the same face. Only the first is
independent; all operators return identical values at these two columns.
Interior operations reuse the selected NumPy/Numba backend.
"""
import numpy as np
from .operators import (
    avg_u_to_center, avg_v_to_center, avg_center_to_v,
    grad_y_on_v, u_on_v, divergence,
)
from . import operators as base


def avg_center_to_u(c):
    out = base.avg_center_to_u(c)
    out[:, 0] = out[:, -1] = 0.5 * (c[:, -1] + c[:, 0])
    return out


def grad_x_on_u(c, dx):
    out = base.grad_x_on_u(c, dx)
    out[:, 0] = out[:, -1] = (c[:, 0] - c[:, -1]) / dx
    return out


def v_on_u(v):
    out = base.v_on_u(v)
    out[:, 0] = out[:, -1] = 0.25 * (
        v[:-1, -1] + v[1:, -1] + v[:-1, 0] + v[1:, 0]
    )
    return out


def _laplacian(a, dx, dy):
    padded = np.pad(a, ((1, 1), (0, 0)), mode="edge")
    return ((np.roll(a, -1, axis=1) - 2*a + np.roll(a, 1, axis=1)) / dx**2
            + (padded[2:] - 2*a + padded[:-2]) / dy**2)


def laplacian_u(u, dx, dy):
    lap = _laplacian(u[:, :-1], dx, dy)
    return np.concatenate((lap, lap[:, :1]), axis=1)


def laplacian_v(v, dx, dy):
    return _laplacian(v, dx, dy)


def curl_on_center(u, v, dx, dy):
    """Corner curl averaged to centres, with periodic x and free-slip y."""
    dvdx = (v - np.roll(v, 1, axis=1)) / dx
    dudy = np.zeros((u.shape[0] + 1, u.shape[1] - 1))
    dudy[1:-1] = (u[1:, :-1] - u[:-1, :-1]) / dy
    corner = dvdx - dudy
    return 0.25 * (corner[:-1] + corner[1:]
                   + np.roll(corner[:-1], -1, axis=1)
                   + np.roll(corner[1:], -1, axis=1))
