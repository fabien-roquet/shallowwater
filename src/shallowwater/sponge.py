"""Sponge-layer tendency hooks for open-boundary experiments."""

import numpy as np

from .operators import avg_center_to_u, avg_center_to_v


def sponge_mask_eta(grid, *, width, sides=("west",), power=2.0):
    """Return a smooth 0..1 sponge mask on eta-points."""
    X, Y = np.meshgrid(grid.x_c, grid.y_c)
    mask = np.zeros((grid.Ny, grid.Nx), dtype=float)
    width = float(width)
    power = float(power)
    for side in sides:
        side = side.lower()
        if side == "west":
            r = np.clip((width - X) / width, 0.0, 1.0)
        elif side == "east":
            r = np.clip((X - (grid.Lx - width)) / width, 0.0, 1.0)
        elif side == "south":
            r = np.clip((width - Y) / width, 0.0, 1.0)
        elif side == "north":
            r = np.clip((Y - (grid.Ly - width)) / width, 0.0, 1.0)
        else:
            raise ValueError("sides may contain west, east, south, or north")
        mask = np.maximum(mask, r**power)
    return mask


def make_sponge_hook(*, width, tau=600.0, sides=("west",), power=2.0, damp_eta=True, damp_velocity=True):
    """Create a hook that damps eta, u and v inside a boundary sponge layer."""
    if tau <= 0.0:
        raise ValueError("tau must be positive.")

    def hook(state, t, grid, params):
        mask = sponge_mask_eta(grid, width=width, sides=sides, power=power)
        sigma_eta = mask / float(tau)
        sigma_u = avg_center_to_u(sigma_eta)
        sigma_v = avg_center_to_v(sigma_eta)
        deta = -sigma_eta * state["eta"] if damp_eta else None
        du = -sigma_u * state["u"] if damp_velocity else None
        dv = -sigma_v * state["v"] if damp_velocity else None
        return deta, du, dv

    return hook
