"""Bathymetry helpers for scalar or variable-depth shallow-water runs."""

import numpy as np

from .operators import avg_center_to_u, avg_center_to_v


def center_depth(grid, H):
    """Return H on eta-points with shape (Ny, Nx)."""
    arr = np.asarray(H, dtype=float)
    if arr.ndim == 0:
        out = np.full((grid.Ny, grid.Nx), float(arr))
    elif arr.ndim == 1 and arr.shape[0] == grid.Nx:
        out = np.repeat(arr[None, :], grid.Ny, axis=0)
    elif arr.shape == (grid.Ny, grid.Nx):
        out = arr.copy()
    else:
        raise ValueError(
            "H must be scalar, length-Nx 1-D array, or shape (Ny, Nx); "
            f"got shape {arr.shape} for grid {(grid.Ny, grid.Nx)}"
        )
    if np.any(~np.isfinite(out)) or np.any(out <= 0.0):
        raise ValueError("All depths in H must be finite and positive.")
    return out


def depth_on_u(grid, H):
    """Return background depth averaged to u-points, shape (Ny, Nx+1)."""
    return avg_center_to_u(center_depth(grid, H))


def depth_on_v(grid, H):
    """Return background depth averaged to v-points, shape (Ny+1, Nx)."""
    return avg_center_to_v(center_depth(grid, H))


def shelf_bathymetry(grid, *, H_deep=4000.0, H_coast=80.0, shelf_width=None, coast="east", power=1.5):
    """Make a smooth one-dimensional continental shelf bathymetry."""
    if H_deep <= H_coast:
        raise ValueError("H_deep should be larger than H_coast.")
    if shelf_width is None:
        shelf_width = 0.45 * grid.Lx
    x = np.asarray(grid.x_c, dtype=float)
    if coast.lower() == "east":
        distance_from_coast = grid.Lx - x
    elif coast.lower() == "west":
        distance_from_coast = x
    else:
        raise ValueError("coast must be 'east' or 'west'.")
    r = np.clip(distance_from_coast / float(shelf_width), 0.0, 1.0)
    profile = H_coast + (H_deep - H_coast) * r**float(power)
    return np.repeat(profile[None, :], grid.Ny, axis=0)


def wave_speed(grid, params):
    """Return c=sqrt(gH) on eta-points."""
    return np.sqrt(params.g * center_depth(grid, params.H))
