"""Bathymetry helpers for scalar or variable-depth shallow-water runs."""

from pathlib import Path

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


def load_bathymetry(path, grid, *, key="H", validate_coordinates=True):
    """Load a cell-centred bathymetry map from ``.npy``, ``.npz`` or ``.csv``.

    The returned array has shape ``(grid.Ny, grid.Nx)`` and contains positive
    water depth in metres.  Its first index runs south-to-north and its second
    index west-to-east.  This helper deliberately does not interpolate,
    reproject, or represent dry land.

    An ``.npz`` archive must contain ``key`` (``"H"`` by default).  Optional
    one-dimensional ``x`` and ``y`` arrays, when present, are checked against
    the grid's cell-centre coordinates unless ``validate_coordinates=False``.
    """
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    try:
        if suffix == ".npy":
            values = np.load(file_path, allow_pickle=False)
        elif suffix == ".npz":
            with np.load(file_path, allow_pickle=False) as archive:
                if key not in archive:
                    raise ValueError(
                        f"Bathymetry archive {file_path} does not contain key {key!r}."
                    )
                values = np.asarray(archive[key], dtype=float)
                if validate_coordinates:
                    if "x" in archive:
                        x = np.asarray(archive["x"], dtype=float)
                        if x.shape != grid.x_c.shape or not np.allclose(x, grid.x_c):
                            raise ValueError("Bathymetry x coordinates do not match the grid.")
                    if "y" in archive:
                        y = np.asarray(archive["y"], dtype=float)
                        if y.shape != grid.y_c.shape or not np.allclose(y, grid.y_c):
                            raise ValueError("Bathymetry y coordinates do not match the grid.")
        elif suffix in {".csv", ".txt"}:
            delimiter = "," if suffix == ".csv" else None
            values = np.loadtxt(file_path, delimiter=delimiter)
        else:
            raise ValueError(
                f"Unsupported bathymetry format {suffix!r}; use .npy, .npz, .csv, or .txt."
            )
    except OSError as exc:
        raise ValueError(f"Could not read bathymetry file {file_path}: {exc}") from exc

    values = np.asarray(values, dtype=float)
    expected = (grid.Ny, grid.Nx)
    if values.shape != expected:
        raise ValueError(
            f"Bathymetry map must have cell-centre shape {expected}; got {values.shape}."
        )
    return center_depth(grid, values)
