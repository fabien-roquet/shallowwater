from .params import ModelParams
from .grid import Grid, make_grid
from importlib.metadata import PackageNotFoundError, version
import platform

import numpy as np

from .forcing import zero_forcing, wind_gyre_forcing, tidal_potential_forcing
from .forcing import stommel_arons_forcing, storm_surge_forcing, coastal_alongshore_wind_forcing
from .forcing import (
    center_wind_to_staggered,
    make_wind_forcing,
    make_wind_forcing_from_file,
    uniform_wind_forcing,
)
from .initial import setup_initial_state, geostrophic_velocities_from_eta
from .dynamics import tendencies, enforce_bcs
from .runner import run_model
from .diagnostics import compute_dt_cfl
from .bathymetry import (
    center_depth,
    depth_on_u,
    depth_on_v,
    load_bathymetry,
    shelf_bathymetry,
    wave_speed,
)
from .sponge import make_sponge_hook, sponge_mask_eta
from .visualize import animate_eta, coast_hovmoller, plot_forcings, animate_eta_spectrum

try:
    __version__ = version("shallowwater")
except PackageNotFoundError:  # source tree without installed package metadata
    __version__ = "0.1.4"


def backend_info():
    """Return versions and the active numerical-operator backend."""
    from .operators import NUMBA_AVAILABLE, USE_NUMBA

    return {
        "shallowwater": __version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "numba_available": bool(NUMBA_AVAILABLE),
        "backend": "numba" if USE_NUMBA else "numpy",
    }


__all__ = [
    "ModelParams",
    "Grid",
    "make_grid",
    "zero_forcing",
    "wind_gyre_forcing",
    "uniform_wind_forcing",
    "center_wind_to_staggered",
    "make_wind_forcing",
    "make_wind_forcing_from_file",
    "tidal_potential_forcing",
    "stommel_arons_forcing",
    "storm_surge_forcing",
    "coastal_alongshore_wind_forcing",
    "setup_initial_state",
    "geostrophic_velocities_from_eta",
    "tendencies",
    "enforce_bcs",
    "run_model",
    "compute_dt_cfl",
    "center_depth",
    "depth_on_u",
    "depth_on_v",
    "load_bathymetry",
    "shelf_bathymetry",
    "wave_speed",
    "make_sponge_hook",
    "sponge_mask_eta",
    "animate_eta",
    "coast_hovmoller",
    "plot_forcings",
    "animate_eta_spectrum",
    "backend_info",
    "__version__",
]
