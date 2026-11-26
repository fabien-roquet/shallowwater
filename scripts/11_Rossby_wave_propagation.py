#!/usr/bin/env python3
"""Rossby-wave dispersion from balanced eddies above and below the deformation scale.

Run from the repository root or from the scripts/ directory. Generated movies
are written to animations/, which is intentionally ignored by git.
Each Gaussian movie compares linear SW (left) and full nonlinear SW (right).
Use --case tiny-cyclonic for the sign-reversed tiny eddy. Filtered waves are opt-in.
"""
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
ANIMATION_DIR = PROJECT_ROOT / "animations"
ANIMATION_DIR.mkdir(exist_ok=True)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib
matplotlib.use("Agg")


# Source notebook: 11_Rossby_wave_propagation.ipynb


# %% Cell 0
from shallowwater import (ModelParams, make_grid, setup_initial_state,
                          zero_forcing, run_model, compute_dt_cfl)
# If you exported it at top-level:
from shallowwater import geostrophic_velocities_from_eta
# Otherwise (as you mentioned) it lives in initial.py:
# from shallowwater.initial import geostrophic_velocities_from_eta

import numpy as np
from _rossby_plotting import render_eta_animation


# %% Cell 1
# Initial condition: nearly geostrophic Gaussian at mid-latitude, east of center
def ic_rossby_bump(g, p,
                   amp=0.15,                   # meters
                   R=None,                     # set to Ld by default
                   x0=None, y0=None,
                   degree_of_balance=1,     # 1 = fully geo, <1 leaves a touch of IG
                   alpha=0.95, sponge=6):
    if R  is None: R  = np.sqrt(p.g*p.H)/p.f0
    if x0 is None: x0 = 0.5*g.Lx             # start at the centre to watch westward propagation
    if y0 is None: y0 = p.y0                  # mid-lat line (f ≈ f0)

    # eta field
    eta0, _, _ = setup_initial_state(g, p, mode="gaussian_bump", amp=amp, R=R, x0=x0, y0=y0)
    eta0 -= eta0.mean()
    # geostrophic (u,v) from eta, scaled
    u0, v0 = geostrophic_velocities_from_eta(eta0, g, p,
                                             degree_of_balance=degree_of_balance,
                                             alpha=alpha, sponge=sponge)
    return eta0, u0, v0


def run_existing_cases(args=None):
    from _rossby_comparison import run_comparisons
    if args is None:
        args = make_parser().parse_args(['--case', 'existing'])
    return run_comparisons(args)


def replot_existing_cases(folder):
    args = make_parser().parse_args(['--case', 'existing', '--replot'])
    args.output_dir = Path(folder)
    return run_existing_cases(args)


def make_parser():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('all', 'existing', 'large', 'medium', 'small', 'tiny', 'tiny-cyclonic'), default='all')
    parser.add_argument('--variant', choices=('both', 'gaussian', 'short-waves'), default='gaussian',
                        help='Gaussian by default; the filtered diagnostic is opt-in.')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--days', type=float, default=60.)
    parser.add_argument('--frames', type=int, default=121)
    parser.add_argument('--output-dir', type=Path, default=ANIMATION_DIR)
    parser.add_argument('--replot', action='store_true')
    return parser


def run_experiment(args):
    if args.days <= 0 or args.frames < 2:
        raise ValueError('Use positive days and at least two frames.')
    results = None
    if args.variant != 'short-waves':
        from _rossby_comparison import run_comparisons
        results = run_comparisons(args)
    if args.variant in ('both', 'short-waves'):
        from copy import copy
        from _rossby_small_scales import run_small_scales
        filtered_args = copy(args)
        filtered_args.variant = 'short-waves'
        run_small_scales(filtered_args)
    return results


if __name__ == '__main__':
    run_experiment(make_parser().parse_args())
