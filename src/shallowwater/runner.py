import numpy as np
from typing import Sequence, Optional, Dict, Any
from .dynamics import tendencies

def run_model(
    tmax: float,
    dt: float,
    grid,
    params,
    forcing_fn,
    ic_fn,
    save_every: int = 10,
    out_vars: Sequence[str] = ("eta", "u", "v"),
    hooks=None,
    show_progress: bool = False,
    progress_kwargs: Optional[Dict[str, Any]] = None,
):
    """
    Integrate the shallow-water model from t=0 to tmax with time step dt.

    Parameters
    ----------
    tmax : float
        Final time [s].
    dt : float
        Time step [s].
    grid : Grid
        Grid object returned by make_grid.
    params : ModelParams
        Model parameters.
    forcing_fn : callable
        Forcing function f(t, grid, params) -> (taux_u, tauy_v, Q_eta[, phi_eta]).
    ic_fn : callable
        Initial condition function ic_fn(grid, params) -> (eta0, u0, v0).
    save_every : int, optional
        Save state every this many time steps.
    out_vars : sequence of str, optional
        Which variables to store in the output dict ("eta", "u", "v").
    hooks : list of callables, optional
        Optional list of hooks h(state, t, grid, params) -> (deta_dt, du_dt, dv_dt)
        applied additively to the tendencies.
    show_progress : bool, optional
        If True, display a progress bar over time steps (uses tqdm if available,
        otherwise falls back to simple text updates).
    progress_kwargs : dict, optional
        Extra keyword arguments passed to tqdm.tqdm (e.g., {"desc": "Integrating"}).

    Returns
    -------
    out : dict
        Dictionary with keys "time" and those in out_vars, each containing a list
        of snapshots over time.
    """
    # initial state
    eta0, u0, v0 = ic_fn(grid, params)
    if np.max(np.abs(eta0)) > 1e-6 or np.max(np.abs(u0)) > 1e-6 or np.max(np.abs(v0)) > 1e-6:
        print("WARNING: IC is not near rest! max(|eta|,|u|,|v|) =",
              np.max(np.abs(eta0)), np.max(np.abs(u0)), np.max(np.abs(v0)))
    state = {"eta": eta0.copy(), "u": u0.copy(), "v": v0.copy()}
    t = 0.0

    nsteps = int(np.ceil(tmax / dt))
    save_every = max(1, int(save_every))

    out: Dict[str, Any] = {"time": []}
    for name in out_vars:
        out[name] = []

    # helper to save current state
    def _save_state():
        out["time"].append(t)
        if "eta" in out_vars:
            out["eta"].append(state["eta"].copy())
        if "u" in out_vars:
            out["u"].append(state["u"].copy())
        if "v" in out_vars:
            out["v"].append(state["v"].copy())

    # save initial state
    _save_state()

    # progress bar setup
    iterator = range(1, nsteps + 1)
    if show_progress:
        try:
            from tqdm import tqdm
            kw = dict(total=nsteps, disable=False)
            if progress_kwargs is not None:
                kw.update(progress_kwargs)
            # tqdm over absolute step index
            iterator = tqdm(range(1, nsteps + 1), **kw)
        except Exception:
            # simple text progress fallback
            iterator = range(1, nsteps + 1)
            print(f"Running for {nsteps} steps (dt={dt:.3g}s, tmax={tmax:.3g}s)...")

    for n in iterator:
        # SSP-RK3 step
        # stage 1
        d_eta1, d_u1, d_v1 = tendencies(state, t, grid, params, forcing_fn, hooks=hooks)
        y1 = {
            "eta": state["eta"] + dt * d_eta1,
            "u":   state["u"]   + dt * d_u1,
            "v":   state["v"]   + dt * d_v1,
        }

        # stage 2
        d_eta2, d_u2, d_v2 = tendencies(y1, t + dt, grid, params, forcing_fn, hooks=hooks)
        y2 = {
            "eta": 0.75 * state["eta"] + 0.25 * (y1["eta"] + dt * d_eta2),
            "u":   0.75 * state["u"]   + 0.25 * (y1["u"]   + dt * d_u2),
            "v":   0.75 * state["v"]   + 0.25 * (y1["v"]   + dt * d_v2),
        }

        # stage 3
        d_eta3, d_u3, d_v3 = tendencies(y2, t + 0.5 * dt, grid, params, forcing_fn, hooks=hooks)
        state = {
            "eta": (1.0 / 3.0) * state["eta"] + (2.0 / 3.0) * (y2["eta"] + dt * d_eta3),
            "u":   (1.0 / 3.0) * state["u"]   + (2.0 / 3.0) * (y2["u"]   + dt * d_u3),
            "v":   (1.0 / 3.0) * state["v"]   + (2.0 / 3.0) * (y2["v"]   + dt * d_v3),
        }

        t = n * dt

        # save snapshots
        if (n % save_every) == 0 or n == nsteps:
            _save_state()

        # simple textual progress if tqdm not available but show_progress=True
        if show_progress:
            try:
                # if we are in tqdm mode, iterator is a tqdm object and handles updates
                pass
            except Exception:
                # fallback text every ~10% of total steps
                if nsteps >= 10 and n % max(1, nsteps // 10) == 0:
                    print(f"Step {n}/{nsteps}  (t = {t/86400:.2f} days)")

    return out
