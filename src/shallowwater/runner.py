import numpy as np
from .dynamics import tendencies, enforce_bcs

def run_model(tmax, dt, grid, params, forcing_fn, ic_fn, save_every=10, out_vars=('eta','u','v'), hooks=None, remove_mean_eta=False):
    eta, u, v = ic_fn(grid, params)
    state = {"eta": eta.copy(), "u": u.copy(), "v": v.copy()}
    enforce_bcs(state["u"], state["v"])

    def rhs(s, t):
        return tendencies(s, t, grid, params, forcing_fn, hooks=hooks)

    times = []
    out = {var: [] for var in out_vars}

    t = 0.0
    nsteps = int(np.ceil(tmax / dt))
    for n in range(nsteps + 1):
        if n % save_every == 0:
            times.append(t)
            for var in out_vars:
                out[var].append(state[var].copy())
        if n == nsteps:
            break

        d1 = rhs(state, t)
        y1 = {"eta": state["eta"] + dt * d1[0],
              "u": state["u"] + dt * d1[1],
              "v": state["v"] + dt * d1[2]}
        enforce_bcs(y1["u"], y1["v"])

        d2 = rhs(y1, t + dt)
        y2 = {k: 0.75 * state[k] + 0.25 * (y1[k] + dt * d2[i])
              for i, k in enumerate(state.keys())}
        enforce_bcs(y2["u"], y2["v"])

        d3 = rhs(y2, t + 0.5 * dt)
        state = {k: (1.0/3.0) * state[k] + (2.0/3.0) * (y2[k] + dt * d3[i])
                 for i, k in enumerate(state.keys())}
        enforce_bcs(state["u"], state["v"])

        if remove_mean_eta:
            state["eta"] -= state["eta"].mean()

        t += dt

    out["time"] = np.array(times)
    return out
