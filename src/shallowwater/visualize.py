# src/shallowwater/visualize.py
"""
Visualization utilities for the shallowwater package.

Provided helpers:
- animate_eta(out, grid, ...): 2D animation of eta(x,y,t).
- coast_hovmoller(out, grid, ...): Hovmöller of eta along the domain perimeter vs time,
  with vertical lines at the four corners and clear W/N/E/S labels.
- plot_forcings(forcing_fn, t, grid, params, ...): static 3-panel figure of (tau_x, tau_y, Q) at time t.
- animate_eta_spectrum(out, grid, ...): 2D animation of the power spectrum |FFT2(eta)|^2 vs time.
"""
from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Optional, Tuple, Union, Sequence

__all__ = ["animate_eta", "coast_hovmoller", "plot_forcings", "animate_eta_spectrum"]

# ---- Shared helpers ----
def _format_time(seconds: float) -> str:
    if seconds < 60:      return f"{seconds:.1f} s"
    minutes = seconds / 60.0
    if minutes < 60:      return f"{minutes:.1f} min"
    hours = minutes / 60.0
    if hours < 48:        return f"{hours:.1f} h"
    days = hours / 24.0
    return f"{days:.2f} d"

class _EtaAnimation:
    def __init__(self, anim: animation.FuncAnimation, fig: plt.Figure):
        self._anim = anim; self._fig = fig
    @property
    def animation(self) -> animation.FuncAnimation: return self._anim
    @property
    def figure(self) -> plt.Figure: return self._fig
    def _repr_html_(self) -> str: return self._anim.to_jshtml()
    def to_jshtml(self) -> str: return self._anim.to_jshtml()
    def to_html5_video(self) -> str: return self._anim.to_html5_video()
    def save(self, path: str, fps: Optional[int] = None, writer: Optional[str] = None, dpi: int = 150):
        if writer is None:
            ext = path.lower().rsplit('.', 1)[-1]
            writer = "pillow" if ext == "gif" else "ffmpeg"
        if fps is None:
            fps = max(1, int(round(1000.0 / max(1, getattr(self._anim, "_interval", 100)))))
        self._anim.save(path, writer=writer, fps=fps, dpi=dpi)

def _stack_eta(out) -> np.ndarray:
    if "eta" not in out: raise KeyError("'out' must contain key 'eta'")
    eta_list = out["eta"]
    if isinstance(eta_list, np.ndarray) and eta_list.ndim == 3: return eta_list
    if isinstance(eta_list, np.ndarray) and eta_list.ndim == 2: return eta_list[None, ...]
    return np.stack(eta_list, axis=0)

def _times_array(out, T: int) -> np.ndarray:
    times = np.asarray(out.get("time", np.arange(T, dtype=float)))
    return times if times.shape[0] == T else np.arange(T, dtype=float)

def animate_eta(out: dict,
                grid=None,
                interval: int = 100,
                repeat: bool = True,
                vmin: Optional[float] = None,
                vmax: Optional[float] = None,
                cmap: Optional[str] = "RdBu_r",      # diverging red/blue by default
                figsize: Tuple[float, float] = (6, 5),
                title: str = "Free surface η",
                show_colorbar: bool = True,
                # NEW:
                remove_mean: bool = True,            # 1) remove spatial mean at each frame
                symmetric_limits: bool = True,       # center the colorbar around 0
                contours: bool = True,               # 3) draw contours
                contour_levels: Union[int, Sequence[float]] = 15,
                contour_colors: Union[str, Sequence[str]] = "k",
                contour_alpha: float = 0.8,
                contour_linewidths: float = 0.6,
                frames: Optional[Sequence[int]] = None  # 4) only plot these time indices
                ) -> _EtaAnimation:
    """
    Create a 2D animation of eta snapshots over time, with options for
    mean-removal, diverging colormap, contours, and frame subsetting.
    """
    # --- Stack eta and time vector ---
    eta_stack = _stack_eta(out)            # (T, Ny, Nx)
    T, Ny, Nx = eta_stack.shape
    times_full = _times_array(out, T)

    # --- Select subset of frames, if requested ---
    if frames is None:
        frame_idx = np.arange(T)
    else:
        frame_idx = np.array(frames, dtype=int)
        frame_idx = frame_idx[(frame_idx >= 0) & (frame_idx < T)]
        if frame_idx.size == 0:
            raise ValueError("`frames` produced an empty selection of indices.")
    times = times_full[frame_idx]

    # --- Remove spatial mean at each selected frame (optional) ---
    if remove_mean:
        eta_sel = eta_stack[frame_idx] - eta_stack[frame_idx].reshape(len(frame_idx), -1).mean(axis=1)[:, None, None]
    else:
        eta_sel = eta_stack[frame_idx]

    # --- Compute robust color limits; optionally enforce symmetry around 0 ---
    if vmin is None or vmax is None:
        vmax_robust = float(np.nanpercentile(np.abs(eta_sel), 98.0))
        if vmax_robust == 0.0:
            vmax_robust = 1e-6
        if symmetric_limits:
            vmin_calc, vmax_calc = -vmax_robust, vmax_robust
        else:
            p2 = float(np.nanpercentile(eta_sel, 2.0))
            p98 = float(np.nanpercentile(eta_sel, 98.0))
            vmin_calc, vmax_calc = p2, p98
        vmin = vmin if vmin is not None else vmin_calc
        vmax = vmax if vmax is not None else vmax_calc
        if vmin == vmax:
            vmin -= 1e-6; vmax += 1e-6

    # --- Figure & first frame ---
    fig, ax = plt.subplots(figsize=figsize)
    extent = None
    Xc = Yc = None
    if grid is not None:
        extent = (0.0, float(getattr(grid, "Lx", Nx)), 0.0, float(getattr(grid, "Ly", Ny)))
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        # For proper contour overlay in physical coords
        Xc, Yc = np.meshgrid(grid.x_c, grid.y_c)

    im = ax.imshow(eta_sel[0], origin="lower", extent=extent, vmin=vmin, vmax=vmax, cmap=cmap)
    if title:
        ax.set_title(f"{title} | t = {_format_time(float(times[0]))}")

    if show_colorbar:
        fig.colorbar(im, ax=ax, label="η [m]")

    # --- Contours (robust to Matplotlib variants without QuadContourSet.collections) ---
    contour_artists = []  # will store the actual Artist objects drawn for contours

    def _clear_contours():
        nonlocal contour_artists
        if contour_artists:
            for a in contour_artists:
                try:
                    a.remove()
                except Exception:
                    pass
            contour_artists = []

    def _draw_contours(field2d):
        nonlocal contour_artists
        _clear_contours()

        # levels handling
        if isinstance(contour_levels, int):
            lvls = np.linspace(vmin, vmax, contour_levels)
        else:
            lvls = contour_levels

        # Draw and capture artists safely
        if Xc is not None and Yc is not None:
            cs = ax.contour(Xc, Yc, field2d, levels=lvls,
                            colors=contour_colors, linewidths=contour_linewidths,
                            alpha=contour_alpha)
        else:
            cs = ax.contour(field2d, levels=lvls,
                            colors=contour_colors, linewidths=contour_linewidths,
                            alpha=contour_alpha)

        # Try standard attributes first
        grabbed = False
        if hasattr(cs, "collections"):
            contour_artists = list(cs.collections)
            grabbed = True
        # Some versions also/only expose .artists
        if not grabbed and hasattr(cs, "artists"):
            contour_artists = list(cs.artists)
            grabbed = True
        # Fallback: infer from Axes collections immediately after drawing
        if not grabbed and hasattr(ax, "collections"):
            n = len(lvls) if isinstance(lvls, (list, tuple, np.ndarray)) else 1
            contour_artists = list(ax.collections[-n:])

    if contours:
        _draw_contours(eta_sel[0])

    # --- Animator ---
    def update(i):
        im.set_data(eta_sel[i])
        if title:
            ax.set_title(f"{title} | t = {_format_time(float(times[i]))}")
        if contours:
            _draw_contours(eta_sel[i])
        return (im,)

    anim = animation.FuncAnimation(fig, update, frames=len(frame_idx),
                                   interval=interval, blit=False, repeat=repeat)
    return _EtaAnimation(anim, fig)

# ---- 2) η along the coastline (Hovmöller) ----
def _perimeter_indices(Nx: int, Ny: int):
    idx = []
    for j in range(Ny): idx.append((j, 0))                  # West
    for i in range(1, Nx): idx.append((Ny-1, i))            # North
    for j in range(Ny-2, -1, -1): idx.append((j, Nx-1))     # East
    for i in range(Nx-2, 0, -1): idx.append((0, i))         # South
    return idx

def _perimeter_segments_lengths(Nx: int, Ny: int):
    west_len = Ny; north_len = west_len + (Nx - 1)
    east_len = north_len + (Ny - 1); south_len = east_len + (Nx - 2)
    total = 2*(Nx+Ny)-4
    return west_len, north_len, east_len, south_len, total

def coast_hovmoller(out: dict, grid, units_x: str = "index", cmap: Optional[str] = "RdBu_r",
                    figsize: Tuple[float, float] = (9, 4), title: str = "η along the coastline vs time",
                    show_colorbar: bool = True) -> plt.Figure:
    eta_stack = _stack_eta(out); T, Ny, Nx = eta_stack.shape; times = _times_array(out, T)
    idx = _perimeter_indices(Nx, Ny); P = len(idx)
    ETA = np.empty((T, P)); 
    for t in range(T): ETA[t, :] = np.array([eta_stack[t, j, i] for (j, i) in idx])
    if units_x == "meters":
        dx, dy = float(grid.dx), float(grid.dy); s=[0.0]
        for _ in range(1, Ny): s.append(s[-1]+dy)
        for _ in range(1, Nx): s.append(s[-1]+dx)
        for _ in range(1, Ny): s.append(s[-1]+dy)
        for _ in range(1, Nx-1): s.append(s[-1]+dx)
        xvals = np.array(s); xlabel = "Coastline arclength [m] (W→N→E→S)"
    elif units_x == "km":
        dx, dy = float(grid.dx)/1000.0, float(grid.dy)/1000.0; s=[0.0]
        for _ in range(1, Ny): s.append(s[-1]+dy)
        for _ in range(1, Nx): s.append(s[-1]+dx)
        for _ in range(1, Ny): s.append(s[-1]+dy)
        for _ in range(1, Nx-1): s.append(s[-1]+dx)
        xvals = np.array(s); xlabel = "Coastline arclength [km] (W→N→E→S)"
    else:
        xvals = np.arange(P); xlabel = "Coastline index (W→N→E→S)"
    fig, ax = plt.subplots(figsize=figsize)
    extent = [xvals[0], xvals[-1] if P>1 else xvals[0]+1, float(times[0]), float(times[-1])]
    im = ax.imshow(ETA, aspect="auto", origin="lower", extent=extent, cmap=cmap)
    if title: ax.set_title(title)
    ax.set_xlabel(xlabel); ax.set_ylabel("time [s]")
    w_len, n_len, e_len, s_len, _ = _perimeter_segments_lengths(Nx, Ny)
    corner_indices = [w_len-1, n_len-1, e_len-1, s_len-1]
    for ci in corner_indices: ax.axvline(x=xvals[ci], color="k", lw=1, ls="--", alpha=0.7)
    seg_mids = [(w_len-1)/2, (w_len + (n_len-w_len)/2), (n_len + (e_len-n_len)/2), (e_len + (s_len-e_len)/2)]
    for m, name in zip([xvals[int(round(v))] for v in seg_mids], ["West","North","East","South"]):
        ax.text(m, ax.get_ylim()[1], name, ha="center", va="bottom", fontsize=9, fontweight="bold", transform=ax.transData)
    if show_colorbar: plt.colorbar(im, ax=ax, label="η [m]")
    return fig

# ---- 3) plot_forcings ----
def _to_center_from_u(Au: np.ndarray) -> np.ndarray: return 0.5*(Au[:, :-1] + Au[:, 1:])
def _to_center_from_v(Av: np.ndarray) -> np.ndarray: return 0.5*(Av[:-1, :] + Av[1:, :])

def plot_forcings(forcing_fn,
                  t: float,
                  grid,
                  params,
                  center_for_display: bool = True,
                  what = "all",
                  figsize: Tuple[float, float] = (12, 4),
                  cmap: Optional[str] = "RdBu_r",
                  title: Optional[str] = None) -> plt.Figure:
    """
    Plot surface forcings at time t.

    Parameters
    ----------
    forcing_fn : callable
        Signature: (t, grid, params) -> (taux_u(Ny,Nx+1), tauy_v(Ny+1,Nx), Q(Ny,Nx))
    t : float
        Time (s) passed to forcing_fn.
    grid, params : objects
        As used by the model.
    center_for_display : bool
        If True, average τx and τy to η-points for clean alignment.
    what : {"all","wind","taux","tauy","Q"} or Iterable[str]
        - "all":  show τx, τy, Q (3 panels)
        - "wind": show τx, τy, |τ| (3 panels)
        - "taux"/"tauy"/"Q": single panel
        - Or a list/tuple subset of {"taux","tauy","|tau|","Q"} in display order.
    figsize : (float, float)
        Figure size in inches.
    cmap : str or None
        Colormap for imshow.
    title : str or None
        Suptitle.

    Returns
    -------
    matplotlib.figure.Figure
    """
    taux_u, tauy_v, Q_c = forcing_fn(t, grid, params)

    # Bring stresses to η points for display (also used to compute |τ|)
    if center_for_display:
        taux = _to_center_from_u(taux_u)
        tauy = _to_center_from_v(tauy_v)
        Q = Q_c
    else:
        # For consistent shapes/magnitude, still compute centered fields
        taux = _to_center_from_u(taux_u)
        tauy = _to_center_from_v(tauy_v)
        Q = Q_c

    # Decide which fields to plot
    if isinstance(what, str):
        key = what.lower()
        if key == "all":
            fields = [("τx [N m⁻²]", taux),
                      ("τy [N m⁻²]", tauy),
                      ("Q [m s⁻¹]",   Q)]
        elif key == "wind":
            tau_mag = np.hypot(taux, tauy)
            fields = [("τx [N m⁻²]", taux),
                      ("τy [N m⁻²]", tauy),
                      ("|τ| [N m⁻²]", tau_mag)]
        elif key in ("taux", "tau_x", "tx"):
            fields = [("τx [N m⁻²]", taux)]
        elif key in ("tauy", "tau_y", "ty"):
            fields = [("τy [N m⁻²]", tauy)]
        elif key == "q":
            fields = [("Q [m s⁻¹]", Q)]
        elif key == "Q":   # accept uppercase
            fields = [("Q [m s⁻¹]", Q)]
        else:
            raise ValueError(f"Unknown 'what' option: {what}")
    else:
        # Iterable: custom selection/order
        name_map = {
            "taux": ("τx [N m⁻²]", taux),
            "tauy": ("τy [N m⁻²]", tauy),
            "|tau|": ("|τ| [N m⁻²]", np.hypot(taux, tauy)),
            "q": ("Q [m s⁻¹]", Q),
            "Q": ("Q [m s⁻¹]", Q),
        }
        fields = []
        for k in what:
            k_norm = str(k).strip().lower()
            if k_norm in ("taux", "tx", "tau_x"):
                fields.append(name_map["taux"])
            elif k_norm in ("tauy", "ty", "tau_y"):
                fields.append(name_map["tauy"])
            elif k_norm in ("|tau|", "taumag", "mag"):
                fields.append(name_map["|tau|"])
            elif k_norm in ("q",):
                fields.append(name_map["q"])
            elif k == "Q":
                fields.append(name_map["Q"])
            else:
                raise ValueError(f"Unknown entry in 'what': {k}")

    n = len(fields)
    if n == 0:
        raise ValueError("No fields selected to plot.")

    # Ensure reasonable width for multiple panels
    if figsize is None:
        figsize = (4 * n, 4)
    elif figsize[0] < 4 * n:
        figsize = (max(figsize[0], 4 * n), figsize[1])

    fig, axes = plt.subplots(1, n, figsize=figsize, constrained_layout=True, squeeze=False)
    axes = axes[0]

    # Domain extent (assumes centered fields)
    Ny, Nx = fields[0][1].shape
    extent = (0.0, float(getattr(grid, "Lx", Nx)), 0.0, float(getattr(grid, "Ly", Ny)))

    for ax, (label, arr) in zip(axes, fields):
        im = ax.imshow(arr, origin="lower", extent=extent, cmap=cmap)
        ax.set_title(label)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        fig.colorbar(im, ax=ax)

    if title:
        fig.suptitle(title, fontsize=12)

    return fig

# ---- 4) animate_eta_spectrum ----
def _fft2_power(field2d: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    F = np.fft.fft2(field2d); P = (F*F.conj()).real; return np.fft.fftshift(P) + eps
def _wavenumbers_cycles_per_km(N: int, d: float) -> np.ndarray: return np.fft.fftshift(np.fft.fftfreq(N, d=d)) * 1e3

def animate_eta_spectrum(out: dict,
                         grid,
                         interval: int = 100,
                         repeat: bool = True,
                         log10: bool = True,
                         vmin: Optional[float] = None,
                         vmax: Optional[float] = None,
                         cmap: Optional[str] = "RdBu_r",
                         figsize: Tuple[float, float] = (6, 5),
                         title: str = "Power spectrum |η̂|²",
                         show_colorbar: bool = True,
                         quadrant: str = "full") -> _EtaAnimation:
    """
    2D animation of the power spectrum of eta over time.

    Parameters
    ----------
    quadrant : {"full","ur"}
        "full" -> show fftshifted full spectrum centered at (0,0)
        "ur"   -> show only the upper-right quadrant (kx>=0, ky>=0) without shift
    """
    eta_stack = _stack_eta(out)
    T, Ny, Nx = eta_stack.shape
    times = _times_array(out, T)

    def _power_full(A):
        # full, centered spectrum
        F = np.fft.fft2(A)
        P = (F * F.conj()).real
        return np.fft.fftshift(P)

    def _power_ur(A):
        # upper-right quadrant (no shift): rows -> ky, cols -> kx
        F = np.fft.fft2(A)
        P = (F * F.conj()).real
        return P[0:Ny//2+1, 0:Nx//2+1]

    if quadrant == "ur":
        # non-negative wavenumbers only, in cycles / km
        kx = np.fft.fftfreq(Nx, d=float(getattr(grid, "dx", 1.0))) * 1e3
        ky = np.fft.fftfreq(Ny, d=float(getattr(grid, "dy", 1.0))) * 1e3
        kx = kx[0:Nx//2+1]
        ky = ky[0:Ny//2+1]
        extent = (kx[0], kx[-1], ky[0], ky[-1])   # (0..kx_max, 0..ky_max)
        power_fn = _power_ur
        xlab, ylab = "k_x [cycles km⁻¹]", "k_y [cycles km⁻¹]"
    else:
        # full, fftshifted axes centered at zero
        kx = np.fft.fftshift(np.fft.fftfreq(Nx, d=float(getattr(grid, "dx", 1.0)))) * 1e3
        ky = np.fft.fftshift(np.fft.fftfreq(Ny, d=float(getattr(grid, "dy", 1.0)))) * 1e3
        extent = (kx[0], kx[-1], ky[0], ky[-1])
        power_fn = _power_full
        xlab, ylab = "k_x [cycles km⁻¹]", "k_y [cycles km⁻¹]"

    # Determine color scale from a sample of frames using the chosen power/slicing
    if vmin is None or vmax is None:
        sample_idx = np.linspace(0, T-1, num=min(T, 40), dtype=int)
        vals = []
        for i in sample_idx:
            P = power_fn(eta_stack[i])
            A = np.log10(P + 1e-30) if log10 else P
            vals.append(A)
        V = np.concatenate([a.ravel() for a in vals]) if vals else np.array([0.0])
        p2, p98 = np.nanpercentile(V, [2.0, 98.0])
        vmin = p2 if vmin is None else vmin
        vmax = p98 if vmax is None else vmax
        if vmin == vmax:
            vmin -= 1e-12
            vmax += 1e-12

    # First frame
    P0 = power_fn(eta_stack[0])
    A0 = np.log10(P0 + 1e-30) if log10 else P0

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(A0, origin="lower", extent=extent, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ttl = f"{title} | t = {_format_time(float(times[0]))}"
    if log10:
        ttl += " | log₁₀"
    if quadrant == "ur":
        ttl += " | kx≥0, ky≥0"
    ax.set_title(ttl)
    if show_colorbar:
        fig.colorbar(im, ax=ax, label="power" + (" (log₁₀)" if log10 else ""))

    def update(i):
        P = power_fn(eta_stack[i])
        A = np.log10(P + 1e-30) if log10 else P
        im.set_data(A)
        ttl = f"{title} | t = {_format_time(float(times[i]))}"
        if log10:
            ttl += " | log₁₀"
        if quadrant == "ur":
            ttl += " | kx≥0, ky≥0"
        ax.set_title(ttl)
        return (im,)

    anim = animation.FuncAnimation(fig, update, frames=T, interval=interval, blit=False, repeat=repeat)
    return _EtaAnimation(anim, fig)


