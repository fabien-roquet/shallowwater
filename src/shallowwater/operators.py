# src/shallowwater/operators.py
import numpy as np
import os

try:
    from . import operators_numba  # or however you check availability
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

flag = os.getenv("SHALLOWWATER_USE_NUMBA", "").strip().lower()

if flag == "":
    # default: use numba if available
    USE_NUMBA = NUMBA_AVAILABLE
else:
    # explicit override: SHALLOWWATER_USE_NUMBA=0 / false / no / off disables it
    USE_NUMBA = (flag not in ("0", "false", "no", "off")) and NUMBA_AVAILABLE

if USE_NUMBA:

    from .operators_numba import (
        v_on_u_nb as v_on_u,
        u_on_v_nb as u_on_v,
        avg_u_to_center_nb as avg_u_to_center,
        avg_v_to_center_nb as avg_v_to_center,
        avg_center_to_u_nb as avg_center_to_u,
        avg_center_to_v_nb as avg_center_to_v,
        grad_x_on_u_nb    as grad_x_on_u,
        grad_y_on_v_nb    as grad_y_on_v,
        divergence_nb     as divergence,
        curl_on_center_nb as curl_on_center,
        laplacian_u_nb    as laplacian_u,
        laplacian_v_nb    as laplacian_v,
    )

else:

    def avg_u_to_center(u: np.ndarray) -> np.ndarray:
        """Average u (Ny, Nx+1) to centers (Ny, Nx)."""
        Ny, Nx1 = u.shape
        Nx = Nx1 - 1
        return 0.5 * (u[:, :Nx] + u[:, 1:])
    
    def avg_v_to_center(v: np.ndarray) -> np.ndarray:
        """Average v (Ny+1, Nx) to centers (Ny, Nx)."""
        Ny1, Nx = v.shape
        Ny = Ny1 - 1
        return 0.5 * (v[:Ny, :] + v[1:, :])
    
    def avg_center_to_u(c: np.ndarray) -> np.ndarray:
        """Average centers (Ny, Nx) to u-grid (Ny, Nx+1)."""
        Ny, Nx = c.shape
        out = np.zeros((Ny, Nx + 1), dtype=c.dtype)
        out[:, 1:Nx] = 0.5 * (c[:, :-1] + c[:, 1:])
        out[:, 0] = c[:, 0]
        out[:, -1] = c[:, -1]
        return out
    
    def avg_center_to_v(c: np.ndarray) -> np.ndarray:
        """Average centers (Ny, Nx) to v-grid (Ny+1, Nx)."""
        Ny, Nx = c.shape
        out = np.zeros((Ny + 1, Nx), dtype=c.dtype)
        out[1:Ny, :] = 0.5 * (c[:-1, :] + c[1:, :])
        out[0, :] = c[0, :]
        out[-1, :] = c[-1, :]
        return out
    
    def grad_x_on_u(c: np.ndarray, dx: float) -> np.ndarray:
        """
        ∂c/∂x from centers (Ny, Nx) to u-grid (Ny, Nx+1),
        with copied edge gradients to avoid spurious large values.
        """
        Ny, Nx = c.shape
        out = np.empty((Ny, Nx + 1), dtype=c.dtype)
        out[:, 1:Nx] = (c[:, 1:] - c[:, :-1]) / dx
        out[:, 0] = out[:, 1]
        out[:, -1] = out[:, -2]
        return out
    
    def grad_y_on_v(c: np.ndarray, dy: float) -> np.ndarray:
        """
        ∂c/∂y from centers (Ny, Nx) to v-grid (Ny+1, Nx),
        with copied edge gradients to avoid spurious large values.
        """
        Ny, Nx = c.shape
        out = np.empty((Ny + 1, Nx), dtype=c.dtype)
        out[1:Ny, :] = (c[1:, :] - c[:-1, :]) / dy
        out[0, :] = out[1, :]
        out[-1, :] = out[-2, :]
        return out
    
    def divergence(Fx: np.ndarray, Fy: np.ndarray, dx: float, dy: float) -> np.ndarray:
        """
        ∇⋅F from C-grid fluxes:
          Fx (Ny, Nx+1), Fy (Ny+1, Nx) → div at centers (Ny, Nx).
        """
        Ny, Nx1 = Fx.shape
        Ny1, Nx = Fy.shape
        Ny0 = Ny
        Nx0 = Nx
        out = np.empty((Ny0, Nx0), dtype=Fx.dtype)
        out[:, :] = ((Fx[:, 1:] - Fx[:, :-1]) / dx +
                     (Fy[1:, :] - Fy[:-1, :]) / dy)
        return out
    
    def v_on_u(v: np.ndarray) -> np.ndarray:
        """
        Interpolate v (Ny+1, Nx) to u-grid (Ny, Nx+1) by 4-point average
        in the standard C-grid pattern, leaving edge columns zero.
        """
        Ny1, Nx = v.shape
        Ny = Ny1 - 1
        out = np.zeros((Ny, Nx + 1), dtype=v.dtype)
        out[:, 1:Nx] = 0.25 * (v[:-1, :-1] + v[1:, :-1] +
                               v[:-1,  1:] + v[1:,  1:])
        return out
    
    def u_on_v(u: np.ndarray) -> np.ndarray:
        """
        Interpolate u (Ny, Nx+1) to v-grid (Ny+1, Nx) by 4-point average,
        leaving edge rows zero.
        """
        Ny, Nx1 = u.shape
        Nx = Nx1 - 1
        out = np.zeros((Ny + 1, Nx), dtype=u.dtype)
        out[1:Ny, :] = 0.25 * (u[:-1, :-1] + u[:-1, 1:] +
                               u[1:,  :-1] + u[1:,  1:])
        return out
    
    def curl_on_center(u: np.ndarray, v: np.ndarray, dx: float, dy: float) -> np.ndarray:
        """
        Relative vorticity ζ = ∂x v − ∂y u on cell centers (Ny, Nx),
        from C-grid u (Ny, Nx+1) and v (Ny+1, Nx).
        """
        Ny, Nx1 = u.shape
        Ny1, Nx = v.shape
    
        # dv/dx on v points, pad x-edges by copying neighbors
        dv_dx = np.zeros_like(v)
        dv_dx[:, 1:Nx] = (v[:, 1:Nx] - v[:, 0:Nx-1]) / dx
        dv_dx[:, 0] = dv_dx[:, 1]
        dv_dx[:, -1] = dv_dx[:, -2]
        # average to centers in y
        dv_dx_c = 0.5 * (dv_dx[0:Ny, :] + dv_dx[1:Ny1, :])     # (Ny, Nx)
    
        # du/dy on u points, pad y-edges by copying neighbors
        du_dy = np.zeros_like(u)
        du_dy[1:Ny, :] = (u[1:Ny, :] - u[0:Ny-1, :]) / dy
        du_dy[0, :] = du_dy[1, :]
        du_dy[-1, :] = du_dy[-2, :]
        # average to centers in x
        du_dy_c = 0.5 * (du_dy[:, 0:Nx] + du_dy[:, 1:Nx1])     # (Ny, Nx)
    
        return dv_dx_c - du_dy_c
    
    def laplacian_u(u: np.ndarray, dx: float, dy: float) -> np.ndarray:
        """5-point Laplacian on u-grid (Ny, Nx+1) with copy-edge padding."""
        Ny, Nx1 = u.shape
        out = np.zeros_like(u)
        invdx2 = 1.0 / (dx * dx)
        invdy2 = 1.0 / (dy * dy)
        for j in range(Ny):
            jm = j-1 if j > 0 else 0
            jp = j+1 if j < Ny-1 else Ny-1
            for i in range(Nx1):
                im = i-1 if i > 0 else 0
                ip = i+1 if i < Nx1-1 else Nx1-1
                out[j, i] = ((u[j, ip] - 2*u[j, i] + u[j, im]) * invdx2 +
                             (u[jp, i] - 2*u[j, i] + u[jm, i]) * invdy2)
        return out
    
    def laplacian_v(v: np.ndarray, dx: float, dy: float) -> np.ndarray:
        """5-point Laplacian on v-grid (Ny+1, Nx) with copy-edge padding."""
        Ny1, Nx = v.shape
        out = np.zeros_like(v)
        invdx2 = 1.0 / (dx * dx)
        invdy2 = 1.0 / (dy * dy)
        for j in range(Ny1):
            jm = j-1 if j > 0 else 0
            jp = j+1 if j < Ny1-1 else Ny1-1
            for i in range(Nx):
                im = i-1 if i > 0 else 0
                ip = i+1 if i < Nx-1 else Nx-1
                out[j, i] = ((v[j, ip] - 2*v[j, i] + v[j, im]) * invdx2 +
                             (v[jp, i] - 2*v[j, i] + v[jm, i]) * invdy2)
        return out
