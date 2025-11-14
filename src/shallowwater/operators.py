# src/shallowwater/operators.py
import numpy as np
from .operators_numba import NUMBA_AVAILABLE
if NUMBA_AVAILABLE:
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

    def v_on_u(v):
        Ny1, Nx = v.shape
        Ny = Ny1 - 1
        out = np.zeros((Ny, Nx + 1), dtype=v.dtype)
        out[:, 1:Nx] = 0.25 * (v[:-1, :-1] + v[1:, :-1] + v[:-1, 1:] + v[1:, 1:])
        return out
        
    def u_on_v(u):
        Ny, Nx1 = u.shape
        Nx = Nx1 - 1
        out = np.zeros((Ny + 1, Nx), dtype=u.dtype)
        out[1:Ny, :] = 0.25 * (u[:-1, :-1] + u[:-1, 1:] + u[1:, :-1] + u[1:, 1:])
        return out

    def avg_center_to_u(eta):
        Ny, Nx = eta.shape
        out = np.zeros((Ny, Nx + 1), dtype=eta.dtype)
        out[:, 1:Nx] = 0.5 * (eta[:, :-1] + eta[:, 1:])
        out[:, 0] = eta[:, 0]
        out[:, -1] = eta[:, -1]
        return out
    
    def avg_center_to_v(eta):
        Ny, Nx = eta.shape
        out = np.zeros((Ny + 1, Nx), dtype=eta.dtype)
        out[1:Ny, :] = 0.5 * (eta[:-1, :] + eta[1:, :])
        out[0, :] = eta[0, :]
        out[-1, :] = eta[-1, :]
        return out
    
    def grad_x_on_u(eta, dx):
        Ny, Nx = eta.shape
        gout = np.zeros((Ny, Nx + 1), dtype=eta.dtype)
        gout[:, 1:Nx] = (eta[:, 1:] - eta[:, :-1]) / dx
        gout[:, 0] = gout[:, 1]
        gout[:, -1] = gout[:, -2]
        return gout
    
    def grad_y_on_v(eta, dy):
        Ny, Nx = eta.shape
        gout = np.zeros((Ny + 1, Nx), dtype=eta.dtype)
        gout[1:Ny, :] = (eta[1:, :] - eta[:-1, :]) / dy
        gout[0, :] = gout[1, :]
        gout[-1, :] = gout[-2, :]
        return gout
    
    def divergence(Fx, Fy, dx, dy):
        return (Fx[:, 1:] - Fx[:, :-1]) / dx + (Fy[1:, :] - Fy[:-1, :]) / dy
        
    def avg_u_to_center(u: np.ndarray) -> np.ndarray:
        """Average zonal velocity from u-faces (Ny, Nx+1) to cell centers (Ny, Nx)."""
        Ny, Nx1 = u.shape
        Nx = Nx1 - 1
        return 0.5 * (u[:, :Nx] + u[:, 1:])
    
    def avg_v_to_center(v: np.ndarray) -> np.ndarray:
        """Average meridional velocity from v-edges (Ny+1, Nx) to cell centers (Ny, Nx)."""
        Ny1, Nx = v.shape
        Ny = Ny1 - 1
        return 0.5 * (v[:Ny, :] + v[1:, :])
    
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
        """5-point Laplacian on the u-grid (Ny, Nx+1), copy-edge padding."""
        Ny, Nx1 = u.shape
        out = np.zeros_like(u)
    
        # second differences in x (columns 1..Nx-2)
        out[:, 1:Nx1-1] += (u[:, 2:] - 2*u[:, 1:-1] + u[:, :-2]) / (dx*dx)
        # copy-edge pad for x
        out[:, 0]  += (u[:, 1]  - 2*u[:, 0]  + u[:, 1])  / (dx*dx)
        out[:, -1] += (u[:, -2] - 2*u[:, -1] + u[:, -2]) / (dx*dx)
    
        # second differences in y (rows 1..Ny-2)
        out[1:Ny-1, :] += (u[2:, :] - 2*u[1:-1, :] + u[:-2, :]) / (dy*dy)
        # copy-edge pad for y
        out[0,  :]     += (u[1,  :] - 2*u[0,  :] + u[1,  :]) / (dy*dy)
        out[-1, :]     += (u[-2, :] - 2*u[-1, :] + u[-2, :]) / (dy*dy)
    
        return out
    
    def laplacian_v(v: np.ndarray, dx: float, dy: float) -> np.ndarray:
        """5-point Laplacian on the v-grid (Ny+1, Nx), copy-edge padding."""
        Ny1, Nx = v.shape
        out = np.zeros_like(v)
    
        # x-direction
        out[:, 1:Nx-1] += (v[:, 2:] - 2*v[:, 1:-1] + v[:, :-2]) / (dx*dx)
        out[:, 0]      += (v[:, 1]  - 2*v[:, 0]   + v[:, 1])   / (dx*dx)
        out[:, -1]     += (v[:, -2] - 2*v[:, -1]  + v[:, -2])  / (dx*dx)
    
        # y-direction
        out[1:Ny1-1, :] += (v[2:, :] - 2*v[1:-1, :] + v[:-2, :]) / (dy*dy)
        out[0,  :]      += (v[1,  :] - 2*v[0,   :] + v[1,   :])  / (dy*dy)
        out[-1, :]      += (v[-2, :] - 2*v[-1,  :] + v[-2,  :])  / (dy*dy)

        return out
