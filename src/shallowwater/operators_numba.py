# src/shallowwater/numba_ops.py
import numpy as np
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def deco(f): return f
        return deco
    def prange(*args):
        return range(*args)

@njit(cache=True, fastmath=True)
def avg_u_to_center_nb(u):
    Ny, Nx1 = u.shape
    Nx = Nx1 - 1
    out = np.empty((Ny, Nx), dtype=u.dtype)
    for j in range(Ny):
        for i in range(Nx):
            out[j, i] = 0.5*(u[j, i] + u[j, i+1])
    return out

@njit(cache=True, fastmath=True)
def avg_v_to_center_nb(v):
    Ny1, Nx = v.shape
    Ny = Ny1 - 1
    out = np.empty((Ny, Nx), dtype=v.dtype)
    for j in range(Ny):
        for i in range(Nx):
            out[j, i] = 0.5*(v[j, i] + v[j+1, i])
    return out

@njit(cache=True, fastmath=True)
def avg_center_to_u_nb(c):
    Ny, Nx = c.shape
    out = np.empty((Ny, Nx+1), dtype=c.dtype)
    for j in range(Ny):
        out[j, 0] = c[j, 0]
        for i in range(1, Nx):
            out[j, i] = 0.5*(c[j, i-1] + c[j, i])
        out[j, Nx] = c[j, Nx-1]
    return out

@njit(cache=True, fastmath=True)
def avg_center_to_v_nb(c):
    Ny, Nx = c.shape
    out = np.empty((Ny+1, Nx), dtype=c.dtype)
    for i in range(Nx):
        out[0, i] = c[0, i]
        for j in range(1, Ny):
            out[j, i] = 0.5*(c[j-1, i] + c[j, i])
        out[Ny, i] = c[Ny-1, i]
    return out

@njit(cache=True, fastmath=True)
def grad_x_on_u_nb(c, dx):
    Ny, Nx = c.shape
    out = np.empty((Ny, Nx + 1), dtype=c.dtype)
    out[:, 1:Nx] = (c[:, 1:] - c[:, :-1]) / dx
    out[:, 0] = out[:, 1]
    out[:, -1] = out[:, -2]
    return out

@njit(cache=True, fastmath=True)
def grad_y_on_v_nb(c, dy):
    Ny, Nx = c.shape
    out = np.empty((Ny + 1, Nx), dtype=c.dtype)
    out[1:Ny, :] = (c[1:, :] - c[:-1, :]) / dy
    out[0, :] = out[1, :]
    out[-1, :] = out[-2, :]
    return out

@njit(cache=True, fastmath=True)
def divergence_nb(Fx, Fy, dx, dy):
    # Fx: (Ny, Nx+1), Fy: (Ny+1, Nx) -> (Ny, Nx)
    Ny, Nx1 = Fx.shape
    Ny1, Nx = Fy.shape
    Nx0 = Nx1 - 1
    Ny0 = Ny1 - 1
    out = np.empty((Ny0, Nx0), dtype=Fx.dtype)
    for j in range(Ny0):
        for i in range(Nx0):
            dFx = (Fx[j, i+1] - Fx[j, i]) / dx
            dFy = (Fy[j+1, i] - Fy[j, i]) / dy
            out[j, i] = dFx + dFy
    return out

@njit(cache=True, fastmath=True)
def curl_on_center_nb(u, v, dx, dy):
    # ƒÄ = Ýx v | Ýy u at centers
    Ny, Nx1 = u.shape
    Ny1, Nx = v.shape
    Ny0 = Ny
    Nx0 = Nx
    out = np.empty((Ny0, Nx0), dtype=u.dtype)

    # dv/dx to centers
    for j in range(Ny0):
        for i in range(Nx0):
            iL = i-1 if i>0 else 0
            iR = i if i < Nx-1 else Nx-2
            dv_dx_left  = (v[j,   i] - v[j,   iL]) / dx
            dv_dx_right = (v[j+1, i] - v[j+1, iL]) / dx
            dv_dx_c = 0.5*(dv_dx_left + dv_dx_right)

            # du/dy to centers
            jB = j-1 if j>0 else 0
            jT = j if j < Ny-1 else Ny-2
            du_dy_bot = (u[j,   i+1] - u[jB, i+1]) / dy
            du_dy_top = (u[j+1, i+1] - u[jT, i+1]) / dy
            du_dy_c = 0.5*(du_dy_bot + du_dy_top)

            out[j, i] = dv_dx_c - du_dy_c
    return out

@njit(cache=True, fastmath=True)
def laplacian_u_nb(u, dx, dy):
    Ny, Nx1 = u.shape
    out = np.empty_like(u)
    invdx2 = 1.0/(dx*dx)
    invdy2 = 1.0/(dy*dy)
    for j in range(Ny):
        for i in range(Nx1):
            jm = j-1 if j>0 else 0
            jp = j+1 if j<Ny-1 else Ny-1
            im = i-1 if i>0 else 0
            ip = i+1 if i<Nx1-1 else Nx1-1
            out[j, i] = (u[j, ip] - 2*u[j, i] + u[j, im])*invdx2 + (u[jp, i] - 2*u[j, i] + u[jm, i])*invdy2
    return out

@njit(cache=True, fastmath=True)
def laplacian_v_nb(v, dx, dy):
    Ny1, Nx = v.shape
    out = np.empty_like(v)
    invdx2 = 1.0/(dx*dx)
    invdy2 = 1.0/(dy*dy)
    for j in range(Ny1):
        for i in range(Nx):
            jm = j-1 if j>0 else 0
            jp = j+1 if j<Ny1-1 else Ny1-1
            im = i-1 if i>0 else 0
            ip = i+1 if i<Nx-1 else Nx-1
            out[j, i] = (v[j, ip] - 2*v[j, i] + v[j, im])*invdx2 + (v[jp, i] - 2*v[j, i] + v[jm, i])*invdy2
    return out

@njit(cache=True, fastmath=True)
def v_on_u_nb(v):
    """
    Interpolate v (Ny+1, Nx) to u-locations (Ny, Nx+1) using a 4-point average.
    Matches the NumPy reference:
        out[:, 1:Nx] = 0.25*(v[:-1, :-1] + v[1:, :-1] + v[:-1, 1:] + v[1:, 1:])
    Leaves the first and last u-columns (i=0 and i=Nx) as zeros.
    """
    Ny1, Nx = v.shape
    Ny = Ny1 - 1
    out = np.zeros((Ny, Nx + 1), dtype=v.dtype)
    for j in range(Ny):        # 0 .. Ny-1
        for i in range(1, Nx): # interior u-columns only
            out[j, i] = 0.25 * (v[j, i-1] + v[j+1, i-1] + v[j, i] + v[j+1, i])
    return out

@njit(cache=True, fastmath=True)
def u_on_v_nb(u):
    """
    Interpolate u (Ny, Nx+1) to v-locations (Ny+1, Nx) using a 4-point average.
    Matches the NumPy reference:
        out[1:Ny, :] = 0.25*(u[:-1, :-1] + u[:-1, 1:] + u[1:, :-1] + u[1:, 1:])
    Leaves the first and last v-rows (j=0 and j=Ny) as zeros.
    """
    Ny, Nx1 = u.shape
    Nx = Nx1 - 1
    out = np.zeros((Ny + 1, Nx), dtype=u.dtype)
    for j in range(1, Ny):     # interior v-rows only
        for i in range(Nx):
            out[j, i] = 0.25 * (u[j-1, i] + u[j-1, i+1] + u[j, i] + u[j, i+1])
    return out
