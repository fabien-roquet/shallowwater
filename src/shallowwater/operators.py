import numpy as np

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

def divergence(Fx, Fy, dx, dy):
    return (Fx[:, 1:] - Fx[:, :-1]) / dx + (Fy[1:, :] - Fy[:-1, :]) / dy
