from dataclasses import dataclass

@dataclass
class ModelParams:
    g: float = 9.81
    H: float = 1000.0
    rho: float = 1025.0
    f0: float = 1.0e-4
    beta: float = 2.0e-11
    y0: float = 0.0
    r: float = 0.0  # Rayleigh friction rate [1/s]
    linear: bool = True  # keep linear; advection can be added later via hooks
