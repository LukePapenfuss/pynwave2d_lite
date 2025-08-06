import numpy as np
from numba import njit
import sys
import os
import burgutils


# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nwave import *


class Burgers(Equations):
    def __init__(self, NU, g, apply_bc=BCType.NONE):
        super().__init__(NU, g, apply_bc)
        self.U_PHI = 0
        self.alpha = 0.0

    def set_alpha(self, alpha):
        self.alpha = alpha

    def rhs(self, dtu, u, g, t): 
        dtphi = dtu[0]
        phi = u[0]
        X = g.xi[0] # Creates linspaces X and Y based on the given grid
        dxxphi = g.D2.grad(phi) # Defines d^2phi\dx^2
        dtphi[:] = self.alpha * dxxphi[:]

        # BCs
        # dtphi[0] = 0.0
        # dtphi[-1] = 0.0

    def initialize(self, g, params):
        x = g.xi[0]
        x0 = params.get("id_x0", 0.0)
        omega = params.get("id_omega", 100.0)
    # Initial condition for phi(x, 0), smooth bump
        self.u[0][:] = 1 + np.exp(-omega * (x - x0)**2)
        if params["id_type"] == "gaussian":
            x0 = params["id_x0"]
            amp, omega = params["id_amp"], params["id_omega"]
            self.u[0][:] = amp * np.exp(-omega * ((x - x0) ** 2))

        elif params["id_type"] == "colehopf":
            self.u[0] = cole_hopf_solution(x, 0.0, self.alpha)

        else:
            raise ValueError(f"Unknown id_type: {params['id_type']}")
        

    def apply_bcs(self, u, g, t):
        phi = u[0]
        alpha = self.alpha
        x = g.xi[0]
        phi[0] = 1.0 + np.exp(-(x[0] - t) / alpha)
        phi[-1] = 1.0 + np.exp(-(x[-1] - t) / alpha)

def cole_hopf_solution(x, t, alpha, c0=1.0, c1=2.0, m=1, n=2, A=1):
    pi = np.pi
    expm = np.exp(-alpha**2 * m**2 * pi**2 * t)
    expn = np.exp(-alpha**2 * n**2 * pi**2 * t)
    alpha2 = alpha**2

    u = np.empty_like(x)
    numerator = np.empty_like(x)
    denominator = np.empty_like(x)

    numerator[:] = (
        c1 - c0
        + m * pi * expm * np.cos(m * pi * x[:])
        - A * n * pi * expn * np.cos(n * pi * x[:])
    )

    denominator[:] = (
        c1 * x[:]
        + c0 * (1 - x[:])
        + expm * np.sin(m * pi * x[:])
        - A * expn * np.sin(n * pi * x[:])
    )

    phi = burg.u[0]
    dphidx = g.D1.grad(phi)
    u_numeric = -2 * alpha * dphidx / phi

    return u
