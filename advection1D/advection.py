import numpy as np
from numba import njit
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nwave import *


class Advection(Equations):
    def __init__(self, NU, g, apply_bc=BCType.NONE):
        super().__init__(NU, g, apply_bc)
        self.U_PHI = 0
        self.alpha = 0.0
        self.c = 0.0

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_c(self, c):
        self.c = c

    def rhs(self, dtu, u, g): # Creates the rhs of the equation dphi\dt = ...
        dtphi = dtu[0]
        phi = u[0]
        X = g.xi[0] # Creates linspaces X and Y based on the given grid

        dxphi = g.D1.grad(phi) # Defines dphi\dx
        dxxphi = g.D2.grad(phi) # Defines d^2phi\dx^2
        dtphi[:] = (
            -self.c * dxphi[:] + self.alpha * (dxxphi[:])
        ) # This is the rhs of an advection equation with a dissipative term

        # BCs
        dtphi[0] = 0.0
        # dtphi[-1] = 0.0

    def initialize(self, g, params):
        x = g.xi[0]
        x0 = params["id_x0"]
        amp, omega = params["id_amp"], params["id_omega"]
        # X = np.meshgrid(x, y, indexing="ij")
        # self.u[0][:] = amp * np.exp(-omega * ((x - x0) ** 2))
        self.u[0][:] = np.sin(10 * np.pi * x)

    def apply_bcs(self, u, g):
        print("no bcs")
