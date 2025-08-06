import numpy as np
import sys
import os
import tomllib

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import advection
import advectutils as autil
from nwave import *
import nwave.ioxdmf as iox

def main(parfile):
    # Read parameters
    with open(parfile, "rb") as f:
        params = tomllib.load(f)

    g = Grid1D(params) # 2D Grid for running operations
    x = g.xi[0] # X component of the grid (as a linspace)
    dx = g.dx[0] # (xmax - xmin) / (nx - 1)

    D1, D2 = autil.init_derivative_operators(x, params) # Gets the derivative operators from the parameter file
    g.set_D1(D1) # Sets the D1 Operator in the grid
    g.set_D2(D2) # Sets the D2 Operator in the grid

    """
    Need to write 2D filters

    F1 = KreissOligerFilterO6_2D( dx, 0.1, apply_diss_boundaries=False)
    g.set_filter(F1)
    print(f"Filter type: {g.Filter.get_filter_type()}")
    print(f"Filter apply: {g.Filter.get_apply_filter()}")
    print(f"Filter sigma: {F1.get_sigma()}")
    """

    eqs = advection.Advection(1, g) # Creates the equation (1: Number of PDEs in system, g: grid)
    eqs.initialize(g, params) # Initializes the equation with the grid and more parameters (Initial Conditions, eq parameters, ...)
    eqs.set_alpha(params["diss_alpha"]) # Sets the dissipation constant
    eqs.set_c(params["c"])

    output_dir = params["output_dir"] # Sets the output directory
    output_interval = params["output_interval"] # Determines the frequency of outputs
    os.makedirs(output_dir, exist_ok=True) # Create the directories if they don't exist

    # dt = params["cfl"] * dx
    dt = 1.0e-3 # Time stepping constant
    rk4 = RK4(eqs, g) # Create an rk4 time-integrator with the equations and the grid

    time = 0.0 # Start the time at 0.0

    func_names = ["phi"] # Name the function phi
    fname = f"{output_dir}/advection1D_0000.curve"
    autil.write_curve(fname, time, g.xi[0], eqs)
    # iox.write_hdf5(0, eqs.u, x, y, func_names, output_dir) # Print the initial state of the equation

    Nt = params["Nt"] # Total time to be integrated over
    for i in range(1, Nt + 1): # Loop through all time steps
        rk4.step(eqs, g, dt) # Update the equation using rk4
        time += dt # increment the time by dt
        print(f"Step {i:d}  t={time:.2f}") # Print which step and time we are on
        if i % output_interval == 0: # Check if we are supposed to output on this step
            # iox.write_hdf5(i, eqs.u, x, y, func_names, output_dir) # Output the current equation circumstance into the output file    
            fname = f"{output_dir}/advection1D_{i:04d}.curve"
            autil.write_curve(fname, time, g.xi[0], eqs)

    Nx = g.shp[0] # Determine the X-dimension of the grid
    # iox.write_xdmf(output_dir, Nt, Nx, Ny, func_names, output_interval, dt) # Create the compiled folder of all of the timesteps on every point



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage:  python solver.py <parfile>")
        sys.exit(1)

    parfile = sys.argv[1]
    main(parfile)
