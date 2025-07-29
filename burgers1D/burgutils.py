import numpy as np
from nwave import *

def get_filter_type(filter_str):
    try:
        return filter_type_map[filter_str]
    except KeyError:
        raise ValueError(f"Unknown filter type string: '{filter_str}'")


def get_filter_apply(filter_str):
    try:
        return filter_apply_map[filter_str]
    except KeyError:
        raise ValueError(f"Unknown filter apply string: '{filter_str}'")


def get_d1_type(d_str):
    try:
        return d1_type_map[d_str]
    except KeyError:
        raise ValueError(f"Unknown D1 type string: '{d_str}'")


def get_d2_type(d_str):
    try:
        return d2_type_map[d_str]
    except KeyError:
        raise ValueError(f"Unknown D2 type string: '{d_str}'")

def get_cfd_solve(d_str):
    try:
        return cfd_solve_map[d_str]
    except KeyError:
        raise ValueError(f"Unknown Deriv Solve type string: '{d_str}'")

def init_derivative_operators(x, params):
    d1type = get_d1_type(params["D1"]) # Read D1 type from parameter file
    d2type = get_d2_type(params["D2"]) # Read D2 type from parameter file
    method_str = params.get("DerivSolveMethod", "LUSOLVE") 
    method = get_cfd_solve(method_str)
    dx = x[1] - x[0] # Determine the grid spacing in x

    print(f"init_derivative_operators>> Setting D1 type: {d1type}")
    if d1type == DerivType.D1_E44:
        D1 = ExplicitFirst44_1D(dx)
    elif d1type == DerivType.D1_E642:
        D1 = ExplicitFirst642_1D(dx)
    elif d1type in CompactFirstDerivatives: # If the derivative type is a compact first derivative operator
        D1 = NCompactDerivative.deriv(x, d1type, method) # Define the D1 operator in the X direction
    else:
        raise NotImplementedError("D1 Type = {d1type}")

    print(f"init_derivative_operators>>  Setting D2 type: {d2type}")
    if d2type == DerivType.D2_E44:
        D2 = ExplicitSecond44_1D(dx)
    elif d2type == DerivType.D2_E642:
        D2 = ExplicitSecond642_1D(dx)
    elif d2type in CompactSecondDerivatives:
        D2 = NCompactDerivative.deriv(x, d2type, method)
    else:
        raise NotImplementedError("D2 Type = {d2type}")

    return D1, D2

def write_curve(filename, time, x, eqs):
    with open(filename, "w") as f:
        f.write(f"# TIME {time}\n")
        f.write(f"# phi\n")
        for xi, di in zip(x, eqs.u[0]):
            f.write(f"{xi:.8e} {di:.8e}\n")
        

def convergence(lowres_file, medres_file, highres_file):
    lowres = np.loadtxt(lowres_file, skiprows=3)
    medres = np.loadtxt(medres_file, skiprows=5)
    highres = np.loadtxt(highres_file, skiprows=7)

    # Subsample to align resolutions
    medres = medres[::2]
    highres = highres[::4]

    # Trim to smallest length to ensure alignment
    n = min(len(lowres), len(medres), len(highres))
    lowres = lowres[:n]
    medres = medres[:n]
    highres = highres[:n]

    u_low = lowres[:, 1]
    u_med = medres[:, 1]
    u_high = highres[:, 1]

    diff1 = np.linalg.norm(u_med - u_low)
    diff2 = np.linalg.norm(u_high - u_med)

    # Convergence ratio (scalar)
    conv_ratio = diff1 / diff2 if diff2 != 0 else 0.0

    return conv_ratio