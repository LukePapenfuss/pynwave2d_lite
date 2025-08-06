import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import advectutils as autil
from nwave import *

def main(lowres_dir, medres_dir, highres_dir):
    filenames = sorted(os.listdir(lowres_dir))
    convergence_series = []

    for fname in filenames:
        low_file = os.path.join(lowres_dir, fname)
        med_file = os.path.join(medres_dir, fname)
        high_file = os.path.join(highres_dir, fname)

        if not (os.path.isfile(low_file) and os.path.isfile(med_file) and os.path.isfile(high_file)):
            continue  # skip if one is missing

        conv = autil.convergence(low_file, med_file, high_file)

        try:
            time = float(fname.strip("phi_").strip(".curve")) * 0.001
        except:
            time = len(convergence_series) * 0.001  # fallback

        convergence_series.append((time, conv))

    convergence_series = np.array(convergence_series)
    times = convergence_series[:, 0]
    values = convergence_series[:, 1]

    output_file = "convergence_series.txt"
    np.savetxt(output_file, convergence_series, header="time convergence", comments='')
    print(f"Convergence series saved to {output_file}")

    # Plot with logarithmic Y-axis
    plt.figure(figsize=(8, 5))
    plt.semilogy(times, values, marker='o', linestyle='-', color='red', markersize=3)
    plt.xlabel("Time")
    plt.ylabel("Convergence")
    plt.title("Convergence vs Time")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylim(1e-1, 1000)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage:  python convergence.py <low> <med> <high>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2], sys.argv[3])
