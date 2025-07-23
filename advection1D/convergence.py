import numpy as np
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import advectutils as autil
from nwave import *
import nwave.ioxdmf as iox

def main(lowres_dir, medres_dir, highres_dir, output_file="convergence.curve"):
    # Match filenames in all three dirs
    filenames = sorted(os.listdir(lowres_dir))
    convergence_series = []

    for fname in filenames:
        low_file = os.path.join(lowres_dir, fname)
        med_file = os.path.join(medres_dir, fname)
        high_file = os.path.join(highres_dir, fname)

        if not (os.path.isfile(low_file) and os.path.isfile(med_file) and os.path.isfile(high_file)):
            continue  # skip if one is missing

        conv = autil.convergence(low_file, med_file, high_file)
        conv_norm = np.linalg.norm(conv, ord=np.inf)  # e.g., max norm

        # Extract time from filename if encoded (or from file contents)
        # Example: filename = "phi_0005.curve" -> time = 0.005
        try:
            time = float(fname.strip("phi_").strip(".curve")) * 0.001
        except:
            time = len(convergence_series) * 0.001  # fallback

        convergence_series.append((time, conv_norm))

    output_dir = "convergence_output" # Sets the output directory
    os.makedirs(output_dir, exist_ok=True) # Create the directories if they don't exist

    # Write convergence curve file
    output_path = os.path.join(output_dir, output_file)
    with open(output_path, "w") as f:

        f.write("# time    convergence\n")
        for time, conv_val in convergence_series:
            f.write(f"{time:.8e} {conv_val:.8e}\n")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage:  python convergence.py <low> <med> <high>")
        sys.exit(1)

    import sys
    main(sys.argv[1], sys.argv[2], sys.argv[3])