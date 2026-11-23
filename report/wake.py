"""Where the Star-CCM+ wake is unsteady: temporal std of the velocity per 1D strip in x.

    python3 wake.py

The rod forces are steady but the far wake sheds vortices.  Reads every 5th snapshot of the
last 20 % of the Star-CCM+ run (tU/D 2400-3000, the force-averaging window) from the full
snapshot archive (QCFD_STAR_TAR, not shipped) and writes data/archive/star_wake.csv: strip
start x/D and the largest temporal standard deviation of u and v in that strip, in units of
U.  The archived file is enough for `make report`.
Used by  plots.py.
"""

import os
import tarfile
import tempfile

import numpy as np

from _paths import OUT, STAR_TAR
from star_vtk import read_star_vtk

SNAPSHOTS = range(240, 301, 5)  # every 5th snapshot of the force-averaging window
STRIPS = range(-25, 75)  # 1D strips in x, in D


def main():
    velocity, x = [], None
    with tarfile.open(STAR_TAR) as archive, tempfile.TemporaryDirectory() as tmp:
        for k in SNAPSHOTS:
            path = os.path.join(tmp, "snapshot.vtk")
            with open(path, "wb") as fh:
                fh.write(archive.extractfile(f"vtkfiles/cyl_array@{k}000000.vtk").read())
            centres, cells = read_star_vtk(path)
            velocity.append(cells["Velocity"][:, :2])
            x = centres[:, 0]
    write_strip_statistics(x, np.array(velocity).std(0), f"{OUT}/star_wake.csv")
    print("wrote", f"{OUT}/star_wake.csv")


def write_strip_statistics(x, std, path):
    """Largest temporal std of u and v in each 1D strip x0 <= x < x0 + 1."""
    with open(path, "w") as fh:
        fh.write("x,std_u,std_v\n")
        for x0 in STRIPS:
            strip = (x >= x0) & (x < x0 + 1)
            fh.write(f"{x0},{std[strip, 0].max():.4f},{std[strip, 1].max():.4f}\n")


if __name__ == "__main__":
    main()
