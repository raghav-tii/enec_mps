"""Where the Star-CCM+ wake is unsteady: temporal std of the velocity per 1D strip in x.

    python3 wake.py

The rod forces are steady but the far wake sheds vortices.  Reads every 5th snapshot of the last 20 % of
the Star-CCM+ run (tU/D 2400-3000, the force-averaging window) from the full snapshot archive
(QCFD_STAR_TAR, not shipped) and writes data/archive/star_wake.csv: strip start x/D and the largest
temporal standard deviation of u and v in that strip, in units of U.  The archived file is enough
for `make report`.
Used by  plots.py.
"""
import os
import tarfile
import tempfile

import numpy as np

from star import TAR
from star_vtk import read_star_vtk
from _paths import OUT

V = []
with tarfile.open(TAR) as t, tempfile.TemporaryDirectory() as tmp:
    for k in range(240, 301, 5):
        path = os.path.join(tmp, "s.vtk")
        with open(path, "wb") as fh:
            fh.write(t.extractfile(f"vtkfiles/cyl_array@{k}000000.vtk").read())
        centres, cells = read_star_vtk(path)
        V.append(cells["Velocity"][:, :2])
x = centres[:, 0]
sd = np.array(V).std(0)
with open(f"{OUT}/star_wake.csv", "w") as fh:
    fh.write("x,std_u,std_v\n")
    for x0 in range(-25, 75):
        s = (x >= x0) & (x < x0 + 1)
        fh.write(f"{x0},{sd[s, 0].max():.4f},{sd[s, 1].max():.4f}\n")
print("wrote data/star_wake.csv")
