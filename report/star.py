"""Star-CCM+ velocity and vorticity on a regular grid (linear interpolation of its cells).

The solver's own warm start is nearest-neighbour; that is fine as an initial
condition but its stair steps show up as stripes in maps and derivatives, so
figures use this instead.  Coordinates are in D with the bundle centre at 0.
"""
import os
import tarfile

import numpy as np

from _paths import STAR_TAR as TAR
from star_vtk import read_star_vtk
from _paths import REF as DATA  # cached star_t*.vtk snapshots live with the monitors


def star_on_grid(gx, gy, t=2500):
    """(u, v, omega_z) at the points gx x gy, shape (len(gx), len(gy), 3), at tU/D = t.

    Snapshots exist every 10 time units.  The rod forces are steady, the far wake is not.
    """
    from scipy.interpolate import LinearNDInterpolator

    cache = os.path.join(DATA, f"star_t{t}.vtk")
    if not os.path.exists(cache):
        with tarfile.open(TAR) as tf, open(cache, "wb") as fh:
            fh.write(tf.extractfile(f"vtkfiles/cyl_array@{t * 100000}.vtk").read())
    xy, cells = read_star_vtk(cache)
    vals = np.column_stack([cells["Velocity"][:, :2], cells["Vorticityk"]])
    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    return LinearNDInterpolator(xy, vals)(GX, GY)
