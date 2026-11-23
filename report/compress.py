"""Truncation error of the Star-CCM+ field as an MPS on the solver's 1024^2 grid.

    python3 compress.py

Samples data/star_ccm/star_t2500.vtk onto the grid the way the solver's warm start does (nearest cell),
orders the 20 binary digits as the solver does (x and y interleaved, most significant first), and runs a
TT-SVD capped at each bond dimension.
Writes   data/archive/compress.json
Used by  plots.py.
"""
import json
import os

import numpy as np
from scipy.spatial import cKDTree

from star import DATA, star_on_grid
from star_vtk import read_star_vtk
from _paths import OUT

Q, CHIS = 10, (5, 10, 20, 40, 80, 160, 320)


def tt_error(T, chi):
    """Relative error of T after a left-to-right TT-SVD with bond dimension <= chi."""
    C, r, cores = T.reshape(1, -1), 1, []
    for _ in range(2 * Q - 1):
        U, S, Vt = np.linalg.svd(C.reshape(r * 2, -1), full_matrices=False)
        k = min(chi, len(S))
        cores.append(U[:, :k].reshape(r, 2, k))
        C, r = S[:k, None] * Vt[:k], k
    out = C.reshape(r, 2)
    for core in reversed(cores):
        out = np.tensordot(core, out.reshape(core.shape[2], -1), axes=(2, 0)).reshape(core.shape[0], -1)
    return float(np.linalg.norm(out.ravel() - T) / np.linalg.norm(T))


def interleave(f):
    """(x, y) grid -> flat vector in the solver's bit order x9 y9 x8 y8 ... x0 y0."""
    t = f.reshape([2] * (2 * Q))
    return t.transpose([a for k in range(Q) for a in (k, Q + k)]).ravel()


def selfcheck():
    x = np.linspace(0, 1, 2**Q)
    f = np.outer(np.sin(3 * x), np.cos(2 * x))  # sin and cos are TT rank 2 in binary
    assert tt_error(interleave(f), 4) < 1e-10
    assert tt_error(interleave(np.random.default_rng(0).random((2**Q, 2**Q))), 40) > 0.1


if __name__ == "__main__":
    selfcheck()
    star_on_grid(np.zeros(1), np.zeros(1))  # make sure the VTK is cached
    xy, cells = read_star_vtk(os.path.join(DATA, "star_t2500.vtk"))
    vals = np.column_stack([cells["Velocity"][:, :2], cells["Pressure"]])
    F = np.load(f"{OUT}/fields.npz")
    cen, h = F["centres"], 2.0 / 2 ** int(F["cfg_Lf"])
    g = (np.arange(2**Q) - cen[:, 0].mean()) * h, (np.arange(2**Q) - cen[:, 1].mean()) * h
    GX, GY = np.meshgrid(*g, indexing="ij")
    near = vals[cKDTree(xy).query(np.column_stack([GX.ravel(), GY.ravel()]))[1]]
    res = {}
    for c, name in enumerate(("u", "v", "p")):
        T = interleave(near[:, c].reshape(2**Q, 2**Q))
        res[name] = {chi: tt_error(T, chi) for chi in CHIS}
        print(name, {k: f"{v:.2e}" for k, v in res[name].items()}, flush=True)
    json.dump(res, open(f"{OUT}/compress.json", "w"), indent=1)
