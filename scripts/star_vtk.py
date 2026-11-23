"""Reader for the Star-CCM+ VTK snapshots (ASCII unstructured grid, VTK 5.1); numpy only."""

import tarfile

import numpy as np


def read_star_vtk(path, member=None):
    """Cell centres (n, 2) and cell data {name: (n,) or (n, k)} of a VTK file.

    ``member`` reads that file out of the tar at ``path`` instead.  Centre =
    mean of the cell's vertices (exact for the triangles/quads this mesh is
    made of, the area centroid for its polygons).
    """
    if member:
        with tarfile.open(path) as t:
            raw = t.extractfile(member).read()
    else:
        with open(path, "rb") as f:
            raw = f.read()
    tok = raw.split()
    at = {
        k: i
        for i, k in reversed(list(enumerate(tok)))
        if k in (b"POINTS", b"OFFSETS", b"CONNECTIVITY", b"CELL_TYPES", b"CELL_DATA")
    }

    def nums(i, n, dtype=float):  # n numbers from token i
        return np.array(tok[i : i + n], dtype=dtype)

    i = at[b"POINTS"]
    pts = nums(i + 3, 3 * int(tok[i + 1])).reshape(-1, 3)[:, :2]
    i = at[b"OFFSETS"]
    ncell = int(tok[at[b"CELL_TYPES"] + 1])
    off = nums(i + 2, ncell + 1, np.int64)
    conn = nums(at[b"CONNECTIVITY"] + 2, off[-1], np.int64)
    n = np.diff(off)
    centres = np.add.reduceat(pts[conn], off[:-1]) / n[:, None]
    # polygons (> 4 vertices) can have collinear ones that bias the mean:
    # use the area centroid there, as VTK does
    nxt = np.arange(1, len(conn) + 1)
    nxt[off[1:] - 1] = off[:-1]
    (x, y), (x1, y1) = pts[conn].T, pts[conn[nxt]].T
    w = x * y1 - x1 * y
    poly = n > 4
    area6 = 3 * np.add.reduceat(w, off[:-1])
    for k, (a, b) in enumerate(((x, x1), (y, y1))):
        centres[poly, k] = (np.add.reduceat((a + b) * w, off[:-1]) / area6)[poly]

    data, i = {}, at[b"CELL_DATA"] + 2  # past "CELL_DATA n"
    while i < len(tok):
        if tok[i] == b"SCALARS":  # SCALARS name type LOOKUP_TABLE default
            name = tok[i + 1].decode()
            data[name] = nums(i + 5, ncell)
            i += 5 + ncell
        elif tok[i] == b"VECTORS":
            name = tok[i + 1].decode()
            data[name] = nums(i + 3, 3 * ncell).reshape(-1, 3)
            i += 3 + 3 * ncell
        elif tok[i] == b"FIELD":  # FIELD FieldData n, then n x (name nc nt type values)
            i += 3
        else:  # array in a FIELD block: name ncomp ntuples type values
            nc, nt = int(tok[i + 1]), int(tok[i + 2])
            data[tok[i].decode()] = nums(i + 4, nc * nt).reshape(nt, nc).squeeze(1)
            i += 4 + nc * nt
    return centres, data
