"""Finite-difference MPOs on explicit grids, with high-order SBP-SAT wall closures.

Each axis has n = 2**nofq points on [a, b]: "periodic" (h = (b-a)/n), "vertex"
(h = (b-a)/(n-1), nodes on both walls) or "cell" (h = (b-a)/n, walls half a
cell out). Interior stencils have accuracy 2, 4, 6 or 8; wall rows use
diagonal-norm summation-by-parts closures with boundary conditions imposed
weakly, so ``A u + b`` approximates the derivative of a u satisfying them
(A = diff_mpo, b = bc_mps, H = quadrature_mps). Coefficients are in
_wall_closure_tables.py. Nothing here reads global
configuration except ``arch`` when not given, the operators include the grid
spacing, and every builder recompresses to the exact rank (rtol 1e-13).
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from ..networks import MPO, MPS, list_to_mps, matrix_to_mpo, mps_to_mpo, vector_to_mps
from ..utils import ARCH_2D
from ..utils.config import device as default_device
from . import _wall_closure_tables
from .difference_mpo import _add_exact, _toeplitz, zeros_padding

__all__ = [
    "Dirichlet",
    "Neumann",
    "GRIDS",
    "grid_spacing",
    "grid_points",
    "diff_mpo",
    "bc_mps",
    "quadrature_mps",
    "mpo_sum",
    "spectral_radius_bound",
]

GRIDS = ("periodic", "vertex", "cell")


@dataclass(frozen=True)
class Dirichlet:
    """u = value at the wall. `value` is a float, or on a multi-D grid an MPS
    over the other axes (in the arch's site order) giving the wall data."""

    value: float | MPS = 0.0


@dataclass(frozen=True)
class Neumann:
    """du/dx = value at the wall, x being the axis coordinate (not the outward
    normal). `value` is a float or, on a multi-D grid, an MPS over the other axes."""

    value: float | MPS = 0.0


# ---------------------------------------------------------------- grids


def grid_spacing(nofq: int, bounds=(0.0, 1.0), grid: str = "vertex") -> float:
    """Spacing h of an axis with 2**nofq points on [a, b].

    Parameters
    ----------
    nofq : int
        Number of qubits of the axis.
    bounds : (float, float)
        Interval [a, b].
    grid : {"periodic", "vertex", "cell"}

    Returns
    -------
    float
    """
    _check_grid(grid)
    n = 2**nofq
    return (bounds[1] - bounds[0]) / (n - 1 if grid == "vertex" else n)


def grid_points(
    nofq: int, bounds=(0.0, 1.0), grid: str = "vertex", dtype=torch.float64, device=None
) -> torch.Tensor:
    """Coordinates x_j of an axis, in the order the MPS encodes them.

    Parameters
    ----------
    nofq, bounds, grid :
        As in grid_spacing.

    Returns
    -------
    torch.Tensor
        Shape (2**nofq,).
    """
    h = grid_spacing(nofq, bounds, grid)
    j = torch.arange(
        2**nofq, dtype=dtype, device=device if device is not None else default_device()
    )
    return bounds[0] + (j + (0.5 if grid == "cell" else 0.0)) * h


# ---------------------------------------------------------------- public builders


def diff_mpo(
    order: int,
    nofq: int | Sequence[int],
    axis: int = 0,
    *,
    bounds=(0.0, 1.0),
    grid: str | Sequence[str] = "vertex",
    accuracy: int = 8,
    bc=None,
    arch: str | None = None,
    weighted: bool = False,
    dtype=torch.float64,
    device=None,
) -> MPO:
    """MPO of d^order/dx^order along `axis`, scaled by the grid spacing.

    With ``weighted=True`` the result is H A, H = diag(quadrature_mps(...)) for
    the same nofq, bounds, grid, accuracy and arch. H A is symmetric negative
    semidefinite for order 2 with Dirichlet or Neumann ends (definite with a
    Dirichlet end), so a symmetric solver applies to the weighted system:

        HA = tt.mpo_sum(*(tt.diff_mpo(2, nofq, a, weighted=True, **kw_a) for a in axes))
        Hb = sum of tt.bc_mps(2, nofq, a, weighted=True, **kw_a)
        Hf = tt.quadrature_mps(nofq, **kw) * f        # MPS Hadamard product
        u, lognorm = tt.linsys_dmrg2(HA * -1.0, (Hf - Hb) * -1.0, x0, ...)  # -HA is SPD with a Dirichlet end

    (MPS +, - and * compress to the global set_accuracy; tighten it for a
    full-accuracy right-hand side.) A itself is not symmetric, and a Galerkin
    solve of A u = f - b can stagnate. Any order and bc are accepted; H A is
    symmetric only as above.

    Parameters
    ----------
    order : {1, 2}
    nofq : int or sequence of int
        Qubits per axis; an int is a 1D grid.
    axis : int
        Axis to differentiate.
    bounds : (a, b) or sequence of (a, b)
        Domain per axis (one pair is used for every axis).
    grid : str or sequence of str
        "periodic", "vertex" or "cell", per axis.
    accuracy : {2, 4, 6, 8}
        Order of the interior stencil; wall closures have order accuracy / 2.
    bc : None, or (left, right)
        Boundary conditions on `axis` for a wall grid; each end is None (no
        condition: one-sided closure only), Dirichlet, Neumann, or the
        strings "dirichlet" / "neumann" (value 0). A single condition applies
        to both ends. Order 1 accepts Dirichlet only, read as an inflow
        condition: the operator then penalises u toward the value at that end
        (stable for advection entering there). Must be None on a periodic axis.
    arch : {"stacked", "sequential"}, optional
        Site order of multi-D grids; defaults to tensnet's global arch.
    weighted : bool
        Return H A instead of A.
    dtype, device :
        Of the MPO tensors.

    Returns
    -------
    MPO
        A (or H A), the homogeneous part; bc_mps gives the data part b (or H b),
        compressed to its exact rank.
    """
    nofq, bounds, grid, arch = _axes(nofq, bounds, grid, arch)
    _check(order, accuracy, axis, nofq, grid)
    left, right = _bc_pair(bc, grid[axis], order)
    device = device if device is not None else default_device()
    h = grid_spacing(nofq[axis], bounds[axis], grid[axis])
    P = _operator_1d(order, nofq[axis], grid[axis], accuracy, left, right, weighted, dtype, device)
    P = _compress(P * h ** (int(weighted) - order))
    if (
        len(nofq) > 1
    ):  # H A = (H_a A_a) x diag(H_b) on the other axes b; exact-rank factors, so this only regauges
        P = _compress(
            _embed(
                P,
                axis,
                nofq,
                arch,
                {
                    b: _weights(nofq, bounds, grid, accuracy, [b], arch, weighted, dtype, device)
                    for b in range(len(nofq))
                    if b != axis
                },
            )
        )
    return P


def bc_mps(
    order: int,
    nofq: int | Sequence[int],
    axis: int = 0,
    *,
    bounds=(0.0, 1.0),
    grid: str | Sequence[str] = "vertex",
    accuracy: int = 8,
    bc=None,
    arch: str | None = None,
    weighted: bool = False,
    dtype=torch.float64,
    device=None,
) -> MPS:
    """Boundary-data term b: d^order u/dx^order ~ diff_mpo(...) u + bc_mps(...).

    Parameters
    ----------
    order, nofq, axis, bounds, grid, accuracy, bc, arch, weighted, dtype, device :
        As in diff_mpo; pass the same values (weighted=True gives H b). Ends
        without a condition, or with value 0, contribute nothing; the result is
        the zero MPS when all do. An MPS value must have one site of physical
        dimension 2 per qubit of the other axes; a real one is cast to dtype and
        device. Complex values need a complex dtype.

    Returns
    -------
    MPS
        Compressed to its exact rank.
    """
    nofq, bounds, grid, arch = _axes(nofq, bounds, grid, arch)
    _check(order, accuracy, axis, nofq, grid)
    ends = _bc_pair(bc, grid[axis], order)
    device = device if device is not None else default_device()
    h = grid_spacing(nofq[axis], bounds[axis], grid[axis])
    others = [a for a in range(len(nofq)) if a != axis]
    if others:  # what multiplies the wall vector on the other axes: their weights, or ones
        rest = _weights(nofq, bounds, grid, accuracy, others, arch, weighted, dtype, device)
    total = None
    for side, cond in enumerate(ends):
        if cond is None:
            continue
        value = _face(cond.value, sum(nofq[b] for b in others), dtype, device)
        if not isinstance(value, MPS) and value == 0:
            continue
        v = _data_vector(order, grid[axis], accuracy, side, cond, weighted)
        v = v * h ** (int(weighted) - order + (1 if isinstance(cond, Neumann) else 0))
        vec = _corner_mps(v, nofq[axis], side, dtype, device)
        if isinstance(value, MPS):
            face = _hadamard(value, rest) if weighted else value
        else:
            vec = _scale(vec, value)
            face = rest if others else None
        part = vec if not others else _merge([((axis,), vec), (tuple(others), face)], nofq, arch)
        total = part if total is None else _mps_add(total, part)
    if total is None:
        total = _scale(_ones(nofq, range(len(nofq)), arch, dtype, device), 0.0)
    return _compress(total)


def quadrature_mps(
    nofq: int | Sequence[int],
    *,
    bounds=(0.0, 1.0),
    grid: str | Sequence[str] = "vertex",
    accuracy: int = 8,
    arch: str | None = None,
    dtype=torch.float64,
    device=None,
) -> MPS:
    """Quadrature weights w (the SBP norm H times the spacing): sum(w * f) ~ integral of f.

    Exact for polynomials of degree < accuracy along wall axes and for
    trigonometric polynomials along periodic axes. It is the H of the
    summation-by-parts identities, so it is also the inner product to use
    for energy estimates and for symmetric solvers.

    Returns
    -------
    MPS
        Compressed to its exact rank.
    """
    nofq, bounds, grid, arch = _axes(nofq, bounds, grid, arch)
    _check(1, accuracy, 0, nofq, grid)
    device = device if device is not None else default_device()
    parts = []
    for a in range(len(nofq)):
        h = grid_spacing(nofq[a], bounds[a], grid[a])
        w = _scale(_ones(nofq[a : a + 1], [0], arch, dtype, device), h)
        if grid[a] != "periodic":
            hb = np.asarray(_table(grid[a], accuracy)["h"]) - 1.0
            v = np.zeros(_block(grid[a], accuracy))
            v[: len(hb)] = hb * h
            for side in (0, 1):
                w = _mps_add(
                    w, _corner_mps(v if side == 0 else v[::-1].copy(), nofq[a], side, dtype, device)
                )
        parts.append(((a,), w))
    return _compress(parts[0][1] if len(parts) == 1 else _merge(parts, nofq, arch))


def spectral_radius_bound(
    order: int, nofq: int, *, bounds=(0.0, 1.0), grid: str = "vertex", accuracy: int = 8, bc=None
) -> float:
    """Gershgorin bound (largest absolute row sum) on the spectral radius of the 1D d^order/dx^order.

    Read off the stencil and the wall-closure block, so no matrix is formed and
    it holds at any nofq. It overestimates the true radius, by roughly 2x for
    the second derivative on a periodic axis. An explicit step of size dt is
    safe only if dt * bound is inside the time-stepper's stability region
    (about 2.8 on the negative real axis for RK4); closure rows on high-order
    wall grids raise the bound well above the interior's.

    Parameters are those of diff_mpo for one axis; `bc` is a pair as there.
    """
    _check_grid(grid)
    _check(order, accuracy, 0, [nofq], [grid])
    a = _interior(order, accuracy)
    rows = [sum(abs(v) for v in a.values())]
    if grid != "periodic":
        W = _block(grid, accuracy)
        for _side, kind in enumerate(_bc_pair(bc, grid, order)):
            C = np.abs(_left_operator(order, grid, accuracy, kind))
            for i in range(W):  # band entries past the block's last column
                rows.append(C[i].sum() + sum(abs(v) for k, v in a.items() if i + k >= W))
    return max(rows) * grid_spacing(nofq, bounds, grid) ** -order


def mpo_sum(*mpos: MPO) -> MPO:
    """Sum of MPOs on the same sites, exact up to round-off, compressed to its exact rank.

    The bonds are added as a direct sum and the result is recompressed with
    a fixed relative cutoff (no global state). MPO.__add__ instead truncates
    to the global set_accuracy / set_bd_limit, which can corrupt an operator
    (a bond limit below its rank changes it); use this to assemble operators
    such as a Laplacian.

    Parameters
    ----------
    *mpos : MPO
        MPOs with matching site counts and physical dimensions.

    Returns
    -------
    MPO
    """
    if len(mpos) < 1 or len({len(m) for m in mpos}) != 1:
        raise ValueError("mpo_sum needs MPOs with the same number of sites")
    total = mpos[0]
    for m in mpos[1:]:
        total = _add_exact(total, m)
    return _compress(total)


# ---------------------------------------------------------------- 1D construction


def _table(grid, accuracy):
    return _wall_closure_tables.TABLES[(grid, accuracy)]


def _interior(order, accuracy):
    return _wall_closure_tables.INTERIOR[(order, accuracy)]


def _block(grid, accuracy) -> int:
    """Width (a power of two) of the dense corner block that holds a wall closure."""
    T = _table(grid, accuracy)
    width = max(len(T["M"]), len(T["D1"])) + accuracy // 2
    return 2 ** max(1, math.ceil(math.log2(width)))


def _left_operator(order, grid, accuracy, kind, weighted=False) -> np.ndarray:
    """Left corner (W x W, grid units) of the 1D operator A, or of H A, `kind` the left condition."""
    T, W = _table(grid, accuracy), _block(grid, accuracy)
    p = accuracy // 2
    big = W + p + 1
    a = _interior(order, accuracy)
    hd = np.ones(big)
    hd[: len(T["h"])] = T["h"]
    tw = T["t"] if order == 1 else T["t2"]
    t = np.zeros(big)
    t[: len(tw)] = tw
    if order == 1:
        Q = np.zeros((big, big))
        for i in range(big):
            for k, v in a.items():
                if 0 <= i + k < big:
                    Q[i, i + k] = v
        r = len(T["D1"])
        Q[:r, :] = 0.0
        Q[:r, : len(T["D1"][0])] = np.asarray(T["D1"]) * np.asarray(T["h"])[:r, None]
        HA = Q
        if isinstance(kind, Dirichlet):
            HA = HA + np.outer(t, t)
    else:
        M = np.zeros((big, big))
        for i in range(big):
            for k, v in a.items():
                if 0 <= i + k < big:
                    M[i, i + k] = -v
        r2 = len(T["M"])
        M[:r2, :r2] = T["M"]
        d = np.zeros(big)
        d[: len(T["d"])] = T["d"]
        HA = -M
        if not isinstance(kind, Neumann):
            HA = HA - np.outer(t, d)
        if isinstance(kind, Dirichlet):
            HA = HA - np.outer(d, t) + T["tau"] * np.outer(t, t)
    return (HA if weighted else HA / hd[:, None])[:W, :W]


def _data_vector(order, grid, accuracy, side, cond, weighted=False) -> np.ndarray:
    """Per-unit-value data term of b, or of H b (grid units, length W) at `side`, left-to-right order."""
    T, W = _table(grid, accuracy), _block(grid, accuracy)
    hd = np.ones(W)
    hd[: len(T["h"])] = T["h"]
    tw = T["t"] if order == 1 else T["t2"]
    t = np.zeros(W)
    t[: len(tw)] = tw
    if order == 1 or isinstance(cond, Neumann):
        v, sign = -t, -1.0
    else:
        d = np.zeros(W)
        d[: len(T["d"])] = T["d"]
        v, sign = d - T["tau"] * t, 1.0
    if not weighted:
        v = v / hd
    return v if side == 0 else sign * v[::-1].copy()


def _operator_1d(order, nofq, grid, accuracy, left, right, weighted, dtype, device) -> MPO:
    """Uncompressed 1D A (or H A) in grid units: the band plus the two corner corrections."""
    a = _interior(order, accuracy)
    if grid == "periodic":
        return _toeplitz(nofq, a, ["periodic"] * 2, dtype=dtype, device=device)
    W = _block(grid, accuracy)
    band = np.zeros((W, W))
    for i in range(W):
        for k, v in a.items():
            if 0 <= i + k < W:
                band[i, i + k] = v
    P = _toeplitz(nofq, a, ["dirichlet"] * 2, dtype=dtype, device=device)
    q = int(math.log2(W))
    for side, kind in ((0, left), (1, right)):
        C = _left_operator(order, grid, accuracy, kind, weighted) - band
        if side == 1:
            C = (-1.0) ** order * C[::-1, ::-1]
        corner = matrix_to_mpo(torch.tensor(C.copy(), dtype=dtype, device=device), 2, q)
        P = _add_exact(P, zeros_padding(corner, nofq, "ul" if side == 0 else "lr"))
    return P


# ---------------------------------------------------------------- MPS / MPO plumbing


def _weights(nofq, bounds, grid, accuracy, axes, arch, weighted, dtype, device) -> MPS:
    """What multiplies a 1D factor on the other `axes`: their quadrature weights if weighted, else ones."""
    if not weighted:
        return _ones(nofq, axes, arch, dtype, device)
    return quadrature_mps(
        [nofq[b] for b in axes],
        bounds=[bounds[b] for b in axes],
        grid=[grid[b] for b in axes],
        accuracy=accuracy,
        arch=arch,
        dtype=dtype,
        device=device,
    )


def _corner_mps(v: np.ndarray, nofq: int, side: int, dtype, device) -> MPS:
    """MPS of the length-2**nofq vector holding v in its first (side 0) or last W entries."""
    q = int(math.log2(len(v)))
    low = vector_to_mps(torch.tensor(np.asarray(v), dtype=dtype, device=device), 2, q)
    e = torch.zeros(1, 2, 1, dtype=dtype, device=device)
    e[0, side, 0] = 1.0
    return list_to_mps([e] * (nofq - q) + [node.tensor for node in low])


def _scale(mps: MPS, c: float) -> MPS:
    return list_to_mps([node.tensor for node in mps[:-1]] + [mps[-1].tensor * c])


def _mps_add(A: MPS, B: MPS) -> MPS:
    """A + B as an uncompressed direct sum."""
    return MPS([p + q for p, q in zip(A, B)])


def _ones(nofq, axes, arch, dtype, device) -> MPS:
    L = sum(nofq[a] for a in axes)
    return list_to_mps([torch.ones(1, 2, 1, dtype=dtype, device=device)] * L)


def _site_order(nofq, arch):
    """(axis, level) of every site, in MPS order."""
    if len(nofq) == 1 or arch == "stacked":
        return [(a, k) for a in range(len(nofq)) for k in range(nofq[a])]
    return [(a, k) for k in range(nofq[0]) for a in range(len(nofq))]


def _merge(parts, nofq, arch) -> MPS:
    """Product of MPS defined on disjoint sets of axes, laid out in the arch's site order."""
    ptr, bonds, nodes = [0] * len(parts), [1] * len(parts), []
    for axis, _ in _site_order(nofq, arch):
        p = next(i for i, (axes, _) in enumerate(parts) if axis in axes)
        T = parts[p][1][ptr[p]].tensor
        ptr[p] += 1
        pre, post = math.prod(bonds[:p]), math.prod(bonds[p + 1 :])

        def eye(m):
            return torch.eye(m, dtype=T.dtype, device=T.device)

        X = torch.einsum("ab,lsr,cd->alcsbrd", eye(pre), T, eye(post))
        nodes.append(X.reshape(pre * T.shape[0] * post, T.shape[1], pre * T.shape[2] * post))
        bonds[p] = T.shape[2]
    return list_to_mps(nodes)


def _diag(w: MPS) -> MPS:
    """diag(w) as an MPO, in the MPS form of physical dimension 4 that _merge takes."""
    eye = torch.eye(2, dtype=w[0].tensor.dtype, device=w[0].tensor.device)
    return list_to_mps(
        [torch.einsum("lsr,st->lstr", n.tensor, eye).reshape(n.shape[0], 4, n.shape[2]) for n in w]
    )


def _embed(P: MPO, axis: int, nofq, arch, diags) -> MPO:
    """Act with the 1D MPO P on `axis` and as diag(diags[b]) on every other axis b
    (an MPS over that axis); the bonds of all axes are carried in the arch's site order."""
    parts = [((axis,), P.mpo_to_mps())] + [((b,), _diag(w)) for b, w in diags.items()]
    return mps_to_mpo(_merge(parts, nofq, arch))


def _hadamard(A: MPS, B: MPS) -> MPS:
    """Elementwise product A * B, uncompressed."""
    return list_to_mps(
        [
            torch.einsum("asb,csd->acsbd", p.tensor, q.tensor).reshape(
                p.shape[0] * q.shape[0], p.shape[1], p.shape[2] * q.shape[2]
            )
            for p, q in zip(A, B)
        ]
    )


# Exact-rank cutoff, relative to the largest singular value at the bond. The
# redundant directions of the sums built here are round-off, at most ~4e-16
# of it (measured); the smallest genuine ones are ~3e-8 at nofq 20 and shrink
# like 2**(-nofq/2) (corner blocks against the band). 1e-13 clears the first
# by ~250x and the second up to nofq ~50.
_RTOL = 1e-13  # at float64; scaled by the dtype's eps, so float32 cuts at ~5e-5
_EPS64 = torch.finfo(torch.float64).eps


def _split(X: torch.Tensor):
    """X = Q R with orthonormal columns in Q. An SVD that drops the singular
    values below _RTOL times the largest when there are any, a QR otherwise:
    QR keeps every column to its own relative precision, an SVD only to that
    of the largest."""
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    keep = int((_RTOL * max(1.0, torch.finfo(X.dtype).eps / _EPS64) * S[0] < S).sum())
    if keep == len(S):
        return torch.linalg.qr(X)
    keep = max(1, keep)
    return U[:, :keep], S[:keep, None] * Vh[:keep]


def _compress(net: MPS | MPO) -> MPS | MPO:
    """Recompress an MPS or MPO to its exact rank, deterministically, reading no global state.

    Sweeps right-to-left, left-to-right, right-to-left, splitting each bond
    with _split. The first starts from the uncompressed network, so it cuts
    the redundancy of the corner blocks (which sit on the last sites) while
    the norm it carries is still that of a few sites, not of the whole band;
    a canonical sweep would round them to eps times the full norm,
    ~eps sqrt(2**nofq) in grid units. The other two run in canonical form, so
    what they cut is relative to the bond's largest singular value. The last
    leaves the norm in the first core, so that the trailing cores are again
    of order one when a sum or product of these networks is compressed.
    """
    T = [n.tensor for n in (net.mpo_to_mps() if isinstance(net, MPO) else net)]
    for leftward in (True, False, True):
        for k in range(len(T) - 1, 0, -1) if leftward else range(len(T) - 1):
            if leftward:  # orthonormal rows in core k, the rest into core k - 1
                l, s, r = T[k].shape
                Q, R = _split(T[k].reshape(l, s * r).mT)
                T[k] = Q.mT.reshape(-1, s, r)
                T[k - 1] = torch.einsum("lsa,ba->lsb", T[k - 1], R)
            else:
                l, s, r = T[k].shape
                Q, R = _split(T[k].reshape(l * s, r))
                T[k] = Q.reshape(l, s, -1)
                T[k + 1] = torch.einsum("ab,bsr->asr", R, T[k + 1])
    out = list_to_mps(T)
    return mps_to_mpo(out) if isinstance(net, MPO) else out


# ---------------------------------------------------------------- argument handling


def _axes(nofq, bounds, grid, arch):
    nofq = [nofq] if isinstance(nofq, int) else list(nofq)
    d = len(nofq)
    if len(bounds) == 2 and not isinstance(bounds[0], (tuple, list)):
        bounds = [tuple(bounds)] * d
    grid = [grid] * d if isinstance(grid, str) else list(grid)
    if len(bounds) != d or len(grid) != d:
        raise ValueError(f"bounds and grid need one entry per axis ({d})")
    arch = ARCH_2D() if arch is None else arch
    if d > 1:
        if arch not in ("stacked", "sequential"):
            raise ValueError(f"arch must be 'stacked' or 'sequential', not {arch!r}")
        if arch == "sequential" and len(set(nofq)) != 1:
            raise ValueError("the sequential arch needs the same nofq on every axis")
    return nofq, bounds, grid, arch


def _check_grid(grid):
    if grid not in GRIDS:
        raise ValueError(f"grid must be one of {GRIDS}, not {grid!r}")


def _check(order, accuracy, axis, nofq, grid):
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, not {order!r}")
    if accuracy not in (2, 4, 6, 8):
        raise ValueError(f"accuracy must be 2, 4, 6 or 8, not {accuracy!r}")
    if not 0 <= axis < len(nofq):
        raise ValueError(f"axis {axis} out of range for {len(nofq)} axes")
    for q, g in zip(nofq, grid):
        _check_grid(g)
        need = 3 if g == "periodic" else int(math.log2(_block(g, accuracy))) + 1
        if q < need:
            raise ValueError(f"a {g} axis at accuracy {accuracy} needs nofq >= {need}, got {q}")


def _face(value, sites, dtype, device):
    """Validated boundary value: a scalar, or an MPS over the other axes (`sites`
    qubits) cast to dtype / device."""
    is_mps = isinstance(value, MPS)
    if (
        value[0].tensor.is_complex() if is_mps else np.iscomplexobj(value)
    ) and not dtype.is_complex:
        raise ValueError(
            f"complex boundary data needs a complex dtype (e.g. torch.complex128), not {dtype}"
        )
    if not is_mps:
        return value
    if not sites:
        raise ValueError("an MPS boundary value needs a multi-D grid")
    dims = [n.shape[1] for n in value]
    if len(dims) != sites or any(d != 2 for d in dims):
        raise ValueError(
            f"an MPS boundary value needs one site of physical dimension 2 per qubit of the other "
            f"axes ({sites} sites), got {len(dims)} sites of dimensions {sorted(set(dims))}"
        )
    return value.to(device=device, dtype=dtype)


def _bc_pair(bc, grid, order):
    if bc is None:
        pair = (None, None)
    elif isinstance(bc, (str, Dirichlet, Neumann)):
        pair = (bc, bc)
    else:
        pair = tuple(bc)
        if len(pair) != 2:
            raise ValueError("bc must be None, one condition, or a (left, right) pair")
    out = []
    for c in pair:
        if isinstance(c, str):
            c = {"dirichlet": Dirichlet(), "neumann": Neumann()}.get(c.lower())
            if c is None:
                raise ValueError("bc strings are 'dirichlet' or 'neumann'")
        if c is not None and not isinstance(c, (Dirichlet, Neumann)):
            raise ValueError(f"unsupported boundary condition {c!r}")
        if c is not None and grid == "periodic":
            raise ValueError("a periodic axis takes no boundary condition")
        if isinstance(c, Neumann) and order == 1:
            raise ValueError("order 1 takes Dirichlet (inflow) conditions only")
        out.append(c)
    return tuple(out)
