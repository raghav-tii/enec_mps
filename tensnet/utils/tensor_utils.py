import pickle

import torch
from scipy import sparse


def save_object(obj, filename):
    try:
        with open(filename, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)


def load_object(filename):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as ex:
        print("Error during unpickling object (Possibly unsupported):", ex)


def _svd_phase_corrected_torch(A: torch.Tensor, full_matrices: bool = False):
    """Torch-native equivalent of ``svd_phase_corrected`` (no CPU/numpy round-trip).

    Mirrors the numpy implementation's phase convention exactly, but stays on
    ``A``'s device/dtype throughout so SVD-heavy call sites aren't forced to
    transfer to CPU.
    """
    U, S, Vh = torch.linalg.svd(A, full_matrices=full_matrices)

    if not torch.is_complex(A):
        return U, S, Vh

    idx = torch.abs(U).argmax(dim=0)
    phases = torch.angle(U[idx, torch.arange(U.shape[1], device=U.device)])
    corr = torch.exp(-1j * phases)

    U = U * corr.reshape(1, -1)
    Vh = Vh * corr.reshape(-1, 1).conj()

    return U, S, Vh


_LOWRANK_THRESHOLD = 100


def svd(M: torch.Tensor, eps: float = None, bdmax: int = None, renormalize: bool = False):
    """Truncated SVD of ``M``.

    Args:
        M: matrix to decompose.
        eps: maximum relative Frobenius norm of the DISCARDED part, i.e. the
            returned ``(U, S, V)`` are chosen so that
                ``norm(M - U @ diag(S) @ V) <= eps * norm(M)``.
            This is a LINEAR quantity: ``eps=1e-6`` keeps singular values
            until the discarded ones' own vector 2-norm is at most 1e-6
            relative to the full singular-value vector's norm -- the same
            convention as the accuracy bound in the TT-SVD literature
            (Oseledets, "Tensor-Train Decomposition", 2011). It is NOT the
            fraction of squared Frobenius norm discarded (that quantity is
            ``eps**2``); passing what you intend as a squared-norm tolerance
            here will truncate far less aggressively than expected. ``None``
            disables eps-based truncation (only ``bdmax`` applies, if given).
        bdmax: maximum number of singular values/vectors to keep, applied
            after the eps-based selection (the smaller of the two wins).
        renormalize: rescale the kept singular values so their norm equals
            the pre-truncation norm (preserves overall state norm through
            truncation; does not change which/how many values are kept).

    Numerical note -- why the tail is summed, not subtracted: the kept rank
    is chosen from the DISCARDED tail's squared Frobenius norm, computed via
    a REVERSE cumulative sum over ``S**2`` (accumulated from the smallest
    singular values upward) and compared directly against
    ``eps**2 * total_squared_norm``. An implementation that instead computes
    a forward ``cumsum(S**2)`` and compares it against
    ``(1 - eps) * total`` (subtracting from the total, as this function did
    before) is numerically unstable for exactly the tails this function
    exists to detect: once the discarded tail's squared-norm contribution is
    smaller than ``total * float64_eps`` (~2.2e-16 * total), the forward
    cumulative sum becomes bit-identical to the total in floating point, AND
    separately, ``1 - eps`` itself rounds to exactly ``1.0`` for any ``eps``
    below ~1e-16 -- either failure alone silently caps achievable accuracy
    at that floor regardless of how tight an ``eps`` is requested (verified:
    with the old subtract-from-total formulation, tightening ``eps`` from
    1e-16 down to 0 changed nothing, while a matrix with singular values
    spanning ~13 orders of magnitude still lost ~1e-9 relative accuracy no
    matter how small `eps` was set). Summing the tail directly -- always
    adding small numbers to other small numbers, never subtracting a small
    quantity from a much larger one -- avoids both failure modes and lets
    ``eps`` be honored down to the true precision of the singular values
    themselves.
    """
    min_dim = min(M.shape[-2], M.shape[-1])

    # Fast path: when bdmax is set and much smaller than the matrix, use
    # torch.svd_lowrank to compute only the top-q triplets — O(m·n·q)
    # instead of O(m·n·min(m,n)) for the full SVD.
    use_lowrank = (
        bdmax is not None
        and eps is None
        and min_dim >= _LOWRANK_THRESHOLD
        and bdmax < min_dim // 2
        and not torch.is_complex(M)
    )

    if use_lowrank:
        q = min(bdmax + 10, min_dim)
        U, S, V = torch.svd_lowrank(M, q=q)
        V = V.mT
        bd = min(len(S), bdmax)
        if renormalize:
            initial_norm = torch.linalg.norm(S)
            trunc_norm = torch.linalg.norm(S[:bd])
            S = S.clone()
            S[:bd] = S[:bd] * (initial_norm / trunc_norm)
        return U[:, :bd], S[:bd], V[:bd, :]

    device = M.device
    U, S, V = _svd_phase_corrected_torch(M, full_matrices=False)

    # Norm before truncation, used to optionally rescale the kept singular
    # values so that truncation preserves the state norm.
    initial_norm = torch.linalg.norm(S)

    def _truncate(bd):
        if renormalize:
            trunc_norm = torch.linalg.norm(S[:bd])
            S[:bd] = S[:bd] * (initial_norm / trunc_norm)
        return U[:, :bd].to(device), S[:bd].to(device), V[:bd, :].to(device)

    if eps is None:
        if bdmax is None:
            return _truncate(len(S))
        else:
            return _truncate(min(len(S), bdmax))
    else:
        with torch.no_grad():
            S_n = S**2
            tot = torch.sum(S_n)
            # tail[k] = sum(S_n[k:]): the squared Frobenius norm of the part
            # discarded if only the first k singular values are kept. Summed
            # from the smallest singular values upward (reverse cumulative
            # sum) so every partial sum combines comparably tiny terms —
            # never a near-cancelling subtraction from `tot` (see docstring).
            tail = torch.flip(torch.cumsum(torch.flip(S_n, dims=[0]), dim=0), dims=[0])
            tail = torch.cat([tail, torch.zeros(1, dtype=S_n.dtype, device=S_n.device)])
            # Linear eps: discarded tail's Frobenius norm <= eps * ||M||,
            # i.e. squared tail <= eps**2 * squared total. `eps` is squared
            # here, once, explicitly — never folded into a `(1 - eps)` term.
            below_tol = tail <= (eps**2) * tot
            trunc = torch.nonzero(below_tol)[0].item() if torch.any(below_tol) else len(S)
            trunc = max(1, trunc)
        if bdmax is None:
            return _truncate(trunc)
        else:
            return _truncate(min(trunc, bdmax))


# # @torch.compile(dynamic=True)
# def svd(M: torch.Tensor, eps: float = None, bdmax: int = None):
#     # try:
#     U, S, V = torch.linalg.svd(M, full_matrices=False)
#     # except torch._C._LinAlgError:
#     #     U, S, V = torch.linalg.svd(M, full_matrices=False)
#     # finally:
#     #     print("SVD failed.")

#     if eps is None:
#         if bdmax is None:
#             bd = len(S)
#             return U[:, :bd], S[:bd], V[:bd, :]

#         else:
#             bd = min(len(S), bdmax)
#             return U[:, :bd], S[:bd], V[:bd, :]

#     else:
#         with torch.no_grad():
#             S_n = torch.flip(S**2, (0,))
#             trunc = sum(torch.sqrt(torch.cumsum(S_n, dim=0)) >= torch.norm(S) * eps)
#             # S_n = S / S.sum()
#             # trunc = sum(S_n > eps)
#             trunc = max(1, trunc)
#         if bdmax is None:
#             bd = trunc
#             return U[:, :bd], S[:bd], V[:bd, :]

#         else:
#             bd = min(trunc, bdmax)
#             return U[:, :bd], S[:bd], V[:bd, :]


# @torch.compile(dynamic=True)
def qr(tensor):
    return torch.linalg.qr(tensor)


def rq(tensor):
    # Q, R = qr(tensor.permute(1, 0))

    # Q1 = Q.permute(1, 0)
    # R1 = R.permute(1, 0)
    device = tensor.device
    dtype = tensor.dtype
    Q, R = qr((torch.flip(torch.eye(tensor.shape[0], device=device, dtype=dtype), (0,)) @ tensor).T)
    Q1 = torch.flip(torch.eye(Q.T.shape[0], device=device, dtype=dtype), (0,)) @ Q.T
    R1 = (
        torch.flip(torch.eye(R.T.shape[0], device=device, dtype=dtype), (0,))
        @ R.T
        @ torch.flip(torch.eye(R.T.shape[1], device=device, dtype=dtype), (0,))
    )
    return R1, Q1


# @torch.compile(dynamic=True)
def conjgrad(A, b, x=None):
    """
    A function to solve [A]{x} = {b} linear equation system with the
    conjugate gradient method.
    More at: http://en.wikipedia.org/wiki/Conjugate_gradient_method
    ========== Parameters ==========
    A : matrix
        A real symmetric positive definite matrix.
    b : vector
        The right hand side (RHS) vector of the system.
    x : vector
        The starting guess for the solution. Defaults to all ones.
    """
    if x is None:
        x = torch.zeros_like(b)
    r = b - A @ x
    p = torch.clone(r)
    rsold = torch.linalg.norm(r)

    for _i in range(len(b)):
        Ap = A @ p

        alpha = rsold**2 / (p @ Ap)

        x = x + alpha * p
        r = r - alpha * Ap

        rsnew = torch.linalg.norm(r)

        if rsnew < 1e-8:
            break

        p = r + p * (rsnew / rsold) ** 2

        rsold = torch.clone(rsnew)

    return x


# Below this size GMRES is not worth its overhead and we solve densely instead.
GMRES_MIN_SIZE = 300


def gmres_solver(A, b, precond=None):
    """Solve ``A x = b`` with GMRES.

    ``precond`` — optional callable applying ``M^{-1}`` to a torch vector, used
    as GMRES's *left* preconditioner.  When given it displaces the incomplete-LU
    preconditioner this routine otherwise builds for itself; a problem-specific
    ``M^{-1}`` beats one rediscovered from the matrix sparsity.  It is passed
    through as ``M=`` rather than used to form ``M^{-1}A``, which keeps the
    Arnoldi basis well conditioned.
    """
    if len(b) <= GMRES_MIN_SIZE:
        return direct_solver(A, b)

    dtype = b.dtype
    A_np = A.numpy()
    b_np = b.numpy()

    if precond is None:
        sA = sparse.csc_matrix(A_np)
        sA_iLU = sparse.linalg.spilu(sA)
        M = sparse.linalg.LinearOperator(A_np.shape, sA_iLU.solve)
    else:

        def _apply(v):
            return precond(torch.as_tensor(v, dtype=dtype)).detach().numpy()

        M = sparse.linalg.LinearOperator(A_np.shape, _apply, dtype=A_np.dtype)

    x, info = sparse.linalg.gmres(A_np, b_np, M=M)
    if info > 0:
        print(info)
    return torch.tensor(x)


try:  # torch exposes the linalg error type at different paths across versions
    _LINALG_ERR = (RuntimeError, torch._C._LinAlgError)
except AttributeError:  # pragma: no cover
    _LINALG_ERR = (RuntimeError,)

# Relative-residual threshold above which we distrust ``solve`` and fall back to
# the (slower but rank-revealing) least-squares solution.  Well-conditioned
# systems solve to ~1e-14; this only trips on (near-)singular matrices where
# ``solve`` returns finite-but-meaningless output without raising.
_DIRECT_SOLVE_RESIDUAL_FALLBACK = 1e-6


# @torch.compile(dynamic=True)
def direct_solver(A, b):
    """
    Solve [A]{x} = {b} with the built-in torch method.

    Uses ``torch.linalg.solve`` for square systems (markedly faster than
    ``lstsq`` — ~2x on the small dense local solves of DMRG sweeps) and falls
    back to ``torch.linalg.lstsq`` (the previous behaviour) whenever ``solve``
    is inapplicable or numerically unsafe:

      * non-square / non-2D input                          -> lstsq
      * ``solve`` raises ``LinAlgError`` (e.g. zero pivot) -> lstsq
      * solution is non-finite                             -> lstsq
      * relative residual ``||A x - b|| / ||b||`` exceeds a small threshold
        (catches singular matrices where ``solve`` silently returns garbage,
        e.g. duplicate columns)                            -> lstsq

    The residual guard costs one matvec (O(n^2)) — negligible next to the
    O(n^3) solve.

    ========== Parameters ==========
    A : matrix
    b : vector
    """

    def _lstsq_fallback():
        # CUDA's torch.linalg.lstsq only supports the full-rank "gels" driver,
        # so it silently returns garbage on the singular/rank-deficient
        # matrices this fallback exists for. Run it on CPU instead (rare path,
        # transfer cost doesn't matter) so the result matches CPU exactly.
        if A.is_cuda:
            return torch.linalg.lstsq(A.cpu(), b.cpu()).solution.to(A.device)
        return torch.linalg.lstsq(A, b).solution

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return _lstsq_fallback()
    try:
        x = torch.linalg.solve(A, b)
    except _LINALG_ERR:
        return _lstsq_fallback()
    if not torch.isfinite(x).all():
        return _lstsq_fallback()

    b_norm = torch.linalg.norm(b)
    if b_norm > 0:
        rel_res = torch.linalg.norm(A @ x - b) / b_norm
        if not torch.isfinite(rel_res) or rel_res > _DIRECT_SOLVE_RESIDUAL_FALLBACK:
            return _lstsq_fallback()
    return x


def eigen_solver(A, k=1, criteria="abs", descending=True):
    """
    Compute up to k eigenvalues and eigenvectors of A, sorted by a criterion.

    Args:
        A: (n, n) real or complex tensor (square matrix).
        k: number of leading eigenpairs to return after sorting.
        criteria: sorting key for eigenvalues: "real", "imag", or "abs" (magnitude).
        descending: if True, sort from largest to smallest; otherwise ascending.

    Returns:
        evals: (k,) complex tensor of sorted eigenvalues.
        evecs: (n, k) complex tensor whose columns are the corresponding eigenvectors.
    """
    evals, evecs = torch.linalg.eig(A)

    if criteria == "real":
        sort_key = evals.real
    elif criteria == "imag":
        sort_key = evals.imag
    elif criteria == "abs":
        sort_key = evals.abs()

    idx = torch.argsort(sort_key, descending=descending)

    evals_sorted = evals[idx]
    evecs_sorted = evecs[:, idx]

    return evals_sorted[:k], evecs_sorted[:, :k]


def nested_list_to_tensor(nested_list, device=None):
    if isinstance(nested_list, list):
        return torch.stack([nested_list_to_tensor(item, device) for item in nested_list])
    elif torch.is_tensor(nested_list):
        return nested_list.to(device) if device is not None else torch.clone(nested_list)
    else:
        return torch.tensor(nested_list, device=device)


def validate_nodes(nodes):
    if nodes[0].boundary is None:
        nodes[0].boundary = "left"
    if nodes[-1].boundary is None:
        nodes[-1].boundary = "right"
    for node in nodes[1:-1]:
        node.boundary = None
    return nodes


def val_addability(self, other, p_axis=1):
    assert self.order == other.order, (
        "Only tensors of same order can be added."
        + f"Received {self.order} and {other.order} -order tensors"
    )
    assert self.dim[p_axis] == other.dim[p_axis], (
        "Failed to add coz of mismatched shape on the physical leg(s)"
    )

    return (self.order == other.order) and self.dim[p_axis] == other.dim[p_axis]


def get_local_inds(int_inds, nofq, base=2):
    digits = []
    num = int_inds if isinstance(int_inds, int) else int_inds.item()
    if not 0 <= num < base**nofq:  # out of range would never reach nofq digits and loop forever
        raise ValueError(
            f"index {num} out of range for {nofq} sites of dimension {base}: need 0 <= index < {base**nofq}"
        )
    while num:
        digits.append(int(num % base))
        num //= base
    while len(digits) != nofq:
        digits.append(0)
    return digits[::-1]


def unshuffle_axis(A: torch.Tensor, L: int, d: int):
    # tensorize A and unshuffle odd and even axis and return a matrix
    axis_even_odd = [x for x in range(2 * L) if x % 2 == 0]
    axis_even_odd += [x for x in range(2 * L) if x % 2 == 1]
    A = A.reshape([d] * 2 * L).permute(axis_even_odd).reshape(d**L, d**L)
    return A


def shuffle_axis(A: torch.Tensor, L: int):
    # tensorize A and shuffle odd and even axis and return a matrix
    new_axis = []
    for x in range(L):
        new_axis += [x, x + L]
    A = A.reshape([2] * 2 * L).permute(new_axis).reshape(2**L, 2**L)
    return A


# Network Contraction Utils
