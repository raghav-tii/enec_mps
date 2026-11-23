import torch
from einops import einsum, rearrange

from ..networks import (
    MPO,
    MPS,
    list_to_mps,
)
from ..utils import (
    direct_solver,
    qr,
    rq,
)
from .dmrg_solver import (
    _absorb_operator_lenv,
    _absorb_operator_renv,
    _absorb_rhs_lenv,
    _absorb_rhs_renv,
    is_hermitian,
    normal_equations,
)


def _apply(A, x):
    """``A.apply(x)`` without ever holding the uncompressed product.

    ``MPO.apply`` materialises every product core, (R*r, d, R*r) -- 252 MB
    each at R = 124, r = 32 -- and only then compresses.  Here the left half
    is left-orthogonalised and the right half right-orthogonalised site by
    site with untruncated QR, so bond k never exceeds min(d**k, d**(L-k), R*r)
    and a step holds about s * R * r * d numbers.  The halves meet in the
    middle, giving the exact product, which is compressed as ``MPO.apply``
    does -- so results match it to round-off.
    """
    # ponytail: exact, so the centre bond is min(d**(L/2), R*r); past L ~ 24
    # with an incompressible product, truncate in the half-sweeps (zip-up).
    L, m = len(A), len(A) // 2
    one = torch.ones(1, 1, 1, dtype=x[0].tensor.dtype, device=x[0].tensor.device)
    left, D = [], one
    for k in range(m):
        T = einsum(D, x[k].tensor, "s a l, l i r -> s a i r")
        T = einsum(T, A[k].tensor, "s a i r, a i b o -> s o b r")
        s, o, b, r = T.shape
        Q, R = torch.linalg.qr(T.reshape(s * o, b * r))
        left.append(Q.reshape(s, o, -1))
        D = R.reshape(-1, b, r)
    right, C = [], one
    for k in range(L - 1, m - 1, -1):
        T = einsum(x[k].tensor, C, "l i r, b r s -> l i b s")
        T = einsum(A[k].tensor, T, "a i b o, l i b s -> a l o s")
        a, l, o, s = T.shape
        Q, R = torch.linalg.qr(T.reshape(a * l, o * s).mH)  # M = R^H Q^H
        right.insert(0, Q.mH.reshape(-1, o, s))
        C = R.mH.reshape(a, l, -1)
    right[0] = einsum(einsum(D, C, "s a l, a l t -> s t"), right[0], "s t, t o u -> s o u")
    return list_to_mps(left + right).compress()


def compute_env_tensors_amen(A, b, x):
    """
    Computes the environment tensors for the alternating minimization solver (AMEn).

    Args:
        A (MPO): MPO encoding of the operator.
        b (MPS): MPS encoding of the right-hand side.
        x (MPS): MPS encoding of the initial guess.

    Returns:
        tuple: Tuple containing two lists of environment tensors (lhs_env_tensors, rhs_env_tensors)
              computed for both the left and right hand side of the linear system.
    """

    # Each site is absorbed straight into the running environment (the
    # shared DMRG helpers, path-optimised env -> x -> A -> x*).  Building the
    # six-index "Al ll ul Ar lr ur" site tensor first and closing it against
    # the environment afterwards costs (R * r**2)**2 entries -- 34 GB at
    # R = 64, r = 32 -- where this order peaks at about r * R * r * d.
    lhs_env_tensors = []
    rhs_env_tensors = []
    lhs_env_tensor = rhs_env_tensor = None
    for ix in range(len(A) - 1, 0, -1):
        lhs_env_tensor = _absorb_operator_renv(A[ix].tensor, x[ix].tensor, lhs_env_tensor)
        rhs_env_tensor = _absorb_rhs_renv(b[ix].tensor, x[ix].tensor, rhs_env_tensor)
        lhs_env_tensors.insert(0, lhs_env_tensor)
        rhs_env_tensors.insert(0, rhs_env_tensor)

    return lhs_env_tensors, rhs_env_tensors






def _orthogonalize_enrichment_forward(temp_core, enrichment, pd):
    """Project enrichment columns orthogonal to temp_core's column space (QR-based)."""
    X = rearrange(temp_core, "l d r -> (l d) r")
    E = rearrange(enrichment, "l d r -> (l d) r")
    Q_x, _ = qr(X)
    E_orth = E - Q_x @ (Q_x.conj().T @ E)
    return rearrange(E_orth, "(l d) r -> l d r", d=pd)


def _orthogonalize_enrichment_reverse(temp_core, enrichment, pd):
    """Project enrichment rows orthogonal to temp_core's row space (QR-based)."""
    X_T = rearrange(temp_core, "l d r -> (d r) l")
    E_T = rearrange(enrichment, "l d r -> (d r) l")
    Q_x, _ = qr(X_T)
    E_orth_T = E_T - Q_x @ (Q_x.conj().T @ E_T)
    return rearrange(E_orth_T, "(d r) l -> l d r", d=pd)


def forward_sweep_amen(
    A,
    b,
    x,
    env_data,
    residual_rank,
    step_size,
    orthogonalize_enrichment=False,
    preconditioner=None,
):
    """
    Performs a forward sweep in the Alternating Minimal Energy (AMEn) algorithm.

    This function implements the forward sweep phase of the AMEn algorithm for solving
    linear systems in the Matrix Product State (MPS) format. It updates the cores of
    the solution MPS sequentially from left to right, incorporating residual information
    to improve accuracy.

    Args:
        A (MPO): Matrix Product Operator representing the system matrix
        b (MPS): Matrix Product State representing the right-hand side vector
        x (MPS): Current approximation of the solution in MPS format
        env_data (tuple): Tuple containing left and right environment tensors
            (lhs_env_tensors, rhs_env_tensors)
        residual_rank (int): Maximum bond dimension for the compressed residual
        step_size (float): Step size for the residual directions

    Returns:
        tuple: A tuple containing:
            - MPS: Updated solution after the forward sweep
            - tuple: Updated environment tensors for the next sweep
                (lhs_env_tensors_n, rhs_env_tensors_n)
    """
    lhs_env_tensors, rhs_env_tensors = env_data

    L = len(A)

    lhs_env_tensors_n = []
    rhs_env_tensors_n = []

    new_tensors = []

    residual = b - _apply(A, x)
    if preconditioner is not None:
        residual = _apply(preconditioner, residual)
    res_tensors = residual.tensors

    # --- Site 0 ---
    ix = 0
    lbd = 1
    pd = x[ix].shape[1]

    local_matrix = einsum(
        A[ix].tensor,
        lhs_env_tensors[ix],
        "l i r o, r lr ur -> l i o lr ur",
    )
    local_matrix = rearrange(
        local_matrix,
        "l i o lr ur -> (o ur) (l i lr)",
    )
    local_vector = einsum(
        b[ix].tensor,
        rhs_env_tensors[ix],
        " l d r, r lr-> l d lr",
    ).flatten()

    loc_sol = direct_solver(local_matrix, local_vector)
    temp_core = loc_sol.reshape(lbd, pd, -1)

    local_residual_tensor = res_tensors[ix][:, :, :residual_rank]

    updated_core = torch.cat(
        (temp_core, local_residual_tensor * step_size),
        dim=2,
    )
    if updated_core.shape[-1] > 2:
        updated_core = updated_core[:, :, :2]

    updated_core_q, _ = qr(rearrange(updated_core, "l d r -> (l d) r"))
    updated_core = rearrange(updated_core_q, "(l d) r -> l d r", d=pd)

    new_tensors.append(updated_core)

    # Incremental residual-overlap environment
    res_left_env = einsum(
        updated_core.conj(), res_tensors[ix],
        "l i r, l1 i r1 -> r r1",
    )

    # Update the operator/rhs environment tensors
    lhs_env_tensor_n = _absorb_operator_lenv(A[ix].tensor, updated_core)
    rhs_env_tensor_n = _absorb_rhs_lenv(b[ix].tensor, updated_core)

    lhs_env_tensors_n.insert(0, lhs_env_tensor_n)
    rhs_env_tensors_n.insert(0, rhs_env_tensor_n)

    # --- Interior sites ---
    for ix in range(1, L - 1):
        lbd = updated_core.shape[-1]
        pd = x[ix].shape[1]

        local_matrix = einsum(
            lhs_env_tensor_n,
            A[ix].tensor,
            lhs_env_tensors[ix],
            "l ll ul, l i r o, r lr ur -> ll ul i o lr ur",
        )
        local_matrix = rearrange(
            local_matrix,
            "ll ul i o lr ur -> (ul o ur) (ll i lr)",
        )

        local_vector = einsum(
            rhs_env_tensor_n,
            b[ix].tensor,
            rhs_env_tensors[ix],
            "l ll, l d r, r lr-> ll d lr",
        ).flatten()

        loc_sol = direct_solver(local_matrix, local_vector)
        temp_core = loc_sol.reshape(lbd, pd, -1)

        enrichment = einsum(
            res_left_env, res_tensors[ix],
            "l r1, r1 i r -> l i r",
        )[:, :, :residual_rank]

        if orthogonalize_enrichment:
            enrichment = _orthogonalize_enrichment_forward(temp_core, enrichment, pd)

        updated_core = torch.cat(
            (temp_core, enrichment * step_size),
            dim=2,
        )
        max_allowed_chi = 2 ** (min(ix + 1, L - ix - 1))
        if updated_core.shape[-1] > max_allowed_chi:
            updated_core = updated_core[:, :, :max_allowed_chi]

        updated_core_q, _ = qr(rearrange(updated_core, "l d r -> (l d) r"))
        updated_core = rearrange(updated_core_q, "(l d) r -> l d r", d=pd)

        new_tensors.append(updated_core)

        res_left_env = einsum(
            res_left_env, updated_core.conj(), res_tensors[ix],
            "l1 l2, l1 i r1, l2 i r2 -> r1 r2",
        )

        lhs_env_tensor_n = _absorb_operator_lenv(
            A[ix].tensor, updated_core, lhs_env_tensor_n
        )
        rhs_env_tensor_n = _absorb_rhs_lenv(b[ix].tensor, updated_core, rhs_env_tensor_n)

        lhs_env_tensors_n.insert(0, lhs_env_tensor_n)
        rhs_env_tensors_n.insert(0, rhs_env_tensor_n)

    # --- Last site (no enrichment) ---
    ix = L - 1
    lbd = updated_core.shape[-1]
    pd = x[ix].shape[1]

    local_matrix = einsum(
        lhs_env_tensor_n,
        A[ix].tensor,
        "l ll ul, l i r o -> ll ul i o r",
    )
    local_matrix = rearrange(
        local_matrix,
        "ll ul i o r -> (ul o r) (ll i)",
    )
    local_vector = einsum(
        rhs_env_tensor_n,
        b[ix].tensor,
        "l ll, l d r-> ll d r",
    ).flatten()

    loc_sol = direct_solver(local_matrix, local_vector)
    new_tensors.append(loc_sol.reshape(lbd, pd, -1))

    env_data_n = (lhs_env_tensors_n, rhs_env_tensors_n)
    return (list_to_mps(new_tensors), env_data_n)


def reverse_sweep_amen(
    A,
    b,
    x,
    env_data,
    residual_rank,
    step_size,
    orthogonalize_enrichment=False,
    preconditioner=None,
):
    """
    Performs a reverse sweep in the Alternating Minimal Energy (AMEn) algorithm.

    This function implements the reverse sweep phase of the AMEn algorithm for solving
    linear systems in the Matrix Product State (MPS) format. It updates the cores of
    the solution MPS sequentially from left to right, incorporating residual information
    to improve accuracy.

    Args:
        A (MPO): Matrix Product Operator representing the system matrix
        b (MPS): Matrix Product State representing the right-hand side vector
        x (MPS): Current approximation of the solution in MPS format
        env_data (tuple): Tuple containing left and right environment tensors
            (lhs_env_tensors, rhs_env_tensors)
        residual_rank (int): Maximum bond dimension for the compressed residual
        step_size (float): Step size for the residual directions

    Returns:
        tuple: A tuple containing:
            - MPS: Updated solution after the forward sweep
            - tuple: Updated environment tensors for the next sweep
                (lhs_env_tensors_n, rhs_env_tensors_n)
    """

    lhs_env_tensors, rhs_env_tensors = env_data

    L = len(A)

    lhs_env_tensors_n = []
    rhs_env_tensors_n = []

    new_tensors = []

    residual = b - _apply(A, x)
    if preconditioner is not None:
        residual = _apply(preconditioner, residual)
    res_tensors = residual.tensors

    # --- Last site (L-1) ---
    rbd = 1
    pd = x[-1].shape[1]

    local_matrix = einsum(
        lhs_env_tensors[0],
        A[-1].tensor,
        "l ll ul, l i r o -> ll ul i r o",
    )
    local_matrix = rearrange(
        local_matrix,
        "ll ul i r o -> (ul r o) (ll i)",
    )

    local_vector = einsum(
        rhs_env_tensors[0],
        b[-1].tensor,
        "l lr, l d r -> lr d r",
    ).flatten()

    loc_sol = direct_solver(local_matrix, local_vector)
    temp_core = loc_sol.reshape(-1, pd, rbd)

    local_residual_tensor = res_tensors[-1][:residual_rank, :, :]

    updated_core = torch.cat(
        (temp_core, local_residual_tensor * step_size),
        dim=0,
    )

    if updated_core.shape[0] > 2:
        updated_core = updated_core[:2, :, :]

    _, updated_core_q = rq(rearrange(updated_core, "l d r -> l (d r)"))
    updated_core = rearrange(updated_core_q, "l (d r) -> l d r", d=pd)

    new_tensors.insert(0, updated_core)

    # Incremental residual-overlap environment
    res_right_env = einsum(
        updated_core.conj(), res_tensors[-1],
        "l i r, l1 i r1 -> l l1",
    )

    # Update the operator/rhs environment tensors
    lhs_env_tensor_n = _absorb_operator_renv(A[-1].tensor, updated_core)
    rhs_env_tensor_n = _absorb_rhs_renv(b[-1].tensor, updated_core)

    lhs_env_tensors_n.insert(0, lhs_env_tensor_n)
    rhs_env_tensors_n.insert(0, rhs_env_tensor_n)

    # --- Interior sites ---
    for ix1 in range(1, L - 1):
        ix = L - ix1 - 1
        rbd = updated_core.shape[0]
        pd = x[ix].shape[1]

        local_matrix = einsum(
            lhs_env_tensors[ix1],
            A[ix].tensor,
            lhs_env_tensor_n,
            "l ll ul, l i r o, r lr ur -> ll ul i o lr ur",
        )
        local_matrix = rearrange(
            local_matrix,
            "ll ul i o lr ur -> (ul o ur) (ll i lr)",
        )
        local_vector = einsum(
            rhs_env_tensors[ix1],
            b[ix].tensor,
            rhs_env_tensor_n,
            "l ll, l d r, r lr-> ll d lr",
        ).flatten()

        loc_sol = direct_solver(local_matrix, local_vector)
        temp_core = loc_sol.reshape(-1, pd, rbd)

        enrichment = einsum(
            res_tensors[ix], res_right_env,
            "l i r1, r2 r1 -> l i r2",
        )[:residual_rank, :, :]

        if orthogonalize_enrichment:
            enrichment = _orthogonalize_enrichment_reverse(temp_core, enrichment, pd)

        updated_core = torch.cat(
            (temp_core, enrichment * step_size),
            dim=0,
        )
        max_allowed_chi = 2 ** (min(ix1 + 1, L - ix1 - 1))
        if updated_core.shape[0] > max_allowed_chi:
            updated_core = updated_core[:max_allowed_chi, :, :]

        _, updated_core_q = rq(rearrange(updated_core, "l d r -> l (d r)"))
        updated_core = rearrange(updated_core_q, "l (d r) -> l d r", d=pd)

        new_tensors.insert(0, updated_core)

        res_right_env = einsum(
            res_right_env, updated_core.conj(), res_tensors[ix],
            "r1 r2, l1 i r1, l2 i r2 -> l1 l2",
        )

        lhs_env_tensor_n = _absorb_operator_renv(
            A[ix].tensor, updated_core, lhs_env_tensor_n
        )
        rhs_env_tensor_n = _absorb_rhs_renv(b[ix].tensor, updated_core, rhs_env_tensor_n)

        lhs_env_tensors_n.insert(0, lhs_env_tensor_n)
        rhs_env_tensors_n.insert(0, rhs_env_tensor_n)

    # --- First site (no enrichment) ---
    ix = 0
    rbd = updated_core.shape[0]
    pd = x[ix].shape[1]
    local_matrix = einsum(
        A[ix].tensor,
        lhs_env_tensor_n,
        "l i r o, r lr ur -> l i o lr ur",
    )
    local_matrix = rearrange(
        local_matrix,
        "l i o lr ur -> (o ur) (l i lr)",
    )
    local_vector = einsum(
        b[ix].tensor,
        rhs_env_tensor_n,
        "l d r, r lr-> l d lr",
    ).flatten()

    loc_sol = direct_solver(local_matrix, local_vector)
    new_tensors.insert(0, loc_sol.reshape(-1, pd, rbd))

    env_data_n = (lhs_env_tensors_n, rhs_env_tensors_n)

    return (list_to_mps(new_tensors), env_data_n)


def linsys_amen(
    A: MPO,
    b: MPS,
    x: MPS,
    residual_rank: int = 2,
    step_size: float = 1.0,
    max_iter: int = 2,
    max_rank: int = None,
    eps: float = None,
    orthogonalize_enrichment: bool = False,
    rank_padding: int = 0,
    preconditioner: MPO = None,
    verbose=False,
    save_best=False,
    symmetrize=False,
):
    """Solve linear system Ax=b using Alternating Minimal Energy (AMEn) method.

    This function implements the AMEn algorithm for solving linear systems in the MPS/MPO format.
    The method performs alternating sweeps (forward and reverse) to minimize the energy and
    improve the solution iteratively.

    Args
    ----
    A : MPO
        The system matrix in MPO (Matrix Product Operator) format
    b : MPS
        The right-hand side vector in MPS (Matrix Product State) format
    x : MPS
        Initial guess for the solution in MPS format
    residual_rank : int, optional
        Maximum rank for the residual approximation (default: 2)
    step_size : float, optional
        Step size for the residual directions (default: 1.0).
        Setting ``step_size=0`` disables enrichment (pure ALS), but the
        QR/RQ orthogonalisation of ``cat(core, zeros)`` still introduces
        arbitrary completion vectors that inflate bond dimensions without
        adding information.  Prefer a small positive value instead.
    max_iter : int, optional
        Maximum number of iterations/sweeps (default: 2)
    max_rank : int, optional
        Maximum bond dimension for compression after each sweep. If None, no compression is performed
    eps : float, optional
        Relative residual stopping tolerance.  When the relative residual
        ``||Ax - b|| / ||b||`` drops below this threshold the solver
        terminates early.  ``None`` (default) disables early stopping,
        preserving the original "run all iterations" behaviour.
    orthogonalize_enrichment : bool, optional
        Project residual enrichment directions orthogonal to the current
        solution's column/row space before concatenation (default: False).
        Uses a numerically stable QR-based projector.  Reduces redundant
        rank growth at a small per-site QR cost.
    rank_padding : int, optional
        Extra singular values to keep beyond the truncation threshold during
        intermediate compression steps (default: 0).  Prevents rank
        underestimation on early sweeps.  Padding is removed on the final
        iteration so the output respects ``max_rank`` exactly.
    preconditioner : MPO, optional
        Left preconditioner.  When given the solver works with the
        preconditioned system ``M·A·x = M·b`` by applying ``M`` to
        the residual before computing enrichment directions.
    verbose : bool, optional
        If True, returns both solution and convergence history (default: False)
    save_best : bool, optional
        If True, returns the best solution found during the iterations (default: False)
    symmetrize : {"auto", True, False}, optional
        As in :func:`linsys_dmrg2` -- AMEn solves the same Galerkin local
        system, and residual enrichment often rescues a non-symmetric ``A``
        but nothing guarantees it.  Reported residuals, the ``eps`` test and
        ``save_best`` stay on the original ``‖Ax - b‖/‖b‖``.  ``False`` is the
        default; ``"auto"`` switches whenever ``A`` is not Hermitian.

    Returns
    -------
    MPS or tuple
        If verbose=False, returns only the solution MPS
        If verbose=True, returns a tuple (solution, history) where:
            - solution: The final solution in MPS format
            - history: List of residual norms at each half-sweep
        If save_best=True, returns a tuple (solution, best_solution) where:
            - solution: The final solution in MPS format
            - best_solution: The solution with the lowest residual during the iterations

    Notes
    -----
    The algorithm performs alternating forward and reverse sweeps, updating the solution
    tensor-by-tensor. After each sweep, the residual norm is computed and stored in the history.
    If max_rank is specified, the solution is compressed after each full iteration.

    References
    ----------
    S. V. Dolgov and D. V. Savostyanov,
    "Alternating Minimal Energy Methods for Linear Systems in Higher Dimensions"
    SIAM J. Sci. Comput., 36(5), A2248-A2271, 2014
    """

    if symmetrize == "auto":
        symmetrize = not is_hermitian(A)
    A_sys, b_sys = normal_equations(A, b) if symmetrize else (A, b)
    if symmetrize and preconditioner is not None:
        # M approximates A^-1, so the normal system's is M·M†.
        preconditioner = preconditioner.dagger().apply(preconditioner)

    solution = x.right_canonical()

    env_data = compute_env_tensors_amen(
        A_sys,
        b_sys,
        solution,
    )

    history = []
    loss = _apply(A, solution) - b
    history.append(loss.log_norm - b.log_norm)
    best_loss = float("inf")

    for i in range(max_iter):
        solution, env_data = forward_sweep_amen(
            A_sys,
            b_sys,
            solution,
            env_data,
            residual_rank=residual_rank,
            step_size=step_size,
            orthogonalize_enrichment=orthogonalize_enrichment,
            preconditioner=preconditioner,
        )

        loss = _apply(A, solution) - b
        history.append(loss.log_norm - b.log_norm)

        if history[-1] <= best_loss:
            best_solution = solution.copy()
            best_loss = history[-1]

        if eps is not None and torch.exp(history[-1]) <= eps:
            break

        solution, env_data = reverse_sweep_amen(
            A_sys,
            b_sys,
            solution,
            env_data,
            residual_rank=residual_rank,
            step_size=step_size,
            orthogonalize_enrichment=orthogonalize_enrichment,
            preconditioner=preconditioner,
        )

        is_last = (i == max_iter - 1) or (eps is not None and torch.exp(history[-1]) <= eps)
        effective_max_rank = max_rank
        if max_rank is not None and rank_padding > 0 and not is_last:
            effective_max_rank = max_rank + rank_padding
        pre_ranks = solution.ranks_list
        solution = solution.compress(max_bd=effective_max_rank)

        if solution.ranks_list != pre_ranks:
            env_data = compute_env_tensors_amen(A_sys, b_sys, solution)

        loss = _apply(A, solution) - b
        history.append(loss.log_norm - b.log_norm)

        if history[-1] <= best_loss:
            best_solution = solution.copy()
            best_loss = history[-1]

        if eps is not None and torch.exp(history[-1]) <= eps:
            break

    print("Residuals:", torch.exp(torch.tensor(history)),flush=True)

    if save_best:
        solution = (solution, best_solution)

    if verbose:
        return solution, history

    return solution
