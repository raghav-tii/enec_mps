import inspect
from warnings import warn

import torch
from einops import einsum, rearrange

from ..networks import MPO, MPS, MPO_Node, MPS_Node
from ..utils import (
    GMRES_MIN_SIZE,
    conjgrad,
    direct_solver,
    eigen_solver,
    gmres_solver,
    svd,
)

_DMRG_PATHS_CACHE = {}

MATFREE_THRESHOLD = 200


def _matfree_cg(matvec_fn, b_vec, x0=None, maxiter=None, tol=1e-6, precond=None):
    """(Preconditioned) conjugate gradient for SPD local systems.

    ``precond`` — when given, a callable applying ``M^{-1}`` to a vector.  It
    enters the *recurrence* (textbook PCG) rather than being multiplied into
    the system: forming ``M^{-1}A`` explicitly would destroy the symmetry CG
    relies on.  PCG is invariant under a uniform rescaling of ``M``, so the
    caller need not normalise the preconditioner.

    Returns (solution, converged).  When *converged* is False the caller
    should fall back to a direct solve.
    """
    n = b_vec.shape[0]
    if maxiter is None:
        maxiter = min(n, 300)
    x = torch.zeros_like(b_vec) if x0 is None else x0.clone()
    r = b_vec - matvec_fn(x)
    b_norm = b_vec.norm()
    if b_norm == 0:
        return x, True
    z = r if precond is None else precond(r)
    p = z.clone()
    rz_old = torch.dot(r.conj(), z).real

    for _ in range(maxiter):
        Ap = matvec_fn(p)
        pAp = torch.dot(p.conj(), Ap).real
        if pAp <= 0:
            return x, False
        alpha = rz_old / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        # With a preconditioner ``rz`` is no longer ‖r‖², so test ‖r‖ directly.
        if r.norm() < tol * b_norm:
            return x, True
        z = r if precond is None else precond(r)
        rz_new = torch.dot(r.conj(), z).real
        p = z + (rz_new / rz_old) * p
        rz_old = rz_new
    return x, False


def _normalized(tensor):
    """Return ``(tensor / ‖tensor‖, log ‖tensor‖)``.

    Every environment tensor is carried at unit norm with its scale tracked
    separately in log space — this is what keeps deep sweeps from over/underflowing.
    """
    norm = torch.norm(tensor)
    return tensor / norm, torch.log(norm)


# ---------------------------------------------------------------------------
# Supercore projection
#
# At supercore (k, k+1) the sweep projects an operator onto the current
# interface subspace spanned by the partially-updated cores:
#
#     O_hat_k = (X_<k ⊗ I ⊗ X_>k+1)^T · O · (X_<k ⊗ I ⊗ X_>k+1)
#
# ``project_operator_2site`` / ``project_rhs_2site`` are pure functions of
# (operator block, environments) so the *same* code path serves the system
# operator A, the right-hand side b, and an optional preconditioner P.
#
# Environment leg conventions:
#   operator env : "<mpo bond> <ket bond> <bra bond>"
#   rhs env      : "<mps bond> <bra bond>"
# A ``None`` environment means the supercore touches that end of the chain,
# where the corresponding bonds are all singletons.
# ---------------------------------------------------------------------------


def project_operator_2site(op_l, op_r, left_env=None, right_env=None):
    """Project a two-site operator block onto the current interface subspace.

    Args:
        op_l, op_r: the operator's two MPO core tensors at sites (k, k+1),
            each with legs ``(left_bond, input_physical, right_bond, output_physical)``.
        left_env:  environment over sites ``< k``  (``None`` at the left edge).
        right_env: environment over sites ``> k+1`` (``None`` at the right edge).

    Returns:
        The dense reduced operator, shaped ``(out_dim, in_dim)`` and ready for
        the local solve.
    """
    if left_env is None and right_env is None:
        raise ValueError(
            "project_operator_2site needs at least one environment; a two-site "
            "block with neither is the whole chain."
        )

    if left_env is None:
        block = einsum(
            op_l,
            op_r,
            right_env,
            "l i d o, d i1 r o1, r lr ur -> l i i1 o o1 lr ur",
        )
        return rearrange(block, "l i i1 o o1 lr ur -> (o o1 ur) (l i i1 lr)")

    if right_env is None:
        block = einsum(
            left_env,
            op_l,
            op_r,
            "l ll ul, l i d o, d i1 r o1 -> ll ul i i1 o o1 r",
        )
        return rearrange(block, "ll ul i i1 o o1 r -> (ul o o1 r) (ll i i1)")

    block = einsum(
        left_env,
        op_l,
        op_r,
        right_env,
        "l ll ul, l i d o, d i1 r o1, r lr ur -> ll ul i i1 o o1 lr ur",
    )
    return rearrange(block, "ll ul i i1 o o1 lr ur -> (ul o o1 ur) (ll i i1 lr)")


def project_rhs_2site(rhs_l, rhs_r, left_env=None, right_env=None):
    """Project a two-site right-hand-side block onto the interface subspace.

    Mirrors :func:`project_operator_2site` for an MPS rather than an MPO.
    Returns the flattened reduced vector.
    """
    if left_env is None and right_env is None:
        raise ValueError("project_rhs_2site needs at least one environment.")

    if left_env is None:
        block = einsum(rhs_l, rhs_r, right_env, "l d p, p d1 r, r lr -> l d d1 lr")
    elif right_env is None:
        block = einsum(left_env, rhs_l, rhs_r, "l ll, l d p, p d1 r -> ll d d1 r")
    else:
        block = einsum(
            left_env,
            rhs_l,
            rhs_r,
            right_env,
            "l ll, l d p, p d1 r, r lr -> ll d d1 lr",
        )
    return block.flatten()


# ---------------------------------------------------------------------------
# Incremental environment updates
#
# Each absorbs exactly one site into an environment, given the core the sweep
# has just written.  Called once per operator per site per sweep — adding a
# preconditioner adds one call here, and duplicates nothing on the A/b side.
#
# Operands are listed env, core, operator, bra: even a plain left-to-right
# einsum (no opt_einsum path) then peaks at ~r*R*r*d.  Operator and both
# cores first would build the (R * r**2)**2 six-index site tensor.
# ---------------------------------------------------------------------------


def _absorb_operator_lenv(op_node, core, prev_env=None, bra=None):
    """Extend a left operator environment rightwards by one site.

    ``bra`` is the test-space core; ``None`` means the Galerkin choice, the
    trial core itself.  See :func:`linsys_dmrg2_petrov`.
    """
    bra = core.conj() if bra is None else bra.conj()
    if prev_env is None:
        env = einsum(
            op_node,
            core,
            bra,
            "Al i Ar o, ll i lr, ul o ur -> Al ll ul Ar lr ur",
        )
        # At the left edge Al/ll/ul are all singletons; the regroup just drops them.
        return rearrange(env, "Al ll ul Ar lr ur-> (Al ll ul Ar) lr ur")
    return einsum(
        prev_env,
        core,
        op_node,
        bra,
        "Ar1 lr1 ur1, lr1 i lr, Ar1 i Ar o, ur1 o ur -> Ar lr ur",
    )


def _absorb_operator_renv(op_node, core, prev_env=None, bra=None):
    """Extend a right operator environment leftwards by one site."""
    bra = core.conj() if bra is None else bra.conj()
    if prev_env is None:
        env = einsum(
            op_node,
            core,
            bra,
            "Al i Ar o, ll i lr, ul o ur -> Al ll ul Ar lr ur",
        )
        # At the right edge Ar/lr/ur are all singletons.
        return rearrange(env, "Al ll ul Ar lr ur-> Al ll (ul Ar lr ur)")
    return einsum(
        prev_env,
        core,
        op_node,
        bra,
        "Ar lr ur, ll i lr, Al i Ar o, ul o ur -> Al ll ul",
    )


def _absorb_rhs_lenv(rhs_node, core, prev_env=None, bra=None):
    """Extend a left right-hand-side environment rightwards by one site."""
    bra = core.conj() if bra is None else bra.conj()
    if prev_env is None:
        env = einsum(rhs_node, bra, "bl i br, ul i ur -> bl ul br ur")
        return rearrange(env, "bl ul br ur-> (bl ul br) ur")
    return einsum(prev_env, rhs_node, bra, "br1 ur1, br1 i br, ur1 i ur -> br ur")


def _absorb_rhs_renv(rhs_node, core, prev_env=None, bra=None):
    """Extend a right right-hand-side environment leftwards by one site."""
    bra = core.conj() if bra is None else bra.conj()
    if prev_env is None:
        env = einsum(rhs_node, bra, "bl i br, ul i ur -> bl ul br ur")
        return rearrange(env, "bl ul br ur-> bl (ul br ur)")
    return einsum(prev_env, rhs_node, bra, "br ur, bl i br, ul i ur -> bl ul")


class LocalPreconditioner:
    """Applies ``P_hat^{-1}`` for one supercore, without ever forming it.

    ``P_hat`` is the preconditioner projected onto the *same* interface subspace
    as the reduced system, produced by :func:`project_operator_2site`.

    Two regimes:

    * **diagonal** ``P_hat`` → elementwise scaling, O(n) per apply.
    * **general** ``P_hat`` → one LU factorisation, reused for every apply at
      this supercore.  ``P_hat`` is supercore-sized, so this is small.

    Note that a diagonal TT preconditioner does *not* generally stay diagonal
    after projection: ``X^T P X`` is diagonal only when the interface isometry
    ``X`` aligns with ``P``'s eigenspaces (e.g. ``P = c·I``).  The regime is
    therefore detected at runtime rather than assumed.

    The overall scale of ``P_hat`` is irrelevant — it left-multiplies both sides
    of the reduced system, so it cancels out of the solution and leaves the
    sweep's log-norm bookkeeping untouched.
    """

    # Off-diagonal mass below this fraction of the largest entry counts as zero.
    DIAG_RTOL = 1e-12

    def __init__(self, p_hat, diag_rtol=None):
        if p_hat.ndim != 2 or p_hat.shape[0] != p_hat.shape[1]:
            raise ValueError(f"projected preconditioner must be square, got {tuple(p_hat.shape)}")
        rtol = self.DIAG_RTOL if diag_rtol is None else diag_rtol

        diag = torch.diagonal(p_hat)
        scale = p_hat.abs().max()
        off_diag_mass = (p_hat - torch.diag(diag)).abs().max()
        self.is_diagonal = bool(off_diag_mass <= rtol * scale) and bool((diag.abs() > 0).all())

        self.is_singular = False
        if self.is_diagonal:
            self._inv_diag = 1.0 / diag
            self._lu = None
        else:
            self._inv_diag = None
            lu, piv, info = torch.linalg.lu_factor_ex(p_hat)
            if int(info) != 0:
                # A singular P_hat carries no usable information and inverting
                # it would divide by zero.  Degrade to no preconditioning at
                # this supercore rather than aborting the whole sweep.
                warn(
                    "Projected preconditioner is singular at this supercore; "
                    "solving without preconditioning here."
                )
                self.is_singular = True
                self._lu = None
            else:
                self._lu = (lu, piv)

    def apply(self, vector):
        """Return ``P_hat^{-1} · v``."""
        return self.apply_matrix(vector.reshape(-1, 1)).reshape(-1)

    def apply_matrix(self, matrix):
        """Return ``P_hat^{-1} · M`` for a matrix (or column-vector) ``M``."""
        if self.is_singular:
            return matrix
        if self.is_diagonal:
            dtype = torch.promote_types(self._inv_diag.dtype, matrix.dtype)
            return self._inv_diag.to(dtype).unsqueeze(1) * matrix.to(dtype)
        lu, piv = self._lu
        # A real P against a complex system (or vice versa) must meet in the
        # common promoted dtype before lu_solve, which requires matching inputs.
        dtype = torch.promote_types(lu.dtype, matrix.dtype)
        return torch.linalg.lu_solve(lu.to(dtype), piv, matrix.to(dtype))


def make_local_solver(
    method="direct",
    maximal_eigenval=False,
    matfree_threshold=MATFREE_THRESHOLD,
):
    """Build the supercore solver used by the DMRG sweeps.

    Replaces the old pile of mutually-exclusive boolean flags with one name.
    The returned callable has a single uniform signature::

        solve(matrix, vector, precond=None) -> solution

    where ``precond`` is an optional :class:`LocalPreconditioner`.  How it is
    applied depends on the method — iterative solvers take it as a genuine
    preconditioner (never pre-multiplied, which would cost stability), direct
    solvers apply it to both sides:

    ``"direct"``   dense solve of ``P^{-1}A x = P^{-1}b``.
    ``"cg"``       conjugate gradient.  Unpreconditioned this is the library's
                   :func:`conjgrad`; with a preconditioner it is textbook PCG
                   (:func:`_matfree_cg`), since ``conjgrad`` has no hook for one.
    ``"gmres"``    GMRES with ``P^{-1}`` as left preconditioner, displacing the
                   ILU that :func:`gmres_solver` builds for itself otherwise.
    ``"matfree"``  direct below ``matfree_threshold``, else (P)CG with a direct
                   fallback when it fails to converge.
    ``"eigen"``    eigen-decomposition; ignores the right-hand side entirely and
                   has no meaningful preconditioner.
    """
    if method == "direct":

        def solve(matrix, vector, precond=None):
            if precond is not None:
                matrix = precond.apply_matrix(matrix)
                vector = precond.apply(vector)
            return direct_solver(matrix, vector)

    elif method == "cg":

        def solve(matrix, vector, precond=None):
            if precond is None:
                return conjgrad(matrix, vector)
            sol, converged = _matfree_cg(lambda v: matrix @ v, vector, precond=precond.apply)
            if not converged:
                return direct_solver(matrix, vector)
            return sol

    elif method == "gmres":

        def solve(matrix, vector, precond=None):
            if precond is None:
                return gmres_solver(matrix, vector)
            if vector.shape[0] <= GMRES_MIN_SIZE:
                # gmres_solver solves densely at this size anyway; go straight
                # there so we can use the cheap matrix-wide preconditioner apply.
                return direct_solver(precond.apply_matrix(matrix), precond.apply(vector))
            return gmres_solver(matrix, vector, precond=precond.apply)

    elif method == "matfree":

        def solve(matrix, vector, precond=None):
            if vector.shape[0] <= matfree_threshold:
                if precond is not None:
                    return direct_solver(precond.apply_matrix(matrix), precond.apply(vector))
                return direct_solver(matrix, vector)
            sol, converged = _matfree_cg(
                lambda v: matrix @ v,
                vector,
                precond=None if precond is None else precond.apply,
            )
            if not converged:
                return direct_solver(matrix, vector)
            return sol

    elif method == "eigen":

        def solve(matrix, vector=None, precond=None):
            _, eigvec = eigen_solver(matrix, descending=maximal_eigenval)
            return eigvec

    else:
        raise ValueError(
            f"unknown local solver {method!r}; expected one of "
            "'direct', 'cg', 'gmres', 'matfree', 'eigen'"
        )

    return solve


def _adapt_solver_signature(fn):
    """Adapt a caller-supplied solver to the ``(matrix, vector, precond)`` protocol.

    Solvers written against the old two-argument signature still work: the
    preconditioner is applied to both sides before handing over.  A solver that
    declares ``precond`` (or ``**kwargs``) receives it directly.
    """
    try:
        params = inspect.signature(fn).parameters
        handles_precond = "precond" in params or any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
    except (TypeError, ValueError):  # builtins / C callables have no signature
        handles_precond = False

    if handles_precond:
        return fn

    def solve(matrix, vector, precond=None):
        if precond is not None:
            matrix = precond.apply_matrix(matrix)
            vector = precond.apply(vector)
        return fn(matrix, vector)

    return solve


def _resolve_local_solver(
    local_solver,
    cgrad,
    gmres_flag,
    eig_solver,
    matfree,
    maximal_eigenval,
    matfree_threshold,
):
    """Map the boolean flags onto :func:`make_local_solver`.

    ``local_solver`` may be a method name, an already-built callable, or None
    (in which case the booleans decide, preserving the precedence
    cgrad > eigen > gmres > matfree > direct).
    """
    if callable(local_solver):
        return _adapt_solver_signature(local_solver)

    if local_solver is None:
        if cgrad:
            local_solver = "cg"
        elif eig_solver:
            local_solver = "eigen"
        elif gmres_flag:
            local_solver = "gmres"
        elif matfree:
            local_solver = "matfree"
        else:
            local_solver = "direct"

    return make_local_solver(
        local_solver,
        maximal_eigenval=maximal_eigenval,
        matfree_threshold=matfree_threshold,
    )


def compute_env_tensors(A, b, x, test=None):
    """
    Computes the environment tensors for the MPO-MPS sandwich.

    Args:
        A (MPO): MPO encoding of the operator.
        b (MPS): MPS encoding of the right-hand side.
        x (MPS): MPS encoding of the initial guess.
        test (MPS, optional): Test space whose cores take the bra slot.
            ``None`` (default) is the Galerkin choice, ``x`` itself.
        xlognorm (optional): Lognorm of the `x` MPS. Defaults to None.

    Returns:
        tuple: Environment tensors computed for both the left and right hand side of the linear system.
    """

    lhs_env_tensors = []
    rhs_env_tensors = []

    lhs_env_lognorms = []
    rhs_env_lognorms = []

    bra = x if test is None else test
    lhs_env_tensor = einsum(
        A[-1].tensor,
        x[-1].tensor,
        bra[-1].tensor.conj(),
        "Al i Ar o, ll i lr, ul o ur -> Al ll ul Ar lr ur",
    )
    lhs_env_tensor = rearrange(
        lhs_env_tensor,
        "Al ll ul Ar lr ur-> Al ll (ul Ar lr ur)",
    )
    rhs_env_tensor = einsum(
        b[-1].tensor,
        bra[-1].tensor.conj(),
        "bl i br, ul i ur -> bl ul br ur",
    )
    rhs_env_tensor = rearrange(rhs_env_tensor, "bl ul br ur-> bl (ul br ur)")

    lhs_lognorm = torch.log(torch.norm(lhs_env_tensor))
    rhs_lognorm = torch.log(torch.norm(rhs_env_tensor))

    lhs_env_tensor = lhs_env_tensor / torch.norm(lhs_env_tensor)
    rhs_env_tensor = rhs_env_tensor / torch.norm(rhs_env_tensor)

    lhs_env_tensors.insert(0, lhs_env_tensor)
    rhs_env_tensors.insert(0, rhs_env_tensor)

    lhs_env_lognorms.insert(0, lhs_lognorm)
    rhs_env_lognorms.insert(0, rhs_lognorm)

    for ix in range(1, len(A) - 1):
        ix = len(A) - ix - 1
        # Contract each site straight into the running environment.  Forming
        # the six-index intermediate "Al ll ul Ar lr ur" first and only then
        # closing it against the environment costs O(chi_A**2 * chi_x**4)
        # memory -- a single 22.8 GB allocation at chi_A = chi_x = 40, which
        # is what made a 12-site complex solve unrunnable -- while this order
        # peaks at O(chi_A * chi_x**2 * pd).  The sweeps below already build
        # their environments this way; this loop was the last one that did not.
        lhs_env_tensor = einsum(
            lhs_env_tensor,
            A[ix].tensor,
            x[ix].tensor,
            bra[ix].tensor.conj(),
            "Ar lr ur, Al i Ar o, ll i lr, ul o ur -> Al ll ul",
        )

        rhs_env_tensor = einsum(
            rhs_env_tensor,
            b[ix].tensor,
            bra[ix].tensor.conj(),
            "br ur, bl i br, ul i ur -> bl ul",
        )

        # Normalising the contracted environment rather than the (now never
        # built) intermediate preserves the same invariant the rest of this
        # file relies on -- true_env == tensor * exp(lognorm) -- and leaves
        # every stored environment at unit norm, which the old form did not.
        lhs_norm = torch.norm(lhs_env_tensor)
        rhs_norm = torch.norm(rhs_env_tensor)

        lhs_env_tensor = lhs_env_tensor / lhs_norm
        rhs_env_tensor = rhs_env_tensor / rhs_norm

        lhs_env_lognorms.insert(0, lhs_env_lognorms[0] + torch.log(lhs_norm))
        rhs_env_lognorms.insert(0, rhs_env_lognorms[0] + torch.log(rhs_norm))

        lhs_env_tensors.insert(0, lhs_env_tensor)
        rhs_env_tensors.insert(0, rhs_env_tensor)

    return lhs_env_tensors, rhs_env_tensors, lhs_env_lognorms, rhs_env_lognorms


def forward_sweep(
    A,
    b,
    x,
    trunc_eps,
    max_rank,
    env_data,
    local_solver,
    track_cond: bool = False,
    preconditioner=None,
    precond_env=None,
    test=None,
):
    """Left-to-right half-sweep over every supercore ``(ix, ix + 1)``.

    Consumes the right-environment stack built by the preceding reverse sweep
    and builds the left-environment stack the next one will consume.

    When ``preconditioner`` is given its environments ride along in
    ``precond_env`` and are extended from the *same* freshly-written cores as
    A's and b's — one extra contraction per site, with nothing on the A/b side
    recomputed.

    Returns ``(solution, env_data, sol_lognorm, cond_list, precond_env)``.
    """
    lhs_env_tensors, rhs_env_tensors, lhs_env_lognorms, rhs_env_lognorms = env_data

    lhs_env_tensors_n = []
    rhs_env_tensors_n = []

    lhs_env_lognorms_n = []
    rhs_env_lognorms_n = []

    precond_env_n = None if preconditioner is None else []

    local_sol_lognorms = []
    cond_list = []

    new_nodes = []

    L = len(A)
    # How many solution factors each environment carries; see the note by the
    # log-norm updates below.
    n_sol_lhs, n_sol_rhs = (2, 1) if test is None else (1, 0)

    for ix in range(L - 1):
        at_left = ix == 0
        at_right = ix == L - 2

        # Environments bracketing this supercore.  Left: built earlier in *this*
        # sweep.  Right: inherited from the previous (reverse) sweep.
        left_env = None if at_left else lhs_env_tensors_n[0]
        right_env = None if at_right else lhs_env_tensors[ix + 1]
        rhs_left_env = None if at_left else rhs_env_tensors_n[0]
        rhs_right_env = None if at_right else rhs_env_tensors[ix + 1]

        lbd = 1 if at_left else left_env.shape[1]
        pd = x[ix].shape[1]

        # --- reduced system ------------------------------------------------
        local_matrix = project_operator_2site(A[ix].tensor, A[ix + 1].tensor, left_env, right_env)
        local_vector = project_rhs_2site(
            b[ix].tensor, b[ix + 1].tensor, rhs_left_env, rhs_right_env
        )

        local_matrix_n, mat_lognorm = _normalized(local_matrix)
        local_vector_n, vec_lognorm = _normalized(local_vector)

        local_mat_lognorm = mat_lognorm
        local_vec_lognorm = vec_lognorm
        if not at_left:
            local_mat_lognorm = local_mat_lognorm + lhs_env_lognorms_n[0]
            local_vec_lognorm = local_vec_lognorm + rhs_env_lognorms_n[0]
        if not at_right:
            local_mat_lognorm = local_mat_lognorm + lhs_env_lognorms[ix + 1]
            local_vec_lognorm = local_vec_lognorm + rhs_env_lognorms[ix + 1]

        # --- preconditioner, projected onto the *same* interface subspace ---
        precond = None
        if preconditioner is not None:
            precond = LocalPreconditioner(
                project_operator_2site(
                    preconditioner[ix].tensor,
                    preconditioner[ix + 1].tensor,
                    None if at_left else precond_env_n[0],
                    None if at_right else precond_env[ix + 1],
                )
            )

        if track_cond:
            _sv = torch.linalg.svdvals(local_matrix_n)
            cond_list.append((_sv[0].item(), _sv[-1].item()))

        loc_sol = local_solver(local_matrix_n, local_vector_n, precond)

        # --- split the supercore, keep the left factor ----------------------
        matrix = loc_sol.reshape(lbd * pd, -1)
        U, S, V = svd(matrix, trunc_eps, max_rank)
        tensor1 = U.reshape(lbd, pd, -1)
        new_nodes.append(MPS_Node(tensor1))
        local_sol_lognorms.append(local_vec_lognorm - local_mat_lognorm)

        # --- absorb the new core into the left environments -----------------
        prev_lhs_lognorm = 0.0 if at_left else lhs_env_lognorms_n[0]
        prev_rhs_lognorm = 0.0 if at_left else rhs_env_lognorms_n[0]

        bra = None if test is None else test[ix].tensor
        lhs_env_tensor_n, lhs_lognorm = _normalized(
            _absorb_operator_lenv(A[ix].tensor, tensor1, left_env, bra)
        )
        rhs_env_tensor_n, rhs_lognorm = _normalized(
            _absorb_rhs_lenv(b[ix].tensor, tensor1, rhs_left_env, bra)
        )

        lhs_env_lognorms_n.insert(
            0, prev_lhs_lognorm + lhs_lognorm + n_sol_lhs * local_sol_lognorms[-1]
        )
        rhs_env_lognorms_n.insert(
            0, prev_rhs_lognorm + rhs_lognorm + n_sol_rhs * local_sol_lognorms[-1]
        )
        lhs_env_tensors_n.insert(0, lhs_env_tensor_n)
        rhs_env_tensors_n.insert(0, rhs_env_tensor_n)

        if preconditioner is not None:
            # Same core, same interface subspace.  P's scale cancels out of the
            # reduced system, so unlike A and b it needs no log-norm tracking.
            p_env, _ = _normalized(
                _absorb_operator_lenv(
                    preconditioner[ix].tensor,
                    tensor1,
                    None if at_left else precond_env_n[0],
                    bra,
                )
            )
            precond_env_n.insert(0, p_env)

    env_data_n = (
        lhs_env_tensors_n,
        rhs_env_tensors_n,
        lhs_env_lognorms_n,
        rhs_env_lognorms_n,
    )

    # Add the last tensor from the latest local solution
    rbd = x[-1].shape[2]
    Sn = S / torch.norm(S)
    tensor2 = Sn.type(matrix.dtype).unsqueeze(1) * V
    tensor2 = tensor2.reshape(-1, pd, rbd)
    new_nodes.append(MPS_Node(tensor2))
    local_sol_lognorms.append(torch.log(torch.norm(S)))

    return (
        MPS(new_nodes),
        env_data_n,
        sum(local_sol_lognorms),
        cond_list,
        precond_env_n,
    )


def reverse_sweep(
    A,
    b,
    x,
    trunc_eps,
    max_rank,
    env_data,
    local_solver,
    track_cond: bool = False,
    preconditioner=None,
    precond_env=None,
    test=None,
):
    """Right-to-left half-sweep — the mirror of :func:`forward_sweep`.

    Consumes the left-environment stack built by the preceding forward sweep
    and builds the right-environment stack for the next one.  Preconditioner
    handling is identical: one extra contraction per site off the same cores.

    Returns ``(solution, env_data, sol_lognorm, cond_list, precond_env)``.
    """
    lhs_env_tensors, rhs_env_tensors, lhs_env_lognorms, rhs_env_lognorms = env_data

    lhs_env_tensors_n = []
    rhs_env_tensors_n = []

    lhs_env_lognorms_n = []
    rhs_env_lognorms_n = []

    precond_env_n = None if preconditioner is None else []

    local_sol_lognorms = []
    cond_list = []

    new_nodes = []

    L = len(A)
    n_sol_lhs, n_sol_rhs = (2, 1) if test is None else (1, 0)

    for ix in range(L - 2, -1, -1):
        at_left = ix == 0
        at_right = ix == L - 2
        # Index into the inherited (forward-built) left-environment stack:
        # entry j covers sites 0 .. L-2-j, and we want sites 0 .. ix-1.
        inherited = L - 1 - ix

        # Right: built earlier in *this* sweep.  Left: inherited from the last.
        left_env = None if at_left else lhs_env_tensors[inherited]
        right_env = None if at_right else lhs_env_tensors_n[0]
        rhs_left_env = None if at_left else rhs_env_tensors[inherited]
        rhs_right_env = None if at_right else rhs_env_tensors_n[0]

        rbd = 1 if at_right else right_env.shape[1]
        pd = x[ix + 1].shape[1]

        # --- reduced system ------------------------------------------------
        local_matrix = project_operator_2site(A[ix].tensor, A[ix + 1].tensor, left_env, right_env)
        local_vector = project_rhs_2site(
            b[ix].tensor, b[ix + 1].tensor, rhs_left_env, rhs_right_env
        )

        local_matrix_n, mat_lognorm = _normalized(local_matrix)
        local_vector_n, vec_lognorm = _normalized(local_vector)

        local_mat_lognorm = mat_lognorm
        local_vec_lognorm = vec_lognorm
        if not at_left:
            local_mat_lognorm = local_mat_lognorm + lhs_env_lognorms[inherited]
            local_vec_lognorm = local_vec_lognorm + rhs_env_lognorms[inherited]
        if not at_right:
            local_mat_lognorm = local_mat_lognorm + lhs_env_lognorms_n[0]
            local_vec_lognorm = local_vec_lognorm + rhs_env_lognorms_n[0]

        # --- preconditioner, projected onto the *same* interface subspace ---
        precond = None
        if preconditioner is not None:
            precond = LocalPreconditioner(
                project_operator_2site(
                    preconditioner[ix].tensor,
                    preconditioner[ix + 1].tensor,
                    None if at_left else precond_env[inherited],
                    None if at_right else precond_env_n[0],
                )
            )

        if track_cond:
            _sv = torch.linalg.svdvals(local_matrix_n)
            cond_list.insert(0, (_sv[0].item(), _sv[-1].item()))

        loc_sol = local_solver(local_matrix_n, local_vector_n, precond)

        # --- split the supercore, keep the right factor ---------------------
        matrix = loc_sol.reshape(-1, pd * rbd)
        U, S, V = svd(matrix, trunc_eps, max_rank)
        tensor1 = V.reshape(-1, pd, rbd)
        new_nodes.insert(0, MPS_Node(tensor1))
        local_sol_lognorms.insert(0, local_vec_lognorm - local_mat_lognorm)

        # --- absorb the new core into the right environments -----------------
        prev_lhs_lognorm = 0.0 if at_right else lhs_env_lognorms_n[0]
        prev_rhs_lognorm = 0.0 if at_right else rhs_env_lognorms_n[0]

        bra = None if test is None else test[ix + 1].tensor
        lhs_env_tensor_n, lhs_lognorm = _normalized(
            _absorb_operator_renv(A[ix + 1].tensor, tensor1, right_env, bra)
        )
        rhs_env_tensor_n, rhs_lognorm = _normalized(
            _absorb_rhs_renv(b[ix + 1].tensor, tensor1, rhs_right_env, bra)
        )

        lhs_env_lognorms_n.insert(
            0, prev_lhs_lognorm + lhs_lognorm + n_sol_lhs * local_sol_lognorms[0]
        )
        rhs_env_lognorms_n.insert(
            0, prev_rhs_lognorm + rhs_lognorm + n_sol_rhs * local_sol_lognorms[0]
        )
        lhs_env_tensors_n.insert(0, lhs_env_tensor_n)
        rhs_env_tensors_n.insert(0, rhs_env_tensor_n)

        if preconditioner is not None:
            p_env, _ = _normalized(
                _absorb_operator_renv(
                    preconditioner[ix + 1].tensor,
                    tensor1,
                    None if at_right else precond_env_n[0],
                    bra,
                )
            )
            precond_env_n.insert(0, p_env)

    env_data_n = (
        lhs_env_tensors_n,
        rhs_env_tensors_n,
        lhs_env_lognorms_n,
        rhs_env_lognorms_n,
    )

    # Add the last tensor from the latest local solution
    lbd = x[0].shape[0]
    Sn = S / torch.norm(S)
    tensor2 = U * Sn.type(matrix.dtype).unsqueeze(0)
    tensor2 = tensor2.reshape(lbd, pd, -1)
    new_nodes.insert(0, MPS_Node(tensor2))
    local_sol_lognorms.insert(0, torch.log(torch.norm(S)))

    return (
        MPS(new_nodes),
        env_data_n,
        sum(local_sol_lognorms),
        cond_list,
        precond_env_n,
    )


def is_hermitian(A: MPO) -> bool:
    """Is ``A`` Hermitian to within 1e-10 of its Frobenius norm?"""
    return float((A + A.dagger() * -1.0).norm / A.norm) <= 1e-10


def normal_equations(A: MPO, b: MPS):
    """Return ``(A†A, A†b)``, the symmetrized form of ``A x = b``.

    ``MPO.apply`` composes in *reverse* order --- ``U.apply(V)`` is the operator
    ``V ∘ U`` --- so ``A.apply(A.dagger())`` is ``A†A``, not ``AA†``.
    """
    A_dag = A.dagger()
    return A.apply(A_dag), A_dag.apply(b)


def _petrov_test_space(A, x):
    """Orthonormal test frames for a Petrov-Galerkin sweep over ``A x = b``.

    The test space is ``A·x`` at its *natural* rank ``r_A·r_x``.  Truncating it
    back to the solution's ranks is tempting and wrong: the extra factor ``r_A``
    is exactly the operator bond that couples the free supercore to the frozen
    cores, so dropping it leaves a test space that no longer spans ``A·V`` and
    the projection loses its minimal-residual property (measured: a residual of
    1.6 where Galerkin reaches 9e-10 on an SPD system).

    Keeping it makes the reduced system tall — ``(r_A r_x d² r_A r_x)`` rows
    against ``(r_x d² r_x)`` columns — which is the point: the local solve is a
    least-squares problem, so it minimises the residual.

    Returned twice, left- and right-canonical, because a forward half-sweep
    needs left-orthonormal cores behind it and a reverse one needs
    right-orthonormal cores ahead of it — the mixed-canonical gauge the trial
    space already uses.
    """
    w = A.apply(x)
    return w.left_canonical(), w.right_canonical()


def linsys_dmrg2(
    A: MPO,
    b: MPS,
    x: MPS,
    trunc_eps: float,
    max_rank: int,
    max_iter: int = 2,
    preconditioner: MPO = None,
    local_solver=None,
    cgrad=False,
    gmres_flag=False,
    eig_solver=False,
    maximal_eigenval=False,
    noisy_convergence=False,
    verbose=False,
    track_cond: bool = False,
    matfree: bool = False,
    matfree_threshold: int = MATFREE_THRESHOLD,
    symmetrize=False,
    test_space="galerkin",
):
    """Solve ``A x = b`` in MPO/MPS format with two-site DMRG (mALS) sweeps.

    Args:
        A: MPO encoding of the operator.
        b: MPS encoding of the right-hand side.
        x: Initial guess (MPS).
        trunc_eps: SVD truncation tolerance when splitting each supercore.
        max_rank: Hard cap on the MPS bond dimension.
        max_iter: Number of forward+reverse sweep pairs.
        symmetrize: Sweep on the normal equations ``A†A x = A†b`` instead.  The
            Galerkin system built here minimises ``½⟨x,Ax⟩ - ⟨x,b⟩``, a minimum
            only for Hermitian positive-definite ``A``; the normal equations
            minimise ``‖Ax - b‖`` for any ``A``, at a squared condition number
            and rank ``r_A²`` before compression.  ``False`` (default) leaves the
            system alone; ``True`` symmetrises; ``"auto"`` symmetrises whenever
            ``A`` is not Hermitian (:func:`is_hermitian`).  For a Hermitian
            *indefinite* ``A`` (Helmholtz) the energy is a saddle and ``"auto"``
            will not catch it — pass ``True``.  ``verbose`` still reports
            ``‖Ax - b‖/‖b‖`` of the original system.
        test_space: ``"galerkin"`` (default) tests the equation against the
            trial space; ``"petrov"`` tests against ``orth(A·X)``, which makes
            each local solve minimise ``‖Ax - b‖`` for any ``A`` without forming
            ``A†A``.  See :func:`linsys_dmrg2_petrov`.
        preconditioner: Optional TT operator ``P`` with the same mode structure
            as ``A``.  When given, each supercore's reduced system is solved as
            ``P_hat^{-1} A_hat s = P_hat^{-1} b_hat``, where ``P_hat`` is ``P``
            projected onto the *same* interface subspace as ``A_hat`` — by the
            same :func:`project_operator_2site` call, off environments extended
            from the same cores.  This changes convergence, not the answer.
        local_solver: Supercore solver — a name accepted by
            :func:`make_local_solver` (``"direct"``, ``"cg"``, ``"gmres"``,
            ``"matfree"``, ``"eigen"``) or a callable.  Supersedes the
            boolean flags below.
        maximal_eigenval: For ``"eigen"``, take the largest eigenvalue.
        verbose: Also return the residual history.
        track_cond: Also record per-supercore extremal singular values.
        matfree_threshold: Size above which ``"matfree"`` switches to CG.

    Returns:
        ``(solution, sol_lognorm)``; the dense solution is
        ``solution.contract() * exp(sol_lognorm)``.  With ``verbose=True`` the
        residual ``history`` is appended, and with ``track_cond=True`` the
        condition-number history after that.
    """
    local_solver = _resolve_local_solver(
        local_solver,
        cgrad,
        gmres_flag,
        eig_solver,
        matfree,
        maximal_eigenval,
        matfree_threshold,
    )

    if symmetrize == "auto":
        symmetrize = not is_hermitian(A)
    A_sys, b_sys = normal_equations(A, b) if symmetrize else (A, b)
    if symmetrize and preconditioner is not None:
        # P approximates A, so the normal system's is P†P -- the same product.
        preconditioner = preconditioner.apply(preconditioner.dagger())

    if test_space not in ("galerkin", "petrov"):
        raise ValueError(f"test_space must be 'galerkin' or 'petrov', got {test_space!r}")
    petrov = test_space == "petrov"
    if petrov and preconditioner is not None:
        raise ValueError(
            "test_space='petrov' has more test than trial directions, so the "
            "projected preconditioner is rectangular and cannot be inverted; "
            "precondition the system instead (see rank1_precondition)."
        )

    solution = x.right_canonical()
    test_fwd = test_rev = None
    if petrov:
        test_fwd, test_rev = _petrov_test_space(A_sys, solution)
    env_data = compute_env_tensors(A_sys, b_sys, solution, test_rev)

    # The preconditioner's own environment stack over the same interface basis.
    # Operator environments carry the operator's MPO bond index, so P needs its
    # own stack; what is shared with A is the basis (the cores), which is what
    # makes one pass over the chain enough for both.
    precond_env = None
    if preconditioner is not None:
        preconditioner = _align_operator_dtype(preconditioner, solution[0].tensor.dtype)
        precond_env = compute_mpo_env(preconditioner, solution)

    # `history`/`loss` exist solely to report per-half-sweep residual decay
    # when the caller asks for it (verbose=True). Computing `loss = A.apply(
    # solution) - b` is a full operator-MPS apply (with its own auto-compress)
    # per half-sweep -- real work that every Stage 3/4 Newton step in
    # qtt_ac_newton was paying for and discarding, since none of those call
    # sites pass verbose=True. Gating it (and the unconditional debug print
    # below, which is independent of verbose entirely) is a pure removal of
    # unused work: `sol_lognorm` itself -- the value the caller actually
    # needs to scale the returned solution -- comes straight from
    # forward_sweep/reverse_sweep and is computed unconditionally either way.
    history = [] if verbose else None
    cond_history = [] if track_cond else None
    if verbose:
        loss = A.apply(solution) - b
        history.append(loss.log_norm - b.log_norm)

    for i in range(max_iter):
        solution, env_data, sol_lognorm, cond_fwd, precond_env = forward_sweep(
            A_sys,
            b_sys,
            solution,
            trunc_eps,
            max_rank,
            env_data,
            local_solver,
            track_cond=track_cond,
            preconditioner=preconditioner,
            precond_env=precond_env,
            test=test_fwd,
        )

        if verbose:
            loss = A.apply(solution) * torch.exp(sol_lognorm) - b
            history.append(loss.log_norm - b.log_norm)
        if track_cond:
            cond_history.append(cond_fwd)

        solution, env_data, sol_lognorm, cond_rev, precond_env = reverse_sweep(
            A_sys,
            b_sys,
            solution,
            trunc_eps,
            max_rank,
            env_data,
            local_solver,
            track_cond=track_cond,
            preconditioner=preconditioner,
            precond_env=precond_env,
            test=test_rev,
        )

        if verbose:
            loss = A.apply(solution) * torch.exp(sol_lognorm) - b
            history.append(loss.log_norm - b.log_norm)
        if track_cond:
            cond_history.append(cond_rev)

        if petrov and i < max_iter - 1:
            # The test space trails the solution, so refresh it once per sweep
            # pair -- and with it the right-environment stack it is baked into.
            # ponytail: one refresh per pair, not per half-sweep; tighten if a
            # problem shows the test space going stale within an iteration.
            test_fwd, test_rev = _petrov_test_space(A_sys, solution)
            env_data = compute_env_tensors(A_sys, b_sys, solution, test_rev)

        if noisy_convergence:
            raise NotImplementedError(
                "Current support does not include noise, while using the lognorm trick."
            )

    if verbose:
        print("Residuals:", torch.exp(torch.tensor(history)), flush=True)
        if track_cond:
            return solution, sol_lognorm, history, cond_history
        return solution, sol_lognorm, history

    return solution, sol_lognorm


def _align_operator_dtype(operator, dtype, label="preconditioner"):
    """Return ``operator`` with every core cast up to ``dtype``.

    A real ``P`` handed to a complex system is the ordinary case (the user built
    ``P`` at the default dtype) and is promoted silently.  A complex ``P`` over a
    real system is not: the solution it implies cannot be represented by the
    real cores of ``A`` and ``b``, so say so here rather than failing later
    inside a contraction.
    """
    if torch.promote_types(operator[0].tensor.dtype, dtype) != dtype:
        raise ValueError(
            f"{label} has dtype {operator[0].tensor.dtype}, which does not fit "
            f"the system's {dtype}; convert A and b to a complex dtype first."
        )
    if all(node.tensor.dtype == dtype for node in operator):
        return operator
    return MPO([MPO_Node(node.tensor.to(dtype)) for node in operator], validate=False)


def compute_mpo_env(A, x, normalize=True):
    """Right environments of the sandwich ``x† · A · x``, one per supercore.

    Entry ``j`` is the environment over sites ``j+1 … L-1``, matching the
    indexing of the operator environments returned by
    :func:`compute_env_tensors` so that the two stacks can be walked together
    during a sweep.  This is what seeds the preconditioner's environment stack
    in :func:`linsys_dmrg2`.

    ``normalize`` keeps every entry at unit norm.  The discarded scale factors
    are deliberately *not* tracked: this stack feeds a preconditioner, whose
    overall scale cancels out of the reduced system.
    """
    env_tensors = []
    env = None
    for ix in range(len(A) - 1, 0, -1):
        env = _absorb_operator_renv(A[ix].tensor, x[ix].tensor, env)
        if normalize:
            env, _ = _normalized(env)
        env_tensors.insert(0, env)
    return env_tensors
