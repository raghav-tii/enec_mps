import math
import torch
from warnings import warn

from ..networks import (
    MPO,
    MPO_Node,
    MPS,
    identity_mpo,
    random_mps,
    vector_to_2nodes,
    vector_to_node,
    mps_to_mpo,
)
from ..utils import (
    conjgrad,
    direct_solver,
    eigen_solver,
    get_arrays_einsum,
    get_best_path,
)

_DMRG_PATHS_CACHE = {}


def get_opt_paths(A_shapes, b_shapes, x_shapes, dmrg_sites):
    warn("Running con path optimization step.")
    opt_paths_ip = []
    opt_paths_opip = []
    for ix in range(len(A_shapes)):
        path_ip, path_opip = get_best_path(
            A_shapes, b_shapes, x_shapes, list(range(ix, ix + dmrg_sites))
        )
        opt_paths_ip.append(path_ip)
        opt_paths_opip.append(path_opip)
    return opt_paths_ip, opt_paths_opip


def linsystem_solver(
    A: MPO,
    b: MPS,
    x: MPS = None,
    max_rank: MPS = 10,
    max_iter: int = 2,
    eps: float = 1e-2,
    dmrg_sites: int = 2,
    opt_paths=True,
    cgrad=False,
    verbose=False,
):
    pd = A[0].shape[1]
    assert dmrg_sites < 3, f"Performing DMRG with {dmrg_sites} sites is not supported"
    var = 0.2
    if x is None:
        x = random_mps(L=len(b), pd=pd, bd=max_rank, var=var)
        x.mixed_canonical(len(x) // 2)
    history = []
    b_norm = b.norm
    error = (A.apply(x) + b * -1.0).norm / b_norm
    history.append(error)
    A_shapes = [node.shape for node in A]
    b_shapes = [node.shape for node in b]
    x_shapes = [node.shape for node in x]

    key = str(
        abs(hash((tuple(A_shapes), tuple(b_shapes), tuple(x_shapes), dmrg_sites)))
    )

    if opt_paths:
        if key in _DMRG_PATHS_CACHE.keys():
            _DMRG_PATHS_CACHE.pop(key)

    try:
        opt_paths1, opt_paths2 = _DMRG_PATHS_CACHE[key]
    except KeyError:
        opt_paths1, opt_paths2 = get_opt_paths(A_shapes, b_shapes, x_shapes, dmrg_sites)
        _DMRG_PATHS_CACHE[key] = (opt_paths1, opt_paths2)

    for iter in range(max_iter):
        for ix in range(len(A) - dmrg_sites + 1):
            if iter % 2 != 0:
                ix = len(A) - ix - dmrg_sites
            b.mixed_canonical(ix)
            x.mixed_canonical(ix)
            arrays1, arrays2 = get_arrays_einsum(
                A, b, x, list(range(ix, ix + dmrg_sites))
            )
            vector = opt_paths1[ix](*arrays1).flatten()
            matrix = opt_paths2[ix](*arrays2).flatten()
            n1 = int(math.sqrt(matrix.numel()))
            matrix = matrix.reshape(n1, n1)
            rl = x.ranks_list
            if ix == 0:
                boundary1 = "left"
                boundary2 = None
            elif ix == len(A) - 2:
                boundary1 = None
                boundary2 = "right"
            else:
                boundary1 = None
                boundary2 = None
            if cgrad:
                loc_sol = conjgrad(matrix, vector)
            else:
                loc_sol = direct_solver(matrix, vector)

            if dmrg_sites == 2:
                x[ix], x[ix + 1] = vector_to_2nodes(
                    vector=loc_sol,
                    lbd=rl[ix],
                    rbd=rl[ix + 2],
                    pd=pd,
                    max_bd=max_rank,
                    boundary1=boundary1,
                    boundary2=boundary2,
                )
            elif dmrg_sites == 1:
                if boundary1 is None and boundary2 is None:
                    boundary = None
                elif boundary2 is None:
                    boundary = boundary1
                else:
                    boundary = boundary2
                x[ix] = vector_to_node(
                    vector=loc_sol,
                    pd=pd,
                    lbd=rl[ix],
                    rbd=rl[ix + 1],
                    boundary=boundary,
                )

            error = (A.apply(x) + b * -1.0).norm / b_norm
            history.append(error)
        if history[-1] < eps:
            break

    if verbose:
        return x, history
    return x


# def linsystem_solver_2(
#     A: MPO,
#     b: MPS,
#     x: MPS = None,
#     max_rank: MPS = 10,
#     max_iter: int = 2,
#     eps: float = 1e-2,
#     dmrg_sites: int = 2,
#     cgrad=False,
#     verbose=False,
# ):
#     raise NotImplementedError()
#     pd = A[0].shape[1]
#     assert dmrg_sites < 3, f"Performing DMRG with {dmrg_sites} sites is not supported"
#     var = 0.2
#     if x is None:
#         x = random_mps(L=len(b), pd=pd, bd=max_rank, var=var)

#     x.left_canonical()

#     history = []
#     b_norm = b.norm
#     error = (A.apply(x) + b * -1.0).norm / b_norm
#     history.append(error)

#     for iter in range(max_iter):
#         left_env_tensor = None
#         right_env_tensor = None
#         for ix in range(len(A) - dmrg_sites + 1):
#             if iter % 2 != 0:
#                 ix = len(A) - ix - dmrg_sites

#             arrays1, arrays2 = get_arrays(A, b, x, list(range(ix, ix + dmrg_sites)))
#             vector, matrix = get_local_system(
#                 arrays1, arrays2, left_env_tensor, right_env_tensor, index=ix
#             )

#             n1 = int(math.sqrt(matrix.numel()))
#             matrix = matrix.reshape(n1, n1)
#             rl = x.ranks_list

#             if ix == 0:
#                 boundary1 = "left"
#                 boundary2 = None
#             elif ix == len(A) - 2:
#                 boundary1 = None
#                 boundary2 = "right"
#             else:
#                 boundary1 = None
#                 boundary2 = None
#             if cgrad:
#                 loc_sol = conjgrad(matrix, vector)
#             else:
#                 loc_sol = direct_solver(matrix, vector)

#             if dmrg_sites == 2:
#                 x[ix], x[ix + 1] = vector_to_2nodes(
#                     vector=loc_sol,
#                     lbd=rl[ix],
#                     rbd=rl[ix + 2],
#                     pd=pd,
#                     max_bd=max_rank,
#                     boundary1=boundary1,
#                     boundary2=boundary2,
#                 )
#             elif dmrg_sites == 1:
#                 if boundary1 is None and boundary2 is None:
#                     boundary = None
#                 elif boundary2 is None:
#                     boundary = boundary1
#                 else:
#                     boundary = boundary2
#                 x[ix] = vector_to_node(
#                     vector=loc_sol,
#                     pd=pd,
#                     lbd=rl[ix],
#                     rbd=rl[ix + 1],
#                     boundary=boundary,
#                 )

#             error = (A.apply(x) + b * -1.0).norm / b_norm
#             history.append(error)
#         if history[-1] < eps:
#             break

#     if verbose:
#         return x, history
#     return x


def ground_state_solver(
    A: MPO,
    x: MPS = None,
    max_rank: MPS = 2,
    max_iter: int = 4,
    dmrg_sites: int = 2,
    maximize=False,
    reverse=False,
    noisy_convergence=False,
    noise_rank=2,
):
    pd = A[0].shape[1]
    var = 0.2
    if x is None:
        x = random_mps(L=len(A), pd=pd, bd=max_rank, var=var)
        x.compress()

    for iter in range(max_iter):
        for ix in range(len(A) - dmrg_sites + 1):
            if iter % 2 != int(reverse):
                ix = len(A) - ix - dmrg_sites
            x.mixed_canonical(ix)
            skip_index = list(range(ix, ix + dmrg_sites))
            matrix = A.partial_inner_prod(x, skip_indices=skip_index)
            rl = x.ranks_list
            if ix == 0:
                boundary1 = "left"
                boundary2 = None
            elif ix == len(A) - 2:
                boundary1 = None
                boundary2 = "right"
            else:
                boundary1 = None
                boundary2 = None
            _, loc_sol = eigen_solver(matrix, maximize=maximize)
            if dmrg_sites == 2:
                x[ix], x[ix + 1] = vector_to_2nodes(
                    vector=loc_sol,
                    lbd=rl[ix],
                    rbd=rl[ix + 2],
                    pd=pd,
                    max_bd=max_rank,
                    boundary1=boundary1,
                    boundary2=boundary2,
                )
            elif dmrg_sites == 1:
                if boundary1 is None and boundary2 is None:
                    boundary = None
                elif boundary2 is None:
                    boundary = boundary1
                else:
                    boundary = boundary2
                x[ix] = vector_to_node(
                    vector=loc_sol,
                    pd=pd,
                    lbd=rl[ix],
                    rbd=rl[ix + 1],
                    boundary=boundary,
                )
        if noisy_convergence and (iter != max_iter - 1):
            x = x + random_mps(len(x), noise_rank, var=var / 2)

    gse = A.inner_prod(x) / (x.norm**2)

    return gse, x


def solve_inv(
    A2: MPO, max_rank, max_iter, opt_paths=True, dmrg_sites: int = 2, verbose=False
):
    A = []
    for i, j in zip(A2, A2.transpose()):
        A.append(MPO_Node(torch.kron(i.tensor, i.tensor)))
    A = MPO(A)
    b = A2.transpose().mpo_to_mps()
    pd = A[0].shape[1]
    L = len(A2)
    assert dmrg_sites < 3, f"Performing DMRG with {dmrg_sites} sites is not supported"
    x = random_mps(L=len(b), bd=20, pd=4, var="shape")
    x.mixed_canonical(len(x) // 2)
    history = []
    error = (mps_to_mpo(x) @ A2 + identity_mpo(L) * -1.0).norm
    history.append(error)
    sol = x.copy()
    A_shapes = [node.shape for node in A]
    b_shapes = [node.shape for node in b]
    x_shapes = [node.shape for node in x]
    _DMRG_PATHS_CACHE = {}

    key = str(
        abs(hash((tuple(A_shapes), tuple(b_shapes), tuple(x_shapes), dmrg_sites)))
    )

    if opt_paths:
        if key in _DMRG_PATHS_CACHE.keys():
            _DMRG_PATHS_CACHE.pop(key)

    try:
        opt_paths1, opt_paths2 = _DMRG_PATHS_CACHE[key]
    except KeyError:
        opt_paths1, opt_paths2 = get_opt_paths(A_shapes, b_shapes, x_shapes, dmrg_sites)
        _DMRG_PATHS_CACHE[key] = (opt_paths1, opt_paths2)

    for iter in range(max_iter):
        print("Iteration: " + str(iter))
        for ix in range(len(A) - dmrg_sites + 1):
            print("Site: " + str(ix))
            if iter % 2 != 0:
                ix = len(A) - ix - dmrg_sites
            b.mixed_canonical(ix)
            x.mixed_canonical(ix)
            arrays1, arrays2 = get_arrays_einsum(
                A, b, x, list(range(ix, ix + dmrg_sites))
            )
            vector = opt_paths1[ix](*arrays1).flatten()
            matrix = opt_paths2[ix](*arrays2).flatten()
            n1 = int(math.sqrt(matrix.numel()))
            matrix = matrix.reshape(n1, n1)
            rl = x.ranks_list
            if ix == 0:
                boundary1 = "left"
                boundary2 = None
            elif ix == len(A) - 2:
                boundary1 = None
                boundary2 = "right"
            else:
                boundary1 = None
                boundary2 = None

            loc_sol = direct_solver(matrix, vector)

            x[ix], x[ix + 1] = vector_to_2nodes(
                vector=loc_sol,
                lbd=rl[ix],
                rbd=rl[ix + 2],
                pd=pd,
                max_bd=max_rank,
                boundary1=boundary1,
                boundary2=boundary2,
            )

            error = (mps_to_mpo(x) @ A2 + identity_mpo(L) * -1.0).norm
            print(error)
            if error < min(history):
                sol = x.copy()
                print("Saved best inv!")
            history.append(error)
    if verbose:
        return mps_to_mpo(sol), history
    else:
        return mps_to_mpo(sol)


def get_inv_diag(mpo: MPO):
    a = float(mpo.get_elem([0, 0]))
    inv_diag = identity_mpo(L=len(mpo)) * (1 / a)
    return inv_diag


def get_diag(mpo: MPO):
    a = float(mpo.get_elem([0, 0]))
    diag = identity_mpo(L=len(mpo)) * a
    return diag


def jacobi_iter(
    mpo: MPO = None,
    mps: MPS = None,
    eps: float = 1e-4,
    max_iter: int = 150,
    verbose: bool = False,
):
    history = []
    sol_mps = random_mps(len(mps), bd=3)
    d_inv = get_inv_diag(mpo)
    B = mpo + get_diag(mpo) * (-1.0)
    B.compress()
    for _ in range(max_iter):
        cur_mps = d_inv.apply(B.apply(sol_mps) + mps)
        error = cur_mps + sol_mps * (-1.0)
        if error.norm < eps:
            break
        elif error.norm >= 1e8:
            raise ValueError("Divergence observed during jacobi iteration.")
        history.append(error.norm)
        cur_mps.compress()
        sol_mps = cur_mps.copy()
    if verbose:
        return sol_mps, history
    return sol_mps


def iterative_inversion(
    mpo: MPO = None,
    alpha: float = 1e-8,
    errorrate: float = 1,
    eps: float = 1e-4,
    max_iter: int = 150,
    order: str = "low",
    opt_flag: bool = False,
    accuracy=None,
    max_bd=None,
    verbose: bool = False,
):
    if opt_flag or not hasattr(iterative_inversion, "cache"):
        history = []
        inv_mpo = mpo.copy() * alpha + identity_mpo(len(mpo))
        id_mpo = identity_mpo(len(mpo))
        error = inv_mpo.apply(mpo) + id_mpo * (-1.0)
        error = error.mpo_to_mps().norm
        history.append(error)
        if order == "low":
            for k in range(max_iter):
                psi = inv_mpo.apply(mpo)
                cur_mpo = inv_mpo * 2.0 + psi.apply(inv_mpo) * -1.0
                error = cur_mpo.apply(mpo) + id_mpo * (-1.0)
                error = error.mpo_to_mps().norm
                if (error < eps) or (error > errorrate * history[-1]):
                    history.append(error)
                    break
                history.append(error)
                cur_mpo.compress()
                inv_mpo = cur_mpo.copy()

        elif order == "high":
            for k in range(max_iter):
                psi = mpo.apply(inv_mpo, accuracy, max_bd)
                cur_mpo = psi.apply(psi, accuracy, max_bd)
                cur_mpo = psi.apply(cur_mpo, accuracy, max_bd) * -1.0
                cur_mpo = cur_mpo + psi.apply(psi, accuracy, max_bd) * 4.0
                cur_mpo = cur_mpo + psi * -6.0 + id_mpo * 4.0
                cur_mpo = inv_mpo.apply(cur_mpo, accuracy, max_bd)

                error = cur_mpo.apply(mpo) + id_mpo * (-1.0)
                error = error.mpo_to_mps().norm

                if (error < eps) or (error > errorrate * history[-1]):
                    history.append(error)
                    break
                history.append(error)
                inv_mpo = cur_mpo.copy()

        history.pop(0)

        iterative_inversion.cache = (inv_mpo, history)

        if verbose:
            return inv_mpo, history
        return inv_mpo
    else:
        if verbose:
            return iterative_inversion.cache
        return iterative_inversion.cache[0]
