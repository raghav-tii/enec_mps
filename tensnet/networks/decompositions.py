from typing import Callable

import tntorch as tnt
import torch

from .base_mpo import identity_mpo, mps_to_mpo
from ..utils import COMP_ACC, svd, qr, get_arrays_einsum, get_best_path
from .base_mps import MPS, MPS_Node, list_to_mps, random_mps
import concurrent


def _tensors_from_sv(
    sv, d: int, L: int, eps: float = None, bdmax: int = None, naive_svd=True
):
    assert (
        torch.numel(sv) == d**L
    ), "Size of the State Vector does not match the dimensions"
    temp = torch.clone(sv)
    if naive_svd:
        S = torch.ones(1)
        A = []
        for i in range(1, L):
            U, S, V = svd(temp.reshape([d * len(S), d ** (L - i)]), eps, bdmax)
            A.append(U.reshape(-1, d, len(S)))
            temp = torch.diag(S) @ V
        A.append(temp.reshape(len(S), d, 1))
        return A
    else:
        A = []
        W = temp.reshape(-1, d)
        for i in range(L - 1, 0, -1):
            Q, R = qr(W)
            U, S, V = svd(R, eps, bdmax)
            A.append(V.reshape(V.shape[0], d, -1))
            W = (W @ V.T).reshape(-1, d * A[-1].shape[0])
        A.append(W.reshape(1, d, -1))
        return A[::-1]


# SLIGHTLY FASTER IMPLEMENATION WHEN BD NOT GIVEN
def _tensors_from_sv2(sv, d: int, L: int, eps: float = None, bdmax: int = None):
    vec = sv.reshape([d] * L)
    U_list = []
    S_list = []
    V_list = []

    def process(i):
        vec_i = vec.reshape(d**i, -1)
        U, S, V = svd(vec_i, eps, bdmax)
        return U, torch.diag(S), V

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process, range(1, L)))
    for U, S, V in results:
        U_list.append(U)
        S_list.append(S)
        V_list.append(V)
    G_list = [U_list[0].reshape(1, d, -1)]
    for j in range(0, L - 2):
        Uk = U_list[j]
        Uk1 = U_list[j + 1]
        W_k1 = Uk.T @ Uk1.reshape(d ** (j + 1), -1)
        G_list.append(W_k1.reshape(-1, d, int(W_k1.shape[1] / d)))
    G_list.append((S_list[-1] @ V_list[-1]).reshape(-1, d, 1))
    return G_list


def vector_to_mps(
    vec,
    d: int,
    N: int,
    eps: float = None,
    bdmax: int = None,
    naive_svd=True,
    accelerate=False,
):
    """
    Generates a MPS encoding of the vector using sequential SVD decompositions.


    Args:
        vec (tensor): 1D array of size (d^N)
        d (int): The local dimension of each MPS tensor/site.
        N (int): The total number of MPS sites
        eps (float, optional): SVD cutoff threshold for truncation of singular values in l2 norm. Defaults to None.
        bdmax (int, optional): Additional threshold maximum number of singular values to keep during truncation. Defaults to None.

    Returns:
        mps: The MPS encoding of vector
    """
    if bdmax is None and accelerate:
        return list_to_mps(_tensors_from_sv2(vec, d, N, eps, bdmax))
    else:
        return list_to_mps(_tensors_from_sv(vec, d, N, eps, bdmax, naive_svd))


def matrix_to_mpo(matrix, d: int, L: int, eps: float = None, bdmax: int = None):
    new_axis = []
    for x in range(0, L):
        new_axis += [x, x + L]
    vec = matrix.flatten().reshape([d] * 2 * L).clone().detach().permute(new_axis)
    return mps_to_mpo(vector_to_mps(vec, d**2, L, eps, bdmax)).transpose()


def vector_to_node(vector, lbd: int, rbd: int, pd: int = 2, boundary=None):
    tensor = vector.reshape(lbd, pd, rbd)
    return MPS_Node(tensor, boundary=boundary)


def vector_to_2nodes(
    vector, lbd: int, rbd: int, pd: int = 2, max_bd=None, boundary1=None, boundary2=None
):
    matrix = vector.reshape(lbd * pd, pd * rbd)
    U, S, V = svd(matrix, eps=COMP_ACC(), bdmax=max_bd)
    tensor1 = U
    tensor2 = torch.diag(S) @ V
    return MPS_Node(
        tensor1.reshape(lbd, pd, len(S)), boundary=boundary1, canonical="left"
    ), MPS_Node(tensor2.reshape(len(S), pd, rbd), boundary=boundary2)


def cross_approximation(
    function: Callable,
    nofq: int,
    pd: int,
    mps=None,
    eps: float = 1e-6,
    max_rank: int = None,
    max_iter: int = 15,
    val_size: int = 5e3,
    verbose: bool = False,
    suppress_warnings=True,
    **kwargs,
):
    if mps is None:
        domain = [torch.tensor(range(pd))] * nofq
        mps_torch = tnt.cross(
            function,
            domain=domain,
            eps=eps,
            max_iter=max_iter,
            verbose=verbose,
            val_size=val_size,
            rmax=max_rank,
            suppress_warnings=suppress_warnings,
            **kwargs,
        )
    else:
        tensors = tnt.Tensor([node.tensor for node in mps])
        mps_torch = tnt.cross(
            function,
            tensors=tensors,
            eps=eps,
            max_iter=max_iter,
            verbose=verbose,
            val_size=val_size,
            rmax=max_rank,
            **kwargs,
        )
    nodes = []
    for tensor in mps_torch.cores:
        nodes.append(MPS_Node(tensor))
    return MPS(nodes)


def tt_cross(
    function: Callable,
    nofq: int,
    pd: int = 2,
    eps: float = None,
    max_iter: int = None,
    max_rank: int = None,
    verbose: bool = False,
):
    try:
        from teneva import cross
    except ModuleNotFoundError as e:
        print(f"{e}:teneva module not found. Please use cross_approximation function.")
    if max_rank is None:
        max_rank = 0
    Y0 = [node.tensor.numpy() for node in random_mps(nofq, pd=pd, bd=max(1, max_rank))]
    info = {}
    cores = cross(function, Y0, nswp=max_iter, e=eps, info=info, log=verbose)
    mps = list_to_mps(cores)
    print(info)
    return mps


def get_opt_paths(A_shapes, b_shapes, x_shapes, dmrg_sites):
    opt_paths_ip = []
    opt_paths_opip = []
    for ix in range(len(A_shapes)):
        path_ip, path_opip = get_best_path(
            A_shapes, b_shapes, x_shapes, list(range(ix, ix + dmrg_sites))
        )
        opt_paths_ip.append(path_ip)
        opt_paths_opip.append(path_opip)
    return opt_paths_ip, opt_paths_opip


_MALS_PATHS_CACHE = {}


def tt_mals(
    b: MPS,
    init_mps: MPS = None,
    max_rank: MPS = 10,
    nsweeps: int = 3,
    eps: float = 1e-3,
    dmrg_sites: int = 2,
    opt_paths=True,
    verbose=False,
):
    pd = b[0].shape[1]
    L = len(b)
    assert dmrg_sites < 3, f"Performing MALS with {dmrg_sites} sites is not supported"
    var = 0.2
    if init_mps is None:
        x = random_mps(L, pd=pd, bd=max_rank, var=var)
    else:
        x = init_mps.copy()

    x.left_canonical()
    history = []
    b_norm = b.norm
    error = (x + b * -1.0).norm / b_norm
    history.append(error)
    A = identity_mpo(L)
    A_shapes = [node.shape for node in A]
    b_shapes = [node.shape for node in b]
    x_shapes = [node.shape for node in x]

    key = str(abs(hash((tuple(b_shapes), tuple(x_shapes), dmrg_sites))))

    if opt_paths:
        if key in _MALS_PATHS_CACHE.keys():
            _MALS_PATHS_CACHE.pop(key)

    try:
        opt_paths1 = _MALS_PATHS_CACHE[key]
    except KeyError:
        opt_paths1, _ = get_opt_paths(A_shapes, b_shapes, x_shapes, dmrg_sites)
        _MALS_PATHS_CACHE[key] = opt_paths1

    for iter in range(nsweeps):
        for ix in range(L - dmrg_sites + 1):
            if iter % 2 != 0:
                ix = L - ix - dmrg_sites
            b.mixed_canonical(ix)
            x.mixed_canonical(ix)
            arrays1, _ = get_arrays_einsum(A, b, x, list(range(ix, ix + dmrg_sites)))
            vector = opt_paths1[ix](*arrays1).flatten()
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

            if dmrg_sites == 2:
                x[ix], x[ix + 1] = vector_to_2nodes(
                    vector=vector,
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
                    vector=vector,
                    pd=pd,
                    lbd=rl[ix],
                    rbd=rl[ix + 1],
                    boundary=boundary,
                )

            error = (x + b * -1.0).norm / b_norm
            history.append(error)
        if history[-1] < eps:
            break

    if verbose:
        return x, history
    return x
