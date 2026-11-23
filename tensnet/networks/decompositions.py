import concurrent

import torch

from ..utils import qr, svd
from .base_mpo import mps_to_mpo
from .base_mps import list_to_mps


def _tensors_from_sv(
    sv,
    d: int,
    L: int,
    eps: float = None,
    bdmax: int = None,
    naive_svd=True,
):
    """Decomposes a state vector into a tensor train (TT) format using singular value decomposition (SVD) or QR decomposition.

    Args:
        sv (torch.Tensor): The state vector to decompose. Must have a size of d**L.
        d (int): The dimension of each local tensor.
        L (int): The number of local tensors in the TT.
        eps (float, optional): Threshold for singular values. Singular values below this threshold are discarded. Defaults to None.
        bdmax (int, optional): Maximum bond dimension. If the bond dimension exceeds this value, it is truncated. Defaults to None.
        naive_svd (bool, optional): If True, uses a naive SVD approach. If False, uses QR decomposition followed by SVD. Defaults to True.

    Returns:
        list[torch.Tensor]: A list of torch.Tensors representing the TT cores. Each tensor has shape (bond_dim_left, d, bond_dim_right),
        except for the first and last tensors, which have bond_dim_left=1 and bond_dim_right=1, respectively.

    Raises:
        AssertionError: If the size of the state vector does not match the dimensions (d**L).
    """

    assert torch.numel(sv) == d**L, "Size of the state vector does not match the dimensions"
    temp = torch.clone(sv)
    if naive_svd:
        S = torch.ones(1, dtype=temp.dtype, device=temp.device)
        A = []
        for i in range(1, L):
            U, S, V = svd(temp.reshape([d * len(S), d ** (L - i)]), eps, bdmax)
            A.append(U.reshape(-1, d, len(S)))
            temp = S.type(temp.dtype).unsqueeze(1) * V
        A.append(temp.reshape(len(S), d, 1))
        return A
    else:
        A = []
        W = temp.reshape(-1, d)
        for _i in range(L - 1, 0, -1):
            Q, R = qr(W)
            U, S, V = svd(R, eps, bdmax)
            A.append(V.reshape(V.shape[0], d, -1))
            W = (W @ V.T).reshape(-1, d * A[-1].shape[0])
        A.append(W.reshape(1, d, -1))
        return A[::-1]


# SLIGHTLY FASTER IMPLEMENATION WHEN BD NOT GIVEN
def _tensors_from_sv2(
    sv,
    d: int,
    L: int,
    eps: float = None,
    bdmax: int = None,
):
    vec = sv.reshape([d] * L)
    U_list = []
    S_list = []
    V_list = []

    def process(i):
        vec_i = vec.reshape(d**i, -1)
        U, S, V = svd(vec_i, eps, bdmax)
        return U, S, V

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
    G_list.append((S_list[-1].unsqueeze(1) * V_list[-1]).reshape(-1, d, 1))
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


_MALS_PATHS_CACHE = {}


# TODO Implement the TT-MALS algorithm to compress MPS using DMRG
# _EINSUM_PATHS_CACHE = {}


# def tt_mals_2(
#     vector,
#     nofq,
#     pd=2,
#     mps_init=None,
#     init_rank=None,
#     dmrg_sites=2,
#     max_sweeps=3,
#     max_rank=None,
#     verbose=False,
#     opt_paths=True,
# ):
#     assert (
#         torch.numel(vector) == pd**nofq
#     ), "Size of the vector does not match the dimensions"
#     tensor = torch.reshape(vector, [pd] * nofq)

#     if mps_init is None and init_rank is None:
#         raise ValueError(
#             "Both the initialization of MPS and initialization rank cannot be None at the same time. Provide at least one of the arguments."
#         )

#     if mps_init is None:
#         mps_init = random_mps(nofq, init_rank, pd=pd)
#     nodes = mps_init.copy().nodes
#     b_shapes = [node.shape for node in nodes]

#     key = str(abs(hash((pd, nofq, tuple(b_shapes), dmrg_sites))))

#     if opt_paths:
#         if key in _EINSUM_PATHS_CACHE.keys():
#             _EINSUM_PATHS_CACHE.pop(key)

#     try:
#         paths = _EINSUM_PATHS_CACHE[key]
#     except KeyError:
#         paths = []
#         for _ in len(b_shapes):
#             paths.append(compute_best_path())
#         _EINSUM_PATHS_CACHE[key] = paths

#     for i in range(max_sweeps):
#         for ix in range(nofq - dmrg_sites + 1):
#             if i % 2 != 0:
#                 ix = nofq - ix - dmrg_sites
#     return None
