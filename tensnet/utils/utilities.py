import functools
import pickle
from typing import Callable
from warnings import warn

import opt_einsum as oe
import torch

from .config import ARCH_2D, parameters, device


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


# @torch.compile(dynamic=True)
def svd(M: torch.Tensor, eps: float = None, bdmax: int = None):
    U, S, V = torch.linalg.svd(M, full_matrices=False)
    if sum(S**2) != 0.0:
        S_n = S / torch.sqrt(sum(S**2))
    else:
        S_n = S
    if eps is not None:
        trunc = sum(torch.cumsum(torch.flip(S_n, dims=(0,)) ** 2, dim=0) >= eps)
        trunc = max(1, trunc)
        bd = min(trunc, bdmax) if bdmax is not None else trunc
    else:
        bd = len(S) if bdmax is None else min(len(S), bdmax)
    return U[:, :bd], S[:bd], V[:bd, :]



# @torch.compile(dynamic=True)
def qr(tensor):
    return torch.linalg.qr(tensor, "reduced")


def rq(tensor):
    Q, R = qr(
        (torch.flip(torch.eye(tensor.shape[0], device=device()), (0,)) @ tensor).T
    )
    Q1 = torch.flip(torch.eye(Q.T.shape[0], device=device()), (0,)) @ Q.T
    R1 = (
        torch.flip(torch.eye(R.T.shape[0], device=device()), (0,))
        @ R.T
        @ torch.flip(torch.eye(R.T.shape[1], device=device()), (0,))
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
        x = torch.ones_like(b)
    r = b - A @ x
    p = r
    rsold = r.T @ r

    for i in range(len(b)):
        Ap = A @ p
        alpha = rsold / (p.T @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = r.T @ r
        if torch.sqrt(rsnew) < 1e-8:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x


# @torch.compile(dynamic=True)
def direct_solver(A, b):
    """
    A function to solve [A]{x} = {b} linear equation system with the
    built-in torch method.
    ========== Parameters ==========
    A : matrix
    b : vector
    """
    x = torch.linalg.solve(A, b)
    return x


# @torch.compile(dynamic=True)
def eigen_solver(A, k=1, maximize=False):
    """
    A function to solve [A]{x} = l{x} eigenvalue problem with the
    built-in method. Redirects to a 'LAPACK' call.
    ========== Parameters ==========
    A : matrix
    b : vector
    """

    if maximize:
        A = -1.0 * A
    lmd, x = torch.linalg.eigh(A)
    if maximize:
        lmd, x = -lmd, -x
    if k == 1:
        return torch.unsqueeze(lmd[0], 0), torch.unsqueeze(x[:, 0], 1)
    else:
        return lmd[:k], x[:, :k]


# @torch.compile(dynamic=True)
def schmidt_coeffs(M: torch.Tensor):
    U, S, V = torch.linalg.svd(M, full_matrices=False)
    S_n = S * (1.0 / torch.sqrt(torch.sum(S**2)))
    return S_n


def nested_list_to_tensor(nested_list):
    if isinstance(nested_list, list):
        return torch.stack([nested_list_to_tensor(item) for item in nested_list])
    elif torch.is_tensor(nested_list):
        return torch.clone(nested_list)
    else:
        return torch.tensor(nested_list)


def validate_tracks(tracks):
    for m in tracks:
        for node in m:
            if node.ndim == 3:
                node.reshape(node.dim + [1])
    return tracks


def validate_nodes(nodes):
    if nodes[0].boundary is None:
        nodes[0].boundary = "left"
    if nodes[-1].boundary is None:
        nodes[-1].boundary = "right"
    for node in nodes[1:-1]:
        node.boundary = None
    return nodes


def cal_padding(shape1, shape2, pad_axis=[0]):
    pad1, pad2 = [], []
    for ix, (x, y) in enumerate(zip(shape1, shape2)):
        if ix in pad_axis:
            pad1.append((0, y))
            pad2.append((x, 0))
        else:
            pad1.append((0, 0))
            pad2.append((0, 0))
    return pad1, pad2


def val_addability(self, other, p_axis=1):
    assert self.order == other.order, (
        "Only tensors of same order can be added."
        + "Received {} and {} -order tensors".format(self.order, other.order)
    )
    assert (
        self.dim[p_axis] == other.dim[p_axis]
    ), "Failed to add coz of mismatched shape on the physical leg(s)"

    return (self.order == other.order) and self.dim[p_axis] == other.dim[p_axis]


def gen_state(d: int, L: int):
    warn("Caution while calling for this. It will use exponentially large memory.")
    torch.manual_seed(42)
    return torch.randn(d**L)


def get_local_inds(int_inds, nofq, base=2):
    digits = []
    num = int_inds if isinstance(int_inds, int) else int_inds.item()
    while num:
        digits.append(int(num % base))
        num //= base
    while len(digits) != nofq:
        digits.append(0)
    return digits[::-1]


def discretize(f: Callable, mesh: torch.Tensor):
    function_values = f(mesh)
    return function_values


def unshuffle_axis(A: torch.Tensor, L: int, d: int):
    # tensorize A and unshuffle odd and even axis and return a matrix
    axis_even_odd = [x for x in range(2 * L) if x % 2 == 1] + [
        x for x in range(2 * L) if x % 2 == 0
    ]
    A = A.reshape([d] * 2 * L).permute(axis_even_odd).reshape(d**L, d**L)
    return A


def shuffle_axis(A: torch.Tensor, L: int):
    # tensorize A and shuffle odd and even axis and return a matrix
    new_axis = []
    for x in range(L):
        new_axis += [x, x + L]
    A = A.reshape([2] * 2 * L).permute(new_axis).reshape(2**L, 2**L)
    return A


def reflect_yaxis(A, L: int):
    x_reflect_y = [0] + list(range(1, L + 1))[::-1]
    A = A.reshape([2**L] + [2] * L).permute(x_reflect_y).reshape(2**L, 2**L)
    return A


def reflect_xaxis(A, L: int):
    reflect_x_y = list(range(L))[::-1] + [L + 1]
    A = A.reshape([2] * L + [2**L]).permute(reflect_x_y).reshape(2**L, 2**L)
    return A


# Decorators


def qubits_to_coordinates(qubits, L, axis=None):
    if axis is None:
        axis = ""
    a = parameters(f"a{axis}")
    b = parameters(f"b{axis}")
    pows = 2 ** torch.arange(L - 1, -1, -1).reshape(L, 1) / 2**L
    ints = (qubits * pows).sum(0)
    ints *= b - a
    ints += a
    return ints


def modify_input(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        flag = torch.is_tensor(args[0])
        if not flag:
            args = torch.Tensor(args[0]).permute(1, 0)
        else:
            args = torch.stack(args)
        L = int(args.shape[0])
        x = qubits_to_coordinates(args, L)
        if not flag:
            x = x.numpy()
        val = function(x, **kwargs)
        return val

    return wrapper


def modify_2inputs(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        L1 = int(len(args) / 2)
        if "Lx" in parameters().keys() and "Ly" in parameters().keys():
            Lx = parameters("Lx")
            Ly = parameters("Ly")
        else:
            Lx = L1
            Ly = L1
        args = torch.stack(args)
        if ARCH_2D() == "stacked":
            args_x = args[:Lx, :]
            args_y = args[Lx:, :]
        elif ARCH_2D() == "sequential":
            assert (
                Lx == Ly
            ), "For sequential architecture, only equal no of qubits in x and y are supported."
            args_x = args[: Lx + Ly : 2, :]
            args_y = args[1 : Lx + Ly : 2, :]
        elif ARCH_2D() == "xyrev":
            args_x = args_x[:Lx, :]
            args_y = torch.flip(args[Lx:, :], (0,))
        elif ARCH_2D() == "xrevy":
            args_x = torch.flip(args_x[:Lx, :], (0,))
            args_y = args[Lx:, :]
        x = qubits_to_coordinates(args_x, Lx, axis="x")
        y = qubits_to_coordinates(args_y, Ly, axis="y")
        val = function(x, y, **kwargs)
        return val

    return wrapper


# Network Contraction Utils


def compute_best_path(pattern: str, shapes: list):
    try:
        from cotengra import ReusableHyperOptimizer

        opt = ReusableHyperOptimizer(
            minimize="combo",
            methods=["kahypar", "greedy"],
            max_repeats=32,
            # progbar=True,
            directory="cotengra_cache",
        )
    except ModuleNotFoundError:
        opt = "auto-hq"
    path = oe.contract_expression(pattern, *shapes, optimize=opt)

    return path


def get_best_path(A_shapes, b_shapes, x_shapes, site_index):
    ip_pattern = einpat_part_ip(len(A_shapes), tuple(site_index))
    opip_pattern = einpat_part_opip(len(A_shapes), tuple(site_index))
    shapes_ip, shapes_opip = get_shapes_einsum(A_shapes, b_shapes, x_shapes, site_index)

    path_ip = compute_best_path(ip_pattern, shapes_ip)
    path_opip = compute_best_path(opip_pattern, shapes_opip)

    return path_ip, path_opip


@functools.lru_cache()
def einpat_part_ip(L, site_index):
    part_ip_path = ""
    output_path = "->"
    for i in range(L):
        j = 3 * (i + 1)
        ll, ul, m, lr, ur = (oe.get_symbol(i) for i in (j - 3, j - 2, j - 1, j, j + 1))

        if i in site_index:
            part_ip_path += "{}{}{},".format(ll, m, lr)
            output_path += "{}{}{}".format(ul, m, ur)
        else:
            part_ip_path += "{}{}{},{}{}{},".format(ll, m, lr, ul, m, ur)

        if i == 0:
            output_path += "{}{}".format(ll, ul)
        if i == L - 1:
            output_path += "{}{}".format(ur, lr)
    for c in set(output_path):
        if output_path.count(f"{c}") > 1:
            output_path = output_path.replace(f"{c}", "")
    pattern = part_ip_path[:-1] + output_path

    return pattern


@functools.lru_cache()
def einpat_part_opip(L, site_index):
    part_opip_path = ""
    path_connector = "->"
    output_path = ""
    for i in range(L):
        j = 5 * (i + 1)
        ll, ml, ul, lm, um, lr, mr, ur = (
            oe.get_symbol(i)
            for i in (j - 5, j - 4, j - 3, j - 2, j - 1, j, j + 1, j + 2)
        )

        if i in site_index:
            part_opip_path += "{}{}{}{},".format(ml, lm, mr, um)
            output_path += "{}{}{}".format(ll, lm, lr)
            path_connector += "{}{}{}".format(ul, um, ur)
        else:
            part_opip_path += "{}{}{},{}{}{}{},{}{}{},".format(
                ll, lm, lr, ml, lm, mr, um, ul, um, ur
            )

        if i == 0:
            output_path += "{}{}{}".format(ll, ml, ul)
        if i == L - 1:
            path_connector += "{}{}{}".format(lr, mr, ur)
    path_connector += output_path
    for c in set(path_connector):
        if path_connector.count(f"{c}") > 1:
            path_connector = path_connector.replace(f"{c}", "")
    pattern = part_opip_path[:-1] + path_connector

    return pattern


def get_arrays_einsum(A, b, x, site_index):
    arrays_ip = []
    arrays_opip = []

    for i in range(len(A)):
        if i in site_index:
            arrays_ip.append(b[i].tensor)
            arrays_opip.append(A[i].tensor)
        else:
            arrays_opip.append(x[i].tensor)
            arrays_opip.append(A[i].tensor)
            arrays_opip.append(x[i].tensor)
            arrays_ip.append(b[i].tensor)
            arrays_ip.append(x[i].tensor)

    return arrays_ip, arrays_opip


def get_shapes_einsum(A_shapes, b_shapes, x_shapes, site_index):
    shapes_ip = []
    shapes_opip = []

    for i in range(len(A_shapes)):
        if i in site_index:
            shapes_ip.append(b_shapes[i])
            shapes_opip.append(A_shapes[i])
        else:
            shapes_ip.append(b_shapes[i])
            shapes_ip.append(x_shapes[i])
            shapes_opip.append(x_shapes[i])
            shapes_opip.append(A_shapes[i])
            shapes_opip.append(x_shapes[i])

    return shapes_ip, shapes_opip
