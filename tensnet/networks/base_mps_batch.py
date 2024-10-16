import torch
from .base_mps import MPS, MPS_Node
from einops import einsum, rearrange
from typing import Union


class MPS_batch(MPS):
    """
    An MPS_batch object stores a batch (list) of MPSs. Each MPS is a list of MPS tensors, of shape T(l,d,r) with l and r being the bonds and d being the physical dimension.
    """

    # INITIALIZE MPS BATCH AND CANONICAL FORM (left or right)
    def __init__(self, mps_batch: list[MPS] = None, canonical: str = "None"):
        self.mps_batch = mps_batch
        self.canonical = canonical

    # REPRESENTATION OF MPS BATCH
    def __repr__(self):
        main_str = f"{type(self)} | Batch Size: {len(self.mps_batch)} | Length of MPS: {len(self.mps_batch[0])} | \n\n"
        str1 = ""
        str2 = ""
        str3 = ""
        for i in self.mps_batch[0].shapes():
            str1 = str1 + str(i[0]) + "--T--"
            str2 = str2 + " " * len(str(i[0])) + "  |  "
            str3 = str3 + " " * len(str(i[0])) + "  " + str(i[1]) + "  "
        str1 = str1 + str(self.mps_batch[0].shapes()[-1][-1])
        return main_str + str1 + "\n" + str2 + "\n" + str3

    # RETURN A SPECIFIC MPS (specified by index running from 0 to batch_size - 1)
    def __getitem__(self, i):
        return self.mps_batch[i]

    # RETURN BATCH SIZE OF MPS BATCH
    def __len__(self):
        return len(self.mps_batch)

    # RETURN A DEEP COPY OF MPS BATCH
    def copy(self):
        new_batch = []
        for mps in self.mps_batch:
            new_mps = mps.copy()
            new_batch.append(new_mps)
        return MPS_batch(new_batch, self.canonical)

    # ADD TWO MPS BATCHES
    def __add__(self, other: "MPS_batch"):
        assert len(self) == len(other)
        new_mps_batch = []
        for i, j in zip(self.mps_batch, other.mps_batch):
            new_mps_batch.append(i + j)

        return MPS_batch(new_mps_batch)

    # MULTIPLY MPS BATCH WITH MPS BATCH/ VALUE
    def __mul__(self, other: Union[float, int, "MPS_batch"]):
        new_mps_batch = []
        # NEED TO OPTIMIZE
        if isinstance(other,MPS_batch):
            assert len(self) == len(other)
            for i, j in zip(self.mps_batch, other.mps_batch):
                new_mps_batch.append(i * j)
            return MPS_batch(new_mps_batch)

        elif isinstance(other,(int,float)):
            new_mps_batch = []
            for i in self.mps_batch:
                new_mps_batch.append(i * other)
            return MPS_batch(new_mps_batch)

    # MATRIX MULTIPLY WITH MPS BATCH
    def __matmul__(self, other: "MPS_batch"):
        mps_b1 = self.copy().mps_batch
        A1 = self.copy().reconfigure()
        if type(other) == MPS_batch:
            mps_b2 = other.copy().mps_batch
            assert len(mps_b1) == len(mps_b2)
            assert len(mps_b1[0]) == len(mps_b2[0])
            A2 = other.reconfigure()
            result = None
            for node1, node2 in zip(A1, A2):
                if result is None:
                    result = rearrange(
                        einsum(node1, node2, "n l d r, n s d t -> n l s r t"),
                        "n l s r t -> n (l s) (r t)",
                    )
                else:
                    result = einsum(
                        result,
                        rearrange(
                            einsum(node1, node2, "n l d r, n s d t -> n l s r t"),
                            "n l s r t -> n (l s) (r t)",
                        ),
                        "n a b, n b c -> n a c",
                    )
            return result

    # CONVERT ALL MPS IN BATCH TO LEFT CANONICAL USING QR.
    def left_canonical(self):
        for i in self.mps_batch:
            i.left_canonical()
        self.canonical = "left"

    # CONVERT ALL MPS IN BATCH TO RIGHT CANONICAL USING RQ.
    def right_canonical(self):
        for i in self:
            i.right_canonical()
        self.canonical = "right"

    # COMPRESS MPS BATCH USING SVD
    # def compress(self, bond_dim:int):
    #     if self.canonical != "right":
    #         A = self.right_canonical()
    #     else:
    #         A = self.copy()
    #     A = A.reconfigure()
    #     d = A[0].shape[-2]
    #     batch_size = len(self.mps_batch)
    #     sigmas = A[0].reshape(batch_size, -1, A[0].shape[3])
    #     A_new = []
    #     for i in range(1, len(A)):
    #         A_1 = A[i]
    #         U_r, S_r, V_r = svd_ceiling_batch(sigmas, bond_dim)
    #         A_new.append(U_r.reshape(batch_size, -1, d, S_r.shape[-1]))
    #         sigma_list = []
    #         for j in range(0,batch_size):
    #             sigma_list.append(torch.einsum('np,pqr -> nqr', torch.diag(S_r[j,:]) @ V_r[j,:,:], A_1[j,:,:,:]))
    #         sigmas = torch.stack(sigma_list)
    #         sigmas = sigmas.reshape(batch_size, -1, sigmas.shape[3])
    #     A_new.append(sigmas.reshape(batch_size, S_r.shape[-1], d, 1))
    #     return MPS_batch(configure(A_new, "left"), "left")

    # UTIL METHOD TO COVERT MPS BATCH TO DIFFERENT FORM USED FOR PARALLELIZATION
    def reconfigure(self):
        B2 = []
        for j in range(0, len(self.mps_batch[0])):
            B3 = []
            for i in range(0, len(self.mps_batch)):
                B3.append(self.mps_batch[i][j].tensor)
            B3 = torch.stack(B3)
            B2.append(B3)
        return B2

    # CONVERT MPS BATCH TO LIST OF VECTORS OF SIZE (batch_size, d**L)
    def contract(self):
        vecs = []
        for i in self:
            vecs.append(i.contract())
        return torch.stack(vecs)


# CREATION TOOLS


# CREATE MPS BATCH FROM VECTORS (VECTORS MUST BE torch.Tensor of shape(batch_size, d**L))
def vectors_to_mpsbatch(batch, d: int, L: int, bond_dim: int):
    batch = batch
    batch_size = batch.shape[0]
    sigmas = torch.clone(batch)
    S_r = torch.zeros(1)
    A = []
    for i in range(1, L):
        U_r, S_r, V_r = svd_batch(
            sigmas.reshape([batch_size, d * S_r.shape[-1], d ** (L - i)]), bond_dim
        )
        A.append(U_r.reshape(batch_size, -1, d, S_r.shape[-1]))
        sigma_list = []
        for j in range(0, batch_size):
            sigma_list.append(torch.diag(S_r[j, :]) @ V_r[j, :, :])
        sigmas = torch.stack(sigma_list)
    A.append(sigmas.reshape(batch_size, S_r.shape[-1], d, 1))
    return configure(A)


# #CREATE A RANDOM MPS BATCH
def random_mps_batch(batch_size, L, bd, pd=2, var="shape"):
    mps_batch = []
    diml = 1
    for i in range(L):
        i = min(i + 1, L - i - 1)
        dimr = min(pd**i, bd)
        shape = (batch_size, diml, pd, dimr)
        diml = dimr

        if var == "shape":
            std = 1.0 / (shape[0] * shape[1] * shape[2])
        else:
            std = var
        mps_batch.append(torch.tensor(torch.normal(0, std, size=shape)))
    return configure(mps_batch)


# UTILS


# TO CONVERT "PARALLELIZATION" FORM OF MPS BATCH BACK TO READABLE FORM. EXACT OPPOSITE OF MPS_batch.reconfigure()
def configure(A_new):
    batch_size = A_new[0].shape[0]
    B = []
    for i in range(0, batch_size):
        C = []
        for j in range(0, len(A_new)):
            C.append(MPS_Node(A_new[j][i]))
        B.append(MPS(C))
    return MPS_batch(B)


# SVD BATCH
def svd_batch(sigmas, bond_dim: int):
    U, S, V = torch.linalg.svd(sigmas, full_matrices=False)
    max_rank = min(len(S[0]), bond_dim)
    U_r = U[:, :, :max_rank]
    S_r = S[:, :max_rank]
    V_r = V[:, :max_rank, :]
    return U_r, S_r, V_r
