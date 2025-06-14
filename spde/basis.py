from typing import Tuple

import numpy as np
import scipy.sparse as sparse
import scipy.stats as stats
import torch

from custom_autograd.compute_cholesky_decomposition import cholesky_decomp
from spde.mesh import Mesh


class Basis:
    def __init__(
        self,
        mesh: Mesh,
        n_freqs: int,
        dtype=torch.double,
    ):
        self.dtype = dtype
        self.A1 = mesh.A1
        self.A2 = mesh.A2
        self.B1 = mesh.B1
        self.B2 = mesh.B2
        self.A = mesh.A
        self.B = mesh.B
        self.domain_area = self.A * self.B
        self.scaling_factor = np.sqrt(self.domain_area / 4)

        self.update_freqs(n_freqs)
        B_mat, Q_diag = self.gen_B_mat(mesh.vertices)
        self.B_mat = torch.from_numpy(B_mat)
        self.Q_diag = torch.from_numpy(Q_diag)
        self.create_Q_RW()

    def update_freqs(self, n_freqs: int) -> None:
        n_freqs_arr = np.arange(n_freqs + 1)
        K, L = np.meshgrid(n_freqs_arr, n_freqs_arr)
        K = K.flatten()
        L = L.flatten()

        # Mask away the zero frequencies
        mask = ~(K == 0) | ~(L == 0)

        # Saving the frequencies
        self.K = K[mask]
        self.L = L[mask]

        self.norm_const = np.where(
            (K == 0) | (L == 0),
            (self.domain_area / 2) ** 0.5,
            (self.domain_area / 4) ** 0.5,
        )

        self.n_freqs: int = n_freqs
        self.n_bases: int = len(self.K)

    def gen_B_mat(
        self,
        locs: np.ndarray,
        store_as_global: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        B_mat = np.zeros((len(locs), 1 + self.n_bases))

        # The intercept
        B_mat[:, 0] = 1

        if self.n_bases > 0:
            Q_diag = np.zeros(self.n_bases)
            for i, (k, l) in enumerate(zip(self.K, self.L)):
                cos_k = np.cos(np.pi * k * (locs[:, 0] - self.A1) / self.A)
                cos_l = np.cos(np.pi * l * (locs[:, 1] - self.B1) / self.B)
                B_mat[:, i + 1] = cos_k * cos_l / self.norm_const[i]
                Q_diag[i] = ((np.pi * k / self.A) ** 2 + (np.pi * l / self.B) ** 2) ** 2
        else:
            Q_diag = np.array([])

        if store_as_global:
            self.B_mat = torch.from_numpy(B_mat)
            self.Q_diag = torch.from_numpy(Q_diag)
        return B_mat, Q_diag

    def create_Q_RW(self) -> None:
        self.Q_RW = torch.diag(self.Q_diag).to(dtype=self.dtype)
        if self.n_bases > 0:
            self.Q_RW_logdet = self.Q_RW.diagonal().log().sum()
        else:
            self.Q_RW_logdet = torch.tensor(0, dtype=self.dtype)

    def compute_vals(self, theta) -> torch.Tensor:
        if self.n_bases == 0:
            return self.B_mat @ theta
        else:
            stat_part = self.B_mat[:, 0] * theta[0]
            non_stat_part = self.scaling_factor * self.B_mat[:, 1:] @ theta[1:]
            return stat_part + non_stat_part

    def gen_unscaled_param_realizations(
        self, Q_diag: np.ndarray, n_realizations: int
    ) -> np.ndarray:
        # Use the locations to generate the B_mat and Q_diag
        n = len(Q_diag)
        Q = sparse.csc_matrix((Q_diag, (np.arange(n), np.arange(n))), shape=(n, n))

        # Use the Q to genrate a realization
        Z = stats.norm.rvs(size=n * n_realizations).reshape(n, n_realizations)
        Q_chol = cholesky_decomp(Q)

        param_realizations = Q_chol.apply_Pt(
            Q_chol.solve_Lt(Q_chol.apply_P(Z), use_LDLt_decomposition=False)
        )

        return param_realizations

    def compute_NS_realization(
        self,
        locs: np.ndarray,
        n_realizations: int = 1,
    ) -> np.ndarray:
        # Create the eccessary matrices
        B_mat, Q_diag = self.gen_B_mat(locs)

        param_realizations = self.gen_unscaled_param_realizations(
            Q_diag=Q_diag, n_realizations=n_realizations
        )

        # Retrieve the non-stat part of the basies
        B_mat_NS = B_mat[:, 1:]

        # Get the realizations of the field at the specifiec locations
        X = B_mat_NS @ param_realizations

        return X
