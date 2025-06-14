import os
from typing import Optional, Tuple

import pandas as pd
import torch

# from custom_autograd.sparse_autograd_operations import ssmm
from custom_autograd.matrix_operations import spspadd, spspmm, spspsub
from spde.mesh import Mesh
from spde.parameters import Parameters

from utils.tensor_operations import (
    init_empty_coo,
    invert_diag_tensor_coo,
    place_sparse_in_larger,
)


class Matrices:
    def __init__(self, mesh: Mesh, params: Parameters):
        self.mesh = mesh
        self.params = params

        self.vertices = torch.from_numpy(mesh.vertices)
        self.simplices = torch.from_numpy(mesh.simplices)

        self.n = self.vertices.size(0)
        self.e0, self.e1, self.e2 = self.retrieve_edges()

        # Assemble the basic matrices
        self.C = self.assemble_C()
        self.Ci = invert_diag_tensor_coo(self.C)
        init_tensor = torch.tensor([], dtype=self.params.dtype)
        self.G = self.assemble_G(init_tensor, init_tensor, anisotropy=False)
        self.I = torch.eye(self.n, dtype=self.params.dtype).to_sparse_coo()

        # Get the rational coefficients
        self.get_rational_coeff_lists()

    def retrieve_edges(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p0 = self.vertices[self.simplices[:, 0]]
        p1 = self.vertices[self.simplices[:, 1]]
        p2 = self.vertices[self.simplices[:, 2]]
        e0 = p2 - p1
        e1 = p0 - p2
        e2 = p1 - p0
        return e0, e1, e2

    def get_centroid_vals(self, vals: torch.Tensor) -> torch.Tensor:
        val0 = vals[self.simplices[:, 0]]
        val1 = vals[self.simplices[:, 1]]
        val2 = vals[self.simplices[:, 2]]
        return (val0 + val1 + val2) / 3

    def assemble_C(
        self,
        vals: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if vals is None:
            vals = torch.ones(self.n, dtype=torch.double)

        # Compute the mean values at the centroids
        mean_vals = self.get_centroid_vals(vals)

        # Unsure. Copied from fmesher
        cross1 = self.e1[:, 0] * self.e2[:, 1] - self.e1[:, 1] * self.e2[:, 0]
        cross2 = self.e2[:, 0] * self.e0[:, 1] - self.e2[:, 1] * self.e0[:, 0]
        cross3 = self.e0[:, 0] * self.e1[:, 1] - self.e0[:, 1] * self.e1[:, 0]
        a = (cross1 + cross2 + cross3) / 6.0

        # Construct the scaled C-matrix
        C_vals = torch.repeat_interleave(mean_vals * a / 3.0, 3)
        row_col_idx = self.simplices.ravel()
        indices = torch.vstack((row_col_idx, row_col_idx))

        return torch.sparse_coo_tensor(
            indices,
            C_vals,
            size=(self.n, self.n),
            dtype=self.params.dtype,
            is_coalesced=None,
        ).coalesce()

    def assemble_H(
        self,
        vx_vals: torch.Tensor,
        vy_vals: torch.Tensor,
        anisotropy: bool = True,
        epsilon: float = 1e-6,
    ) -> Tuple[torch.Tensor, ...]:
        if anisotropy:
            # Compute the mean values at the centroids
            vx = self.get_centroid_vals(vx_vals)
            vy = self.get_centroid_vals(vy_vals)
            v = torch.sqrt(vx**2 + vy**2)

            # Compute the hyperbolic functions. Use epsilon to avoid division by zero.
            cosh_vals = torch.cosh(v)
            sinh_over_magnitude = torch.sinh(v) / (v + epsilon)

            # Asseble the H-matrix controlling the anisotropy
            H00 = cosh_vals + sinh_over_magnitude * vx
            H01 = sinh_over_magnitude * vy
            H10 = H01
            H11 = cosh_vals - sinh_over_magnitude * vx
        else:
            H00 = torch.ones(self.simplices.shape[0], dtype=self.params.dtype)
            H01 = torch.zeros(self.simplices.shape[0], dtype=self.params.dtype)
            H10 = torch.zeros(self.simplices.shape[0], dtype=self.params.dtype)
            H11 = torch.ones(self.simplices.shape[0], dtype=self.params.dtype)

        # Stack them into a matrix with shape (n, 2, 2)
        return H00, H01, H10, H11

    def retrieve_inverse_H(
        self,
        H00: torch.Tensor,
        H01: torch.Tensor,
        H10: torch.Tensor,
        H11: torch.Tensor,
        epsilon: float = 1e-6,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute the trace to extract ||v||
        trace_H = H00 + H11
        v_magnitude = torch.acosh(trace_H / 2)

        # Compute sinh(||v||)/||v|| safely
        sinh_over_magnitude = torch.sinh(v_magnitude) / (v_magnitude + epsilon)

        # Recover vx and vy
        vx = (H00 - torch.cosh(v_magnitude)) / sinh_over_magnitude
        vy = H01 / sinh_over_magnitude

        return vx, vy

    def compute_H_evals(
        self,
        H00: torch.Tensor,
        H01: torch.Tensor,
        H10: torch.Tensor,
        H11: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        trH = H00 + H11
        sqrt_discriminant = torch.sqrt(trH**2 - 4)  # Formula is trH^2 - 4*det(H)
        return 0.5 * (trH + sqrt_discriminant), 0.5 * (trH - sqrt_discriminant)

    def assemble_G(
        self,
        vx_vals: torch.Tensor,
        vy_vals: torch.Tensor,
        anisotropy: bool = True,
    ) -> torch.Tensor:
        # Compute the H-matrix
        H00, H01, H10, H11 = self.assemble_H(vx_vals, vy_vals, anisotropy=anisotropy)

        # Assemble the adjoint matrix
        Hadj00 = H11
        Hadj01 = -H01
        Hadj10 = -H10
        Hadj11 = H00
        Hadj = torch.stack(
            [
                torch.stack([Hadj00, Hadj01], dim=-1),
                torch.stack([Hadj10, Hadj11], dim=-1),
            ],
            dim=-2,
        )

        # "Flat area" better approximation for use in G-calculation.
        cross3 = self.e0[:, 0] * self.e1[:, 1] - self.e0[:, 1] * self.e1[:, 0]
        fa = cross3 / 2.0

        # Compute the G-matrix
        E = torch.stack((self.e0, self.e1, self.e2), dim=-1)
        row_idx = torch.repeat_interleave(self.simplices, 3, dim=1).ravel()
        col_idx = torch.tile(self.simplices, (3,)).ravel()

        indices = torch.vstack((row_idx, col_idx))
        G_values = (1 / (4 * fa)).reshape(-1, 1) * (
            E.permute(0, 2, 1) @ Hadj @ E
        ).reshape(-1, 9)
        G_values = G_values.ravel()

        # Store the H-matrix for later use
        self.H00 = H00
        self.H01 = H01
        self.H10 = H10
        self.H11 = H11

        return torch.sparse_coo_tensor(
            indices, G_values, size=(self.n, self.n), is_coalesced=None
        ).coalesce()

    def get_rational_coeff_lists(
        self,
    ) -> None:
        """
        Retrieve the roots for the rational approximation from the m-tables.
        """
        dir_path = os.path.dirname(os.path.realpath(__file__))
        if self.params.m == 1:
            filepath = os.path.join(dir_path, "m_tables/m1table.csv")
            df = pd.read_csv(filepath)
            beta_vals = df["beta"].to_numpy()
            factor_vals = df["factor"].to_numpy()
            rb1_vals = df["rb.1"].to_numpy()
            rb2_vals = df["rb.2"].to_numpy()
            rc_vals = df["rc"].to_numpy()
        else:
            raise ValueError("m only implemented for 1")

        self.beta_vals = torch.from_numpy(beta_vals)
        self.factor_vals = torch.from_numpy(factor_vals)
        self.rb1_vals = torch.from_numpy(rb1_vals)
        self.rb2_vals = torch.from_numpy(rb2_vals)
        self.rc_vals = torch.from_numpy(rc_vals)

    def interp_rational_coeffs(
        self, beta: torch.Tensor, beta_values: torch.Tensor, f_values: torch.Tensor
    ) -> torch.Tensor:
        indices = torch.searchsorted(beta_values, beta) - 1
        indices = torch.clamp(indices, 0, len(beta_values) - 2)

        beta0 = beta_values[indices]
        beta1 = beta_values[indices + 1]
        f0 = f_values[indices]
        f1 = f_values[indices + 1]

        slope = (f1 - f0) / (beta1 - beta0)
        return f0 + slope * (beta - beta0)

    def cubic_hermite_interp(
        self, beta: torch.Tensor, beta_values: torch.Tensor, f_values: torch.Tensor
    ):
        # Assume x_vals is sorted, equally spaced
        indices = torch.searchsorted(beta_values, beta) - 1
        indices = torch.clamp(indices, 0, len(beta_values) - 2)

        x0 = beta_values[indices]
        x1 = beta_values[indices + 1]
        y0 = f_values[indices]
        y1 = f_values[indices + 1]

        t = (beta - x0) / (x1 - x0)
        h00 = (1 + 2 * t) * (1 - t) ** 2
        h10 = t * (1 - t) ** 2
        h01 = t**2 * (3 - 2 * t)
        h11 = t**2 * (t - 1)

        # Estimate derivatives with finite differences
        dy = (y1 - y0) / (x1 - x0)
        return h00 * y0 + h10 * (x1 - x0) * dy + h01 * y1 + h11 * (x1 - x0) * dy

    def compute_Pl_and_Pr_matrices(self) -> None:
        """
        Compute the Pl and Pr matrices for the SPDE model.
        """
        nu = self.params.nu
        kappa2 = self.params.kappa2

        # Smoothness parameter
        beta = (nu + self.params.d / 2) / 2

        # Scaling for the L-matrix
        scale_factor = kappa2.min()

        kappa2_mult_C = self.assemble_C(kappa2)
        L = spspadd(self.G, kappa2_mult_C)
        L = L / scale_factor
        CiL = spspmm(self.Ci, L)

        if not self.params.fractional_smoothness:
            Pr = self.I
            Pl = L
            if int(beta) > 1:
                for _ in range(int(beta) - 1):
                    Pl = spspmm(Pl, CiL)
            Pl_scaling = scale_factor**beta

        if self.params.fractional_smoothness:
            # Retrieve factors and roots for the rational approximation
            factor = self.cubic_hermite_interp(
                beta=beta, beta_values=self.beta_vals, f_values=self.factor_vals
            )
            rb1 = self.cubic_hermite_interp(
                beta=beta, beta_values=self.beta_vals, f_values=self.rb1_vals
            )
            rb2 = self.cubic_hermite_interp(
                beta=beta, beta_values=self.beta_vals, f_values=self.rb2_vals
            )
            rc = self.cubic_hermite_interp(
                beta=beta, beta_values=self.beta_vals, f_values=self.rc_vals
            )

            # Here we make the assumption that nu is in (0, 3).
            m_beta = 1

            # Here we assume m = 1
            len_rb = 2
            len_rc = 1

            # Construct Pl
            Pl = spspsub(self.I, CiL * rb1)
            if len_rb > 1:
                Pl_temp = spspsub(self.I, CiL * rb2)
                Pl = spspmm(Pl, Pl_temp)

            Lp = self.C
            if m_beta > 1:
                for i in range(m_beta - 1):
                    Lp = spspmm(Lp, CiL)
            Pl = spspmm(Lp, Pl)
            Pl_scaling = (scale_factor**beta) / factor.squeeze()

            # Construct Pr
            Pr = spspsub(self.I, CiL * rc)

        Pl = Pl_scaling * Pl

        self.Pl = Pl.coalesce()
        self.Pr = Pr.coalesce()
        self.CiL = CiL

    def compute_rational_approx_matrices(self):
        self.compute_Pl_and_Pr_matrices()

        # Compute the noise scale matrix
        beta = (self.params.nu + self.params.d / 2) / 2
        kappa2 = self.params.kappa2
        sigma2 = self.params.sigma2

        noise_scale_val = (
            (torch.lgamma(2 * beta).exp() / torch.lgamma(2 * beta - 1).exp())
            * 4
            * torch.pi
            * sigma2  # Seems like it works to put it here!
            * kappa2 ** (2 * beta - 1)
        )

        C_NS = self.assemble_C(noise_scale_val)
        Ci_NS = invert_diag_tensor_coo(C_NS)

        # Compute the Q-matrix
        Q = spspmm(spspmm(self.Pl.T, Ci_NS), self.Pl)

        # Set the matrices
        self.C_NS = C_NS
        self.Ci_NS = Ci_NS
        self.Q = Q.coalesce()

    def construct_Qz(
        self,
        X: Optional[torch.Tensor],
        mean_cov_pen_param: Optional[float] = None,
    ):
        if X is not None:
            I = torch.eye(X.shape[1], dtype=torch.double).to_sparse_coo().coalesce()

            Qz = init_empty_coo(
                (self.Q.shape[0] + X.shape[1], self.Q.shape[1] + X.shape[1])
            )
            Qz = place_sparse_in_larger(Qz, self.Q, target_indices=(0, 0))
            Qz = place_sparse_in_larger(
                Qz,
                mean_cov_pen_param * I,
                target_indices=(self.Q.shape[0], self.Q.shape[1]),
            )
            return Qz

        return self.Q

    def construct_Qc(
        self,
        taue: torch.Tensor,
        APr: torch.Tensor,
        X: Optional[torch.Tensor] = None,
        mean_cov_pen_param: Optional[float] = None,
    ):
        Q = self.Q

        # Construct the blocks of the Qc matrix
        QcAA = spspadd(Q, spspmm(APr.T, APr) * taue)

        if X is not None:
            I = torch.eye(X.shape[1], dtype=torch.double).to_sparse_coo().coalesce()

            QcAB = spspmm(APr.T, X) * taue
            QcBA = QcAB.T
            QcBB = spspadd(spspmm(X.T, X) * taue, mean_cov_pen_param * I)

            Qc = init_empty_coo((Q.shape[0] + X.shape[1], Q.shape[1] + X.shape[1]))
            Qc = place_sparse_in_larger(Qc, QcAA, target_indices=(0, 0))
            Qc = place_sparse_in_larger(Qc, QcAB, target_indices=(0, Q.shape[1]))
            Qc = place_sparse_in_larger(Qc, QcBA, target_indices=(Q.shape[0], 0))
            Qc = place_sparse_in_larger(
                Qc, QcBB, target_indices=(Q.shape[0], Q.shape[1])
            )

            return Qc

        return QcAA

    def assemble_H_old_param(
        self,
        vx_vals: torch.Tensor,
        vy_vals: torch.Tensor,
        gamma_val: float = 1e-3,
        anisotropy: bool = True,
    ) -> Tuple[torch.Tensor, ...]:
        if anisotropy:
            # Compute the mean values at the centroids
            vx = self.get_centroid_vals(vx_vals)
            vy = self.get_centroid_vals(vy_vals)

            # Asseble the H-matrix controlling the anisotropy
            H00 = gamma_val + vx**2
            H01 = vx * vy
            H10 = H01
            H11 = gamma_val + vy**2
        else:
            H00 = torch.ones(self.simplices.shape[0], dtype=torch.double)
            H01 = torch.zeros(self.simplices.shape[0], dtype=torch.double)
            H10 = torch.zeros(self.simplices.shape[0], dtype=torch.double)
            H11 = torch.ones(self.simplices.shape[0], dtype=torch.double)

        # Stack them into a matrix with shape (n, 2, 2)
        return H00, H01, H10, H11

    def compute_vec_field_loss(
        self,
        vx_vals: torch.Tensor,
        vy_vals: torch.Tensor,
        vx_center_true: torch.Tensor,
        vy_center_true: torch.Tensor,
    ):
        vx_mean = self.get_centroid_vals(vx_vals)
        vy_mean = self.get_centroid_vals(vy_vals)

        loss = torch.sum(
            (vx_mean - vx_center_true) ** 2 + (vy_mean - vy_center_true) ** 2
        )
        return loss
