from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sparse
import scipy.stats as stats
import torch

from custom_autograd import CholmodOperations, torch_sparse_coo_to_sparse_structure
from custom_autograd.compute_cholesky_decomposition import cholesky_decomp
from custom_autograd.matrix_operations import spdmm, spspmm
from partial_inverse import qinv_tensor

# from spde3.ll import compute_ll
from spde.matrices import Matrices
from spde.mesh import Mesh
from spde.parameters import Parameters
from utils.scipy_torch_conversion import torch_coo_to_scipy_csc
from utils.tensor_operations import (
    enforce_symmetry,
    init_empty_coo,
    place_sparse_in_larger,
)


class Inference:
    def __init__(
        self,
        mesh: Mesh,
        params: Parameters,
        matrices: Matrices,
    ):
        self.mesh = mesh
        self.params = params
        self.matrices = matrices

        # Related to the custom autograd functions
        self.cholmod_ops_Pl: Optional[CholmodOperations] = None
        self.cholmod_ops_Qc: Optional[CholmodOperations] = None

    def retrieve_inference_matrices(
        self,
        Y: torch.Tensor,
        A: torch.Tensor,
        X: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.cholmod_ops_Pl is None:
            self.cholmod_ops_Pl = CholmodOperations(
                sparse_structure=torch_sparse_coo_to_sparse_structure(self.matrices.Pl),
                use_torch_qinv=False,
            )

        APr = spspmm(A, self.matrices.Pr)

        if X is None:
            S = APr
        else:
            S = init_empty_coo((APr.shape[0], APr.shape[1] + X.shape[1]))
            S = place_sparse_in_larger(S, APr, target_indices=(0, 0))
            S = place_sparse_in_larger(S, X, target_indices=(0, APr.shape[1]))

        # Construct the Qz matrix
        Qz = self.matrices.construct_Qz(
            X=X,
            mean_cov_pen_param=self.params.penalty_mean_covariates,
        )

        Qc = self.matrices.construct_Qc(
            taue=self.params.taue,
            APr=APr,
            X=X,
            mean_cov_pen_param=self.params.penalty_mean_covariates,
        )
        Qc = enforce_symmetry(Qc)
        if self.cholmod_ops_Qc is None:
            self.cholmod_ops_Qc = CholmodOperations(
                sparse_structure=torch_sparse_coo_to_sparse_structure(Qc),
                use_torch_qinv=False,
            )

        return S, Qz, Qc

    def simulate_field(
        self, N_sim: int = 1, seed: Optional[int] = None
    ) -> sparse.csc_matrix:
        # Set the seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Retrieve and convert the relevant matrices
        Q = torch_coo_to_scipy_csc(self.matrices.Q)
        Pr = torch_coo_to_scipy_csc(self.matrices.Pr)

        N = Q.shape[0]

        # Generate iid standard normal random variables
        Z = stats.norm.rvs(size=N * N_sim).reshape(N, N_sim)

        # Compute the Cholesky decomposition of Q
        Q_chol = cholesky_decomp(Q)

        # Transform the iid standard normal random variables
        X = Q_chol.apply_Pt(
            Q_chol.solve_Lt(Q_chol.apply_P(Z), use_LDLt_decomposition=False)
        )

        # Apply the Pr operator
        U = Pr @ X
        return U

    def predict(
        self,
        Y: torch.Tensor,
        A: torch.Tensor,
        Apred: torch.Tensor,
        X: Optional[torch.Tensor] = None,
        Xpred: Optional[torch.Tensor] = None,
        add_observation_noise: bool = False,
        error_handling: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        S, Qz, Qc = self.retrieve_inference_matrices(Y=Y, A=A, X=X)

        taue = self.params.taue

        if Y.ndim == 1:
            Y = Y.unsqueeze(1)

        StTaueY = taue * spdmm(S.T, Y)
        MUc = self.cholmod_ops_Qc.solve_system(Qc, StTaueY)

        ApredPr = spspmm(Apred, self.matrices.Pr)

        if X is not None and Xpred is not None:
            Spred = init_empty_coo((ApredPr.shape[0], ApredPr.shape[1] + X.shape[1]))
            Spred = place_sparse_in_larger(Spred, ApredPr, target_indices=(0, 0))
            Spred = place_sparse_in_larger(
                Spred, X, target_indices=(0, ApredPr.shape[1])
            )
        else:
            Spred = ApredPr

        # Compute the mean of the predictive distribution
        MUpred = spdmm(Spred, MUc)

        # Compute the variance of the predictive distribution.
        if add_observation_noise:
            obs_noise = 1.0 / taue
        else:
            obs_noise = 0.0

        Qcinv = qinv_tensor(Qc, error_handling=error_handling)
        Vpred = spspmm(Spred, spspmm(Qcinv, Spred.T)).to_dense().diagonal() + obs_noise

        return MUpred, Vpred

    def get_N_and_N_realizations_from_Y(
        self,
        Y: torch.Tensor,
    ) -> Tuple[int, int]:
        try:
            N = Y.size(0)
            N_realizations = Y.size(1)
        except Exception as e:
            print(f"Exception in compute_ll: {e}")
            Y = Y.unsqueeze(1)
            N = Y.size(0)
            N_realizations = Y.size(1)
        return N, N_realizations

    def log_likelihood(
        self,
        Y: torch.Tensor,
        A: torch.Tensor,
        X: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        S, Qz, Qc = self.retrieve_inference_matrices(Y=Y, A=A, X=X)
        N, N_realizations = self.get_N_and_N_realizations_from_Y(Y)

        taue = self.params.taue
        StTaueY = spdmm(S.T, taue * Y)
        MUc, logdet_Qc = self.cholmod_ops_Qc.solve_system_and_compute_logdet(
            Qc, StTaueY
        )

        # Compute the LL
        ll = torch.zeros(1, dtype=torch.double)
        ll += self.ll_penalty(N_realizations=N_realizations)
        ll += self.ll_obs_error(
            taue=taue,
            N=N,
            N_realizations=N_realizations,
        )

        if X is not None:
            N_covariates = X.size(1)
        else:
            N_covariates = None

        ll += self.ll_logdetQ(
            N_realizations=N_realizations,
            include_mean_cov=(X is not None),
            N_covariates=N_covariates,
        )

        ll -= 0.5 * logdet_Qc * N_realizations
        ll += self.ll_posterior(
            Y=Y,
            S=S,
            Qz=Qz,
            MUc=MUc,
            taue=taue,
        )

        return ll

    def ll_penalty(
        self,
        N_realizations: int,
    ) -> torch.Tensor:
        """
        Use autodiff. The ll-contributions from the penalties on the non-stationarity
        """
        ll_single_realiz = torch.zeros(1, dtype=torch.double)
        ll_single_realiz += self.log_prior_joint_range_and_marinal_sd()
        if self.params.anisotropy:
            ll_single_realiz += self.log_prior_anisotropy()
        ll_single_realiz += self.log_prior_observation_noise()
        ll_single_realiz += self.log_prior_nu()
        return ll_single_realiz * N_realizations

    def log_prior_non_stat_param(
        self,
        theta: torch.Tensor,
        Q_RW: torch.Tensor,
        Q_RW_logdet: torch.Tensor,
        Q_diag: torch.Tensor,
        tau: float,
        scaling_factor: float,
    ) -> torch.Tensor:
        if len(theta) > 0:
            return -0.5 * torch.sum(tau * Q_diag * (theta * scaling_factor) ** 2)

        else:
            return torch.tensor(0.0, dtype=self.params.dtype)

    def log_prior_joint_range_and_marinal_sd(self) -> torch.Tensor:
        halfdim = torch.tensor(self.params.d / 2)

        lambda_1 = -torch.log(self.params.prior_range.alpha) * (
            self.params.prior_range.get_bound() ** halfdim
        )
        lambda_2 = (
            -torch.log(self.params.prior_sigma.alpha)
            / self.params.prior_sigma.get_bound()
        )
        corr_range_stat = self.params.corr_range_stat
        sigma_stat = self.params.sigma_stat
        log_prior_stat = (
            torch.log(halfdim)
            + torch.log(lambda_1)
            + torch.log(lambda_2)
            - (halfdim + 1) * torch.log(corr_range_stat)
            - lambda_1 * (corr_range_stat ** (-halfdim))
            - lambda_2 * sigma_stat
        )

        # The non-stationary penalty
        log_prior_non_stat = 0.0
        if self.params.basis_logkappa.n_bases > 0:
            log_prior_non_stat += self.log_prior_non_stat_param(
                theta=self.params.theta_logkappa[1:],
                Q_RW=self.params.basis_logkappa.Q_RW,
                Q_RW_logdet=self.params.basis_logkappa.Q_RW_logdet,
                Q_diag=self.params.basis_logkappa.Q_diag,
                tau=self.params.penalty_range,
                scaling_factor=self.params.basis_logkappa.scaling_factor,
            )

        if self.params.basis_logsigma.n_bases > 0:
            log_prior_non_stat += self.log_prior_non_stat_param(
                theta=self.params.theta_logsigma[1:],
                Q_RW=self.params.basis_logsigma.Q_RW,
                Q_RW_logdet=self.params.basis_logsigma.Q_RW_logdet,
                Q_diag=self.params.basis_logsigma.Q_diag,
                tau=self.params.penalty_sd,
                scaling_factor=self.params.basis_logsigma.scaling_factor,
            )

        return log_prior_stat + log_prior_non_stat

    def log_prior_anisotropy(self) -> torch.Tensor:
        sigmav2 = (
            -0.5
            * (torch.log(torch.tensor(self.params.prior_anisotropy.get_bound())) ** 2)
            / torch.log(self.params.prior_anisotropy.alpha)
        )
        corr_range_ratio_stat = self.params.corr_range_ratio_stat
        log_prior_stat = (
            -torch.log(sigmav2)
            + torch.log(
                torch.log(corr_range_ratio_stat) + 1e-10
            )  # Added 1e-10 to avoid log(0)
            - torch.log(corr_range_ratio_stat)
            - (0.5 * (torch.log(corr_range_ratio_stat) ** 2) / sigmav2)
        )

        # The non-stationary penalty
        log_prior_non_stat = 0.0
        if self.params.basis_vx.n_bases > 0:
            # corr_range_ratio_non_stat = corr_range_ratio[1:]
            theta_vx_non_stat = self.params.theta_vx[1:]
            theta_vy_non_stat = self.params.theta_vy[1:]

            log_prior_non_stat += self.log_prior_non_stat_param(
                theta=theta_vx_non_stat,
                Q_RW=self.params.basis_vx.Q_RW,
                Q_RW_logdet=self.params.basis_vx.Q_RW_logdet,
                Q_diag=self.params.basis_vx.Q_diag,
                tau=self.params.penalty_anisotropy,
                scaling_factor=self.params.basis_vx.scaling_factor,
            )
            log_prior_non_stat += self.log_prior_non_stat_param(
                theta=theta_vy_non_stat,
                Q_RW=self.params.basis_vy.Q_RW,
                Q_RW_logdet=self.params.basis_vy.Q_RW_logdet,
                Q_diag=self.params.basis_vy.Q_diag,
                tau=self.params.penalty_anisotropy,
                scaling_factor=self.params.basis_vy.scaling_factor,
            )

        return log_prior_stat + log_prior_non_stat

    def log_prior_observation_noise(self) -> torch.Tensor:
        lambda_val = (
            -torch.log(self.params.prior_sigmae.alpha)
            / self.params.prior_sigmae.get_bound()
        )
        log_prior = torch.log(lambda_val) - lambda_val * self.params.sigmae
        return log_prior

    def log_prior_nu(self) -> torch.Tensor:
        return self.params.prior_nu.log_prob(self.params.nu)

    def ll_obs_error(
        self,
        taue: torch.Tensor,
        N: int,
        N_realizations: int,
    ) -> torch.Tensor:
        ll_single_realiz = torch.zeros(1, dtype=torch.double)
        ll_single_realiz += 0.5 * N * torch.log(taue)
        return ll_single_realiz * N_realizations

    def ll_logdetQ(
        self,
        N_realizations: int,
        include_mean_cov: bool = False,
        N_covariates: Optional[int] = None,
    ) -> torch.Tensor:
        ll_single_realiz = torch.zeros(1, dtype=torch.double)

        ll_single_realiz += self.cholmod_ops_Pl.compute_logdet(self.matrices.Pl)
        ll_single_realiz += 0.5 * torch.sum(
            torch.log(self.matrices.Ci_NS.coalesce().values())
        )

        if include_mean_cov:
            ll_single_realiz += (
                0.5 * N_covariates * np.log(self.params.penalty_mean_covariates)
            )

        return ll_single_realiz * N_realizations

    def ll_posterior(
        self,
        Y: torch.Tensor,
        S: torch.Tensor,
        Qz: torch.Tensor,
        MUc: torch.Tensor,
        taue: torch.Tensor,
    ) -> torch.Tensor:
        ll = torch.zeros(1, dtype=torch.double)
        ll -= 0.5 * torch.sum(torch.diag(MUc.T @ spdmm(Qz, MUc)))
        SMUc = spdmm(S, MUc)
        Y_minus_SpostMu = Y - SMUc
        ll -= 0.5 * torch.sum(torch.diag(taue * Y_minus_SpostMu.T @ Y_minus_SpostMu))
        return ll
