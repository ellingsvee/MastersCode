from typing import List, Optional, Tuple

import numpy as np
import torch
from torch._C import dtype

from spde.basis import Basis
from spde.mesh import Mesh
from spde.prior import BetaPrior, PCPrior


class Parameters:
    def __init__(self, mesh: Mesh):
        self.mesh = mesh

        self.dtype = torch.double
        self.fractional_smoothness: bool = True

        # Model parameters
        self.m: int = 1
        self.d: int = 2
        self.anisotropy: bool = False
        self.mean_cov_pen_param: float = False

        self.n_freqs_logkappa: int = 0
        self.n_freqs_logsigma: int = 0
        self.n_freqs_v: int = 0

        init_tensor = torch.tensor(0, dtype=self.dtype)
        self.lognu: torch.Tensor = init_tensor
        self.logsigmae: torch.Tensor = init_tensor
        self.theta_logkappa: torch.Tensor = init_tensor
        self.theta_logsigma: torch.Tensor = init_tensor
        self.theta_vx: torch.Tensor = torch.tensor([0], dtype=self.dtype)
        self.theta_vy: torch.Tensor = torch.tensor([0], dtype=self.dtype)

        # The penalty parameters for the non-stationarity
        self.penalty_range: float = 1
        self.penalty_sd: float = 1
        self.penalty_anisotropy: float = 1
        self.penalty_mean_covariates: float = 1

        # Set up the priors
        self.set_stationary_priors()

    def set_model_parameters(
        self,
        m: int = 2,
        d: int = 2,
        anisotropy: bool = False,
        fractional_smoothness: bool = True,
        n_freqs_logkappa: int = 0,
        n_freqs_logsigma: int = 0,
        n_freqs_v: int = 0,
    ) -> None:
        self.m = m
        self.d = d
        self.anisotropy = anisotropy
        self.fractional_smoothness = fractional_smoothness
        self.n_freqs_logkappa = n_freqs_logkappa
        self.n_freqs_logsigma = n_freqs_logsigma
        self.n_freqs_v = n_freqs_v

        # Set up the bases
        self.set_bases()

    def set_bases(
        self,
    ) -> None:
        self.basis_logkappa = Basis(
            mesh=self.mesh,
            n_freqs=self.n_freqs_logkappa,
            dtype=self.dtype,
        )
        self.basis_logsigma = Basis(
            mesh=self.mesh,
            n_freqs=self.n_freqs_logsigma,
            dtype=self.dtype,
        )
        self.basis_vx = Basis(
            mesh=self.mesh,
            n_freqs=self.n_freqs_v,
            dtype=self.dtype,
        )
        self.basis_vy = Basis(
            mesh=self.mesh,
            n_freqs=self.n_freqs_v,
            dtype=self.dtype,
        )

    def set_stationary_priors(self) -> None:
        self.prior_nu = BetaPrior()
        self.prior_sigmae = PCPrior(use_upper_bound=False)
        self.prior_range = PCPrior(use_upper_bound=True)
        self.prior_sigma = PCPrior(use_upper_bound=False)
        self.prior_angle = PCPrior(use_upper_bound=False)
        self.prior_anisotropy = PCPrior(use_upper_bound=False)

    def set_penalty_parameters(
        self,
        penalty_range: float,
        penalty_sd: float,
        penalty_anisotropy: float,
        penalty_mean_covariates: float = 1,
    ) -> None:
        self.penalty_range = penalty_range
        self.penalty_sd = penalty_sd
        self.penalty_anisotropy = penalty_anisotropy
        self.penalty_mean_covariates = penalty_mean_covariates

    def log_scale(self, x: torch.Tensor) -> torch.Tensor:
        if torch.any(x == 0):
            raise ValueError(
                "Input tensor contains zero-values, which are not allowed."
            )
        return torch.log(x)

    def set_parameter_values(
        self,
        lognu: torch.Tensor,
        logsigmae: torch.Tensor,
        theta_logkappa: torch.Tensor,
        theta_logsigma: torch.Tensor,
        theta_vx: Optional[torch.Tensor] = None,
        theta_vy: Optional[torch.Tensor] = None,
    ):
        self.lognu = lognu.to(dtype=self.dtype)
        self.logsigmae = logsigmae.to(dtype=self.dtype)
        self.theta_logkappa = theta_logkappa.to(dtype=self.dtype)
        self.theta_logsigma = theta_logsigma.to(dtype=self.dtype)
        self.theta_vx = (
            theta_vx.to(dtype=self.dtype)
            if theta_vx is not None
            else torch.tensor([0], dtype=self.dtype)
        )
        self.theta_vy = (
            theta_vy.to(dtype=self.dtype)
            if theta_vy is not None
            else torch.tensor([0], dtype=self.dtype)
        )

    @property
    def nu(self) -> torch.Tensor:
        return torch.exp(self.lognu)

    @property
    def sigmae(self) -> torch.Tensor:
        return torch.exp(self.logsigmae)

    @property
    def sigmae2(self) -> torch.Tensor:
        return self.sigmae**2

    @property
    def taue(self) -> torch.Tensor:
        return 1 / self.sigmae2

    @property
    def kappa(self) -> torch.Tensor:
        return torch.exp(self.basis_logkappa.compute_vals(self.theta_logkappa))

    @property
    def kappa2(self) -> torch.Tensor:
        return self.kappa**2

    @property
    def corr_range(self) -> torch.Tensor:
        return torch.sqrt(8 * self.nu) / self.kappa

    @property
    def corr_range_stat(self) -> torch.Tensor:
        kappa_stat = torch.exp(self.theta_logkappa[0])
        return torch.sqrt(8 * self.nu) / kappa_stat

    @property
    def sigma(self):
        return torch.exp(self.basis_logsigma.compute_vals(self.theta_logsigma))

    @property
    def tau(self):
        kappa = self.kappa
        sigma = self.sigma
        beta = (self.nu + self.d / 2) / 2

        return (
            sigma
            * (
                (
                    4
                    * torch.pi
                    * (torch.lgamma(2 * beta).exp() / torch.lgamma(2 * beta - 1).exp())
                )
                ** 0.5
            )
            * (kappa ** (2 * beta - 1))
        )

    @property
    def sigma2(self):
        return self.sigma**2

    @property
    def sigma_stat(self):
        return torch.exp(self.theta_logsigma[0])

    @property
    def vx(self) -> torch.Tensor:
        return self.basis_vx.compute_vals(self.theta_vx)

    @property
    def vy(self) -> torch.Tensor:
        return self.basis_vy.compute_vals(self.theta_vy)

    @property
    def angle(self) -> torch.Tensor:
        return 0.5 * torch.atan2(self.vy, self.vx)

    @property
    def angle_stat(self) -> torch.Tensor:
        return 0.5 * torch.atan2(self.theta_vy[0], self.theta_vx[0])

    @property
    def v_norm(self) -> torch.Tensor:
        return torch.sqrt(self.vx**2 + self.vy**2)

    @property
    def corr_range_max_min(self) -> Tuple[torch.Tensor, torch.Tensor]:
        corr_range = self.corr_range
        v_norm = self.v_norm
        corr_range_max = corr_range * (torch.exp(v_norm) ** 0.5)
        corr_range_min = corr_range * (torch.exp(-v_norm) ** 0.5)
        return corr_range_max, corr_range_min

    @property
    def corr_range_ratio(self) -> torch.Tensor:
        return torch.exp(self.v_norm)

    @property
    def corr_range_ratio_stat(self) -> torch.Tensor:
        theta_vx = self.theta_vx[0]
        theta_vy = self.theta_vy[0]
        v_norm = torch.sqrt(theta_vx**2 + theta_vy**2)
        return torch.exp(v_norm)

    def print_parameters(self) -> None:
        def format(x: torch.Tensor):
            return np.round(x.data.numpy(), 3)

        print_str = f"-- nu: {format(self.nu)}, sigmae: {format(self.sigmae)}, theta_logkappa: {format(self.theta_logkappa)}, theta_logsigma: {format(self.theta_logsigma)}"
        if self.anisotropy:
            print_str += f"\n-- theta_vx: {format(self.theta_vx)}, theta_vy: {format(self.theta_vy)}"

        print(print_str)

    def set_parameter_values_from_logparam_list(
        self,
        param_val_list: List[torch.Tensor],
        include_nu: bool = True,
    ) -> None:
        index = 0

        if include_nu:
            self.lognu = param_val_list[index].squeeze()
            index += 1

        self.logsigmae = param_val_list[index].squeeze()
        index += 1

        self.theta_logkappa = param_val_list[index]
        self.theta_logsigma = param_val_list[index + 1]
        index += 2

        if self.anisotropy:
            self.theta_vx, self.theta_vy = param_val_list[index : index + 2]
            index += 2

    def get_logparam_list(
        self,
        include_nu: bool = True,
        rand_noise_amt: float = 0.0,
        zero_val_rand_noise_amt: float = 0.0,
        set_non_stat_params_to_zero: bool = True,
        seed: Optional[int] = None,
    ):
        if seed is not None:
            np.random.seed(seed)
            # torch.manual_seed(seed)

        zero_val_rand_noise_amt = torch.tensor(
            zero_val_rand_noise_amt, dtype=self.dtype
        )

        def get_param(
            true_param: torch.Tensor,
            require_grad: bool = True,
            set_non_stat_params_to_zero: bool = True,
            log_scaled: bool = False,
        ):
            true_param = true_param.detach().clone()

            # if seed is not None:
            #     np.random.seed(None)

            if len(true_param) > 1:
                shape = true_param[1:].shape
                noise = zero_val_rand_noise_amt * torch.from_numpy(
                    np.random.randn(*shape)
                ).to(dtype=true_param.dtype)
                if log_scaled:
                    true_param[1:] += noise
                else:
                    true_param[1:] += noise

                # First element noise
                if rand_noise_amt > 0:
                    noise = rand_noise_amt * torch.tensor(
                        np.random.randn(), dtype=true_param.dtype
                    )
                    true_param[0] += noise
                else:
                    noise = (
                        true_param[0]
                        * rand_noise_amt
                        * torch.tensor(np.random.randn(), dtype=true_param.dtype)
                    )
                    true_param[0] += noise

            else:
                noise = torch.from_numpy(np.random.randn(*true_param.shape)).to(
                    dtype=true_param.dtype
                )
                if true_param == 0:
                    true_param += rand_noise_amt * noise
                else:
                    true_param += true_param * rand_noise_amt * noise

            param = true_param.to(dtype=self.dtype)
            param.requires_grad = require_grad

            return param

        params = []

        if include_nu:
            params.append(get_param(self.lognu.unsqueeze(0), require_grad=True))

        params.append(get_param(self.logsigmae.unsqueeze(0)))

        params.append(
            get_param(
                self.theta_logkappa,
                set_non_stat_params_to_zero=set_non_stat_params_to_zero,
                log_scaled=True,
            )
        )
        params.append(
            get_param(
                self.theta_logsigma,
                set_non_stat_params_to_zero=set_non_stat_params_to_zero,
                log_scaled=True,
            )
        )

        if self.anisotropy:
            params.append(
                get_param(
                    self.theta_vx,
                    set_non_stat_params_to_zero=set_non_stat_params_to_zero,
                )
            )
            params.append(
                get_param(
                    self.theta_vy,
                    set_non_stat_params_to_zero=set_non_stat_params_to_zero,
                )
            )
        if seed is not None:
            np.random.seed(None)

        return params
