from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import optimize


class PCPrior:
    def __init__(self, use_upper_bound: bool):
        self.alpha = torch.tensor(0.5)  # The median
        # self.alpha = torch.tensor(0.05)
        self.use_upper_bound = use_upper_bound

    def get_bound(self) -> Union[float, torch.Tensor]:
        if self.use_upper_bound:
            return self.lower_bound
        else:
            return self.upper_bound

    def set_bound(self, bound: Union[float, torch.Tensor]) -> None:
        if self.use_upper_bound:
            self.lower_bound = bound
        else:
            self.upper_bound = bound


class BetaPrior:
    def __init__(self):
        # Set some initial values
        self.upper_bound = torch.tensor(2.0).double()
        self.mean = torch.min(torch.tensor(1.0), torch.tensor(0.5) * self.upper_bound)
        self.hpd_length: float = 0.2

        self.beta_dist = None
        self.find_beta_params()
        self.set_distribution()

    def set_parameters(
        self,
        upper_bound: Optional[float] = None,
        mean: Optional[float] = None,
        hpd_length: Optional[float] = None,
    ) -> None:
        if upper_bound is not None:
            self.upper_bound = torch.tensor(upper_bound)
        if mean is not None:
            # Handle the case when mean is larger than the upper bound
            if mean >= self.upper_bound:
                raise ValueError(
                    "The mean should be a number between 0 and the upper bound."
                )
            self.mean = torch.tensor(mean)
        if hpd_length is not None and upper_bound is not None:
            if hpd_length <= 0:
                raise ValueError("The HPD length should be a positive number.")
            elif hpd_length >= upper_bound:
                raise ValueError(
                    "The HPD length should be smaller than the upper bound."
                )
            self.hpd_length = hpd_length

        self.find_beta_params()
        self.set_distribution()

    # Functions for calculating the HPD interval for the beta-distributions. Think this is better that the way it is currently spesified.
    def find_beta_params(self, initial_guess=(1.0, 1.0)) -> None:
        target_mean = (self.mean / self.upper_bound).numpy()
        hpd_length_scaled = self.hpd_length / self.upper_bound

        def objective(params):
            alpha, beta = params
            if alpha <= 0 or beta <= 0:
                return 1e10  # Return large value for invalid parameters

            # Create Beta distribution
            dist = torch.distributions.Beta(torch.tensor(alpha), torch.tensor(beta))

            # Calculate mean
            mean = alpha / (alpha + beta)
            mean_error = (mean - target_mean) ** 2

            # Find HPD interval
            x = torch.linspace(0, 1, 1000)
            pdf = torch.exp(dist.log_prob(x))

            # Sort PDF values and corresponding x values
            sorted_indices = torch.argsort(pdf, descending=True)
            sorted_pdf = pdf[sorted_indices]
            sorted_x = x[sorted_indices]

            # Calculate cumulative probability
            cumsum_pdf = torch.cumsum(sorted_pdf, 0) / torch.sum(pdf)

            # Find the 95% threshold index
            threshold_idx = torch.searchsorted(cumsum_pdf, torch.tensor(0.95))
            if threshold_idx >= len(sorted_pdf):
                threshold_idx = len(sorted_pdf) - 1

            # Get x values that are above the density threshold
            selected_x = sorted_x[:threshold_idx]

            if len(selected_x) < 2:
                return 1e10

            # Calculate interval length
            interval_length = torch.max(selected_x) - torch.min(selected_x)
            hpd_error = (interval_length - hpd_length_scaled) ** 2

            return mean_error + hpd_error

        # Find optimal parameters
        result = optimize.minimize(
            objective, initial_guess, method="Nelder-Mead", options={"maxiter": 1000}
        )

        alpha, beta = result.x
        self.alpha = torch.tensor(alpha)
        self.beta = torch.tensor(beta)

    def set_distribution(self) -> None:
        self.beta_dist = torch.distributions.Beta(self.alpha, self.beta)

    def log_prob(self, theta: torch.Tensor, handle_bounds: bool = True) -> torch.Tensor:
        if torch.any(theta >= self.upper_bound) and handle_bounds:
            print(
                f"Error: Proposed theta {theta.data} is not smaller than the upper bound {self.upper_bound.data}. Returning max."
            )
            # return torch.tensor(0)
            log_prior_scaled = self.beta_dist.log_prob(
                (self.upper_bound - 0.001) / self.upper_bound
            )
            log_prior = log_prior_scaled - torch.log(self.upper_bound)
            return log_prior

        log_prior_scaled = self.beta_dist.log_prob(theta / self.upper_bound)
        log_prior = log_prior_scaled - torch.log(self.upper_bound)
        return log_prior

    def plot_distribution(self, n_points: int = 100, log=False) -> None:
        x = torch.linspace(0, self.upper_bound, n_points)
        if log:
            y = self.log_prob(x, handle_bounds=False)
        else:
            y = torch.exp(self.log_prob(x, handle_bounds=False))
        plt.plot(x, y)
        if log:
            plt.title(
                f"Log-prior for mean = {self.mean} and 95% HPD-length = {self.hpd_length}"
            )
        else:
            plt.title(
                f"Beta distribution for mean = {self.mean} and 95% HPD-length = {self.hpd_length}"
            )
        plt.show()
