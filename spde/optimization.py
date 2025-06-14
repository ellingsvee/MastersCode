from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from spde.model import SPDEModel


def optimize_parameters(
    spde: SPDEModel,
    opt_params: List[torch.Tensor],  # the initial parameters
    Y: torch.Tensor,
    A: torch.Tensor,
    X: Optional[torch.Tensor] = None,
    fixed_nu: bool = False,
    fixed_nu_val: Optional[float] = None,
    return_n_opt_steps: bool = True,
    max_opt_iterations: int = 100,
    use_stopping_criterion: bool = True,
    ftol_rel: Optional[float] = 1e-4,
    gtol: Optional[float] = 1e-4,
    lr: float = 0.1,
    progress_update_step: int = -1,
    plot_updates: bool = False,
) -> Tuple[SPDEModel, int]:
    def objective(opt_params) -> torch.Tensor:
        try:
            spde.params.set_parameter_values_from_logparam_list(
                opt_params, include_nu=not fixed_nu
            )
            spde.refresh()
            neg_ll = -spde.inference.log_likelihood(Y, A, X=X)
            return neg_ll

        except Exception as e:
            print(f"Error in likelihood computation: {e}")
            # Return a high loss value to avoid invalid updates
            return torch.tensor(1e6, dtype=spde.params.dtype, requires_grad=True)

    if fixed_nu:
        # spde.params.lognu = opt_params[0].squeeze()
        spde.params.lognu = torch.log(
            torch.tensor(fixed_nu_val, dtype=spde.params.dtype)
        )
        opt_params = opt_params[1:]

    optimizer = torch.optim.Adam(opt_params, lr=lr)
    prev_loss = float("inf")

    iterations_counter = 0
    for iteration in range(max_opt_iterations):
        iterations_counter += 1

        optimizer.zero_grad()

        loss = objective(opt_params)
        if torch.isfinite(loss):  # Only backpropagate if the loss is valid
            loss.backward()

            if use_stopping_criterion and gtol is not None:
                # Compute gradient norm
                grad_norm = torch.norm(
                    torch.cat(
                        [p.grad.view(-1) for p in opt_params if p.grad is not None]
                    )
                )
                print(f"Grad Norm: {grad_norm}")
                # Check gradient-based stopping
                if grad_norm < gtol:
                    print(
                        f"Stopping early: Gradient norm {grad_norm:.2e} < {gtol} at iteration {iteration}"
                    )
                    break

            optimizer.step()
        else:
            print("Skipping update due to inf LL value.")

        current_loss = loss.item()
        if (
            use_stopping_criterion and (ftol_rel is not None)
            # and (iterations_counter > 15)
        ):
            loss_ratio = np.abs(current_loss - prev_loss) / np.abs(current_loss)
            if loss_ratio < ftol_rel:
                print(
                    f"Stopping early: Relative loss change < {ftol_rel} at iteration {iteration}"
                )
                break
        prev_loss = current_loss

        if progress_update_step > 0 and iteration % progress_update_step == 0:
            print(f"- Iteration {iteration + 1}, loss: {current_loss}")
            spde.params.print_parameters()
            if plot_updates:
                spde.gen_and_plot_realization(only_inside_loc_domain=True)

    return spde, iterations_counter
