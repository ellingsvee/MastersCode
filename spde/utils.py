import copy
from typing import Callable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Ellipse

from spde.basis import Basis
from spde.inference import Inference
from spde.matrices import Matrices
from spde.mesh import Mesh
from spde.parameters import Parameters


class Utils:
    def __init__(
        self,
        mesh: Mesh,
        params: Parameters,
        matrices: Matrices,
        inference: Inference,
    ):
        self.mesh = mesh
        self.params = params
        self.matrices = matrices
        self.inference = inference

    def estimate_v_params_from_vec_field(
        self,
        vx_vals: torch.Tensor,
        vy_vals: torch.Tensor,
        gamma_val: float = 1.0,
        penalty_scale: float = 1.0,
        iterations: int = 5000,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        H00_old, H01_old, H10_old, H11_old = self.matrices.assemble_H_old_param(
            gamma_val=gamma_val, vx_vals=vx_vals, vy_vals=vy_vals, anisotropy=True
        )

        H00_old = H00_old[self.mesh.simplices_mask]
        H01_old = H01_old[self.mesh.simplices_mask]
        H10_old = H10_old[self.mesh.simplices_mask]
        H11_old = H11_old[self.mesh.simplices_mask]

        theta_vx_opt = torch.Tensor([0.1] * self.params.basis_vx.n_bases).double()
        theta_vy_opt = torch.Tensor([0.1] * self.params.basis_vx.n_bases).double()
        theta_vx_opt.requires_grad = True
        theta_vy_opt.requires_grad = True
        params = [theta_vx_opt, theta_vy_opt]

        optimizer = torch.optim.Adam(params, lr=0.1)

        def objective(params):
            vx_vals_opt = self.params.basis_vx.B_mat @ params[0]
            vy_vals_opt = self.params.basis_vy.B_mat @ params[1]

            H00, H01, H10, H11 = self.matrices.assemble_H(
                vx_vals_opt, vy_vals_opt, anisotropy=True
            )
            H00 = H00[self.mesh.simplices_mask]
            H01 = H01[self.mesh.simplices_mask]
            H10 = H10[self.mesh.simplices_mask]
            H11 = H11[self.mesh.simplices_mask]

            loss = torch.sum(
                (H00 - H00_old) ** 2
                + (H01 - H01_old) ** 2
                + (H10 - H10_old) ** 2
                + (H11 - H11_old) ** 2
            )

            # Add a penalty for the anisotropy
            log_prior_non_stat = 0.0
            log_prior_non_stat += self.inference.log_prior_non_stat_param(
                theta=params[0][1:],
                Q_RW=self.params.basis_vx.Q_RW,
                Q_RW_logdet=torch.tensor(1e-6),
                tau=self.params.penalty_anisotropy,
                scaling_factor=self.params.basis_vx.scaling_factor,
            )
            log_prior_non_stat += self.inference.log_prior_non_stat_param(
                theta=params[1][1:],
                Q_RW=self.params.basis_vy.Q_RW,
                Q_RW_logdet=torch.tensor(1e-6),
                tau=self.params.penalty_anisotropy,
                scaling_factor=self.params.basis_vx.scaling_factor,
            )

            loss -= log_prior_non_stat * penalty_scale

            return loss

        for i in range(iterations):
            optimizer.zero_grad()
            loss = objective(params)
            loss.backward()
            if i % 1000 == 0:
                print(f"Iteration: {i}. Loss: {loss}")
            optimizer.step()

        return params[0], params[1]

    def plot_iso_correlation(
        self,
        n_pts_x=10,
        n_pts_y=10,
        xlabel=None,
        ylabel=None,
        domain_padding: float = 1,
        save_fig_name: Optional[str] = None,
    ):
        xmin = self.mesh.loc_domain[:, 0].min()
        xmax = self.mesh.loc_domain[:, 0].max()
        ymin = self.mesh.loc_domain[:, 1].min()
        ymax = self.mesh.loc_domain[:, 1].max()
        X, Y = np.meshgrid(
            np.linspace(xmin, xmax, n_pts_x), np.linspace(ymin, ymax, n_pts_y)
        )
        loc = np.column_stack([X.flatten(), Y.flatten()])

        # A_loc = self.mesh.create_observation_matrix(loc).to_dense().numpy()
        A_loc = self.mesh.create_observation_matrix(loc)

        extra_bases_loc = None

        basis_logkappa = copy.deepcopy(self.params.basis_logkappa)
        basis_vx = copy.deepcopy(self.params.basis_vx)
        basis_vy = copy.deepcopy(self.params.basis_vy)

        # basis_logkappa.gen_B_mat(n=loc.shape[0], loc=loc, extra_bases=extra_bases_loc)
        basis_logkappa.gen_B_mat(locs=loc, store_as_global=True)
        basis_vx.gen_B_mat(locs=loc, store_as_global=True)
        basis_vy.gen_B_mat(locs=loc, store_as_global=True)

        logkappa_vals = basis_logkappa.compute_vals(
            self.params.theta_logkappa.detach()
        ).numpy()
        kappa_vals = np.exp(logkappa_vals)
        vx_vals = basis_vx.compute_vals(self.params.theta_vx.detach()).numpy()
        vy_vals = basis_vy.compute_vals(self.params.theta_vy.detach()).numpy()

        angles = 0.5 * np.arctan2(vy_vals, vx_vals)
        v_norm = np.sqrt(vx_vals**2 + vy_vals**2)
        lambda_max = np.exp(v_norm)
        lambda_min = np.exp(-v_norm)
        corr_range_max = np.sqrt(8 * self.params.nu.detach() * lambda_max) / kappa_vals
        corr_range_min = np.sqrt(8 * self.params.nu.detach() * lambda_min) / kappa_vals

        corr_range_max *= 0.3
        corr_range_min *= 0.3

        # corr_range_max = lambda_max
        # corr_range_min = lambda_min

        fig, ax = plt.subplots(figsize=(8, 6))
        for i in range(len(loc)):
            # major_axis = np.exp(v_norm[i])
            # minor_axis = np.exp(-v_norm[i])

            ax.add_patch(
                Ellipse(
                    loc[i, :],
                    corr_range_max[i],
                    corr_range_min[i],
                    angle=np.rad2deg(angles[i]),
                    linewidth=1.5,
                    facecolor="none",
                    edgecolor="blue",
                )
            )

        ax.scatter(loc[:, 0], loc[:, 1], c="red", s=5)

        ax.plot(
            self.mesh.loc_domain[:, 0],
            self.mesh.loc_domain[:, 1],
            color="black",
            linewidth=3,
        )

        xlim = (xmin - domain_padding, xmax + domain_padding)
        ylim = (ymin - domain_padding, ymax + domain_padding)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        ax.set_aspect("equal", adjustable="box")

        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        if save_fig_name is not None:
            plt.savefig(save_fig_name, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def gen_basis_realization(
        self,
        basis: Basis,
        locs: np.ndarray,
        n_realizations: int,
        tau_val: np.ndarray,
        param_stat: float,
        log_scaled: bool = True,
    ) -> np.ndarray:
        param_NS = basis.compute_NS_realization(
            locs=locs, n_realizations=n_realizations
        )

        if log_scaled:
            params_log = np.log(param_stat) + param_NS * (tau_val**-0.5)
            params = np.exp(params_log)
        else:
            params = param_stat + param_NS * (tau_val**-0.5)

        return params

    def evaluate_basis_penalties(
        self,
        basis: Basis,
        locs: np.ndarray,
        n_realizations: int,
        tau_vals: np.ndarray,
        param_stat: float,
        log_scaled: bool = True,
        bound: float = 2.0,
        transform_func: Callable = lambda x: x,
        inv_transform_func: Callable = lambda x: x,
    ) -> np.ndarray:
        param_NS = basis.compute_NS_realization(
            locs=locs, n_realizations=n_realizations
        )

        probs = np.zeros(len(tau_vals))
        if log_scaled:
            bound = np.log(bound)

        param_stat_inv_transform = inv_transform_func(param_stat)

        for i, tau_val in enumerate(tau_vals):
            # Scale the X based on the tau-val
            param_NS_i = param_NS * (tau_val ** (-0.5))

            # Compute the amsolute value
            if log_scaled:
                param = np.exp(np.log(param_stat_inv_transform) + param_NS_i)
            else:
                param = param_stat_inv_transform + param_NS_i
            param_ratio = transform_func(param) / param_stat

            if log_scaled:
                param_ratio = np.log(param_ratio)

            # Compute the max over the domain
            param_ratio_abs_max = np.max(np.abs(param_ratio), axis=0)

            probs[i] = np.mean(param_ratio_abs_max < bound)

        return probs

    def evaluate_basis_penalties_for_v(
        self,
        basis: Basis,
        locs: np.ndarray,
        n_realizations: int,
        tau_vals: np.ndarray,
        a_stat: float,
        v_stat: float,
        bound: float = 2.0,
    ) -> np.ndarray:
        param_NS_vx = basis.compute_NS_realization(
            locs=locs, n_realizations=n_realizations
        )

        param_NS_vy = basis.compute_NS_realization(
            locs=locs, n_realizations=n_realizations
        )

        probs = np.zeros(len(tau_vals))

        for i, tau_val in enumerate(tau_vals):
            param_vx_i = v_stat + param_NS_vx * (tau_val ** (-0.5))
            param_vy_i = v_stat + param_NS_vy * (tau_val ** (-0.5))
            corr_range_ratio_i = np.exp(np.sqrt(param_vx_i**2 + param_vy_i**2))
            param_ratio = corr_range_ratio_i / a_stat
            param_ratio_abs_max = np.max(np.abs(param_ratio), axis=0)
            probs[i] = np.mean(param_ratio_abs_max < bound)

        return probs

    def evaluate_basis_penalties_new(
        self,
        basis: Basis,
        locs: np.ndarray,
        n_realizations: int,
        tau_vals: np.ndarray,
        param_stat: float,
        log_scaled: bool = True,
        bound: float = 2.0,
        transform_func: Callable = lambda x: x,
        inv_transform_func: Callable = lambda x: x,
    ) -> np.ndarray:
        probs = np.zeros(len(tau_vals))
        if log_scaled:
            bound = np.log(bound)

        alphas = basis.gen_unscaled_param_realizations(
            Q_diag=basis.Q_diag.numpy(), n_realizations=n_realizations
        )

        for i, tau_val in enumerate(tau_vals):
            # Scale the X based on the tau-val
            alpha = alphas * (tau_val ** (-0.5))

            L2_norms = np.linalg.norm(alpha, axis=0)

            probs[i] = np.mean(L2_norms < bound)

        return probs
