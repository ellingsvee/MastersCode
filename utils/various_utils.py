import math
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.sparse as sparse
import scipy.stats as stats
import torch
from scipy.special import gamma, kv
from sksparse.cholmod import Factor as CholeskyDecomposition
from sksparse.cholmod import cholesky


def matern_covariance(
    h: np.ndarray, kappa2: float, nu: float, sigma2: float
) -> np.ndarray:
    """
    Evaluates the Matern covariance function,

    Parameters
    ----------
    h:
        Distances to evaluate the covariance function at.
    kappa:
        Range parameter.
    nu:
        Shape parameter.
    sigma:
        Standard deviation.

    Returns
    -------
    np.array
        A vector with the values C(h).
    """
    kappa = np.sqrt(kappa2)
    if nu == 1 / 2:
        C = (sigma2) * np.exp(-kappa * np.abs(h))
    else:
        C = (
            ((sigma2) / ((2 ** (nu - 1)) * gamma(nu)))
            * ((kappa * abs(h)) ** nu)
            * kv(nu, kappa * abs(h))
        )
        C[h == 0] = sigma2
    return C


def convert_to_range_and_angle(kappa2_0, nu, vx0, vy0):
    # kappa2_main = np.zeros_like(kappa2_0)
    # kappa2_sec = np.zeros_like(kappa2_0)
    angle = np.zeros_like(kappa2_0)
    main_eval = np.zeros_like(kappa2_0)
    sec_eval = np.zeros_like(kappa2_0)

    vectors = np.column_stack((vx0, vy0))

    for i, v in enumerate(vectors):
        # Step 1: Compute the Euclidean norm
        v_norm = np.linalg.norm(v)

        if v_norm == 0:
            # angle.append(None)
            angle[i] = None
            main_eval[i] = 1
            sec_eval[i] = 1
        else:
            cosh_vals = np.cosh(v_norm)
            sinh_over_magnitude = np.sinh(v_norm) / v_norm

            H00 = cosh_vals + sinh_over_magnitude * v[0]
            H01 = sinh_over_magnitude * v[1]
            H10 = H01
            H11 = cosh_vals - sinh_over_magnitude * v[0]

            H = np.array([[H00, H01], [H10, H11]])

            # Compute the eigenvalues and eigenvectors
            evals, evecs = np.linalg.eig(H)

            # Sort the eigenvalues in descending order
            idx = np.argsort(evals)[::-1]
            evals = evals[idx]
            evecs = evecs[:, idx]

            main_eval[i] = evals[0]
            sec_eval[i] = evals[1]

            main_axis_vec = evecs[:, 0]
            secondary_axis_vec = evecs[:, 1]
            angle_radians = np.arctan2(main_axis_vec[1], main_axis_vec[0])

            angle_degree = np.degrees(angle_radians)
            angle[i] = angle_degree

    range_main = np.sqrt(main_eval * 8 * nu / kappa2_0)
    range_sec = np.sqrt(sec_eval * 8 * nu / kappa2_0)

    # range_main = np.sqrt(8 * nu / kappa2_sec)
    # range_sec = np.sqrt(8 * nu / kappa2_main)
    return range_main, range_sec, angle


def gen_east_west_mask(locs, boarder=-100):
    mask_E = locs[:, 0] > boarder
    mask_W = ~mask_E
    return mask_E, mask_W


def gen_east_west_sigmae2(
    locs, sigmae2_E, sigmae2_W, boarder=-100, return_as_tensor=True
):
    mask_E, mask_W = gen_east_west_mask(locs, boarder)
    # sigmae2_E = sigmae2 * mask_E
    # sigmae2_W = sigmae2 * mask_W

    if return_as_tensor:
        sigmae2_arr = torch.zeros(locs.shape[0], dtype=torch.double)
        sigmae2_arr[mask_E] = sigmae2_E
        sigmae2_arr[mask_W] = sigmae2_W
        return sigmae2_arr
    else:
        sigmae2_arr = np.zeros(locs.shape[0])
        sigmae2_arr[mask_E] = sigmae2_E
        sigmae2_arr[mask_W] = sigmae2_W
        return sigmae2_E, sigmae2_W


# def cholmod_cholesky_decomposition(
#     A: sparse.csc_matrix,
# ) -> CholeskyDecomposition:

#     try:
#         Achol = cholesky(A)
#     except:
#         print("Error: cholesky() failed. Adding values on diagonal.")
#         max_diag = A.diagonal().max()
#         Achol = cholesky(A + 1e-10 * max_diag * sparse.eye(A.shape[0], format="csc"))
#     return Achol


def sample_data(
    N_samples: int,
    data: np.ndarray,
    vertices: np.ndarray,
    mask: np.ndarray,
    seed: int = None,
    return_data_as_tensor: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    if seed is not None:
        np.random.seed(seed)

    # Mask away the locations in the outer part of the domain
    data_masked = data[mask, :]
    locs_masked = vertices[mask, :]

    # Sample indices randomly
    N_obs = data_masked.shape[0]
    rand_locs = np.random.choice(N_obs, N_samples, replace=False)

    # Sample the data and locations
    sampled_data = data_masked[rand_locs, :]
    sampled_locs = locs_masked[rand_locs, :]

    if return_data_as_tensor:
        return torch.from_numpy(sampled_data).double(), sampled_locs
    return sampled_data, sampled_locs


def add_observation_noise(Y: torch.Tensor, sigmae2: torch.Tensor, seed: int = None):
    if seed is not None:
        np.random.seed(seed)
    N = Y.size(0)
    N_realizations = Y.size(1)
    noise = torch.randn(N, N_realizations) * sigmae2.sqrt()
    return Y + noise


def compute_marg_var(
    sims: np.ndarray, mask: np.ndarray, return_mean: bool = True
) -> Union[np.ndarray, float]:
    sims_filtered = sims[mask, :]
    marg_var = np.var(sims_filtered, axis=1)
    if return_mean:
        return np.mean(marg_var)
    return marg_var


def plot_sparsity_pattern(M: torch.Tensor, markersize: float = 0.1):
    if M.is_sparse:
        M = M.to_dense()
    plt.spy(M, markersize=markersize)
    plt.show()


def compute_mean_from_covariates(
    X: torch.Tensor,
    beta: torch.Tensor,
    N_realizations: int,
) -> torch.Tensor:
    return (X @ beta).repeat(N_realizations, 1).T


def compute_corr(Q, Pr, mesh, point_loc):
    Qinv = torch.linalg.inv(Q.to_dense())
    PrQinvPrt = Pr @ Qinv @ Pr.T

    all_locs = mesh.vertices

    # point_loc = np.array([0, 0])
    distances = np.linalg.norm(all_locs - point_loc, axis=1)
    idx = np.argmin(distances)

    corr = PrQinvPrt[idx, :].numpy()

    corr = (corr - corr.min()) / (corr.max() - corr.min())
    return corr
