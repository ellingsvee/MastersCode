from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np
import scipy.sparse as sparse
import torch
from sksparse.cholmod import Factor as CholeskyDecomposition
from sksparse.cholmod import analyze, analyze_AAt, cholesky

from utils.scipy_torch_conversion import (
    numpy_to_tensor,
    scipy_csc_to_torch_coo,
    tensor_to_numpy,
    torch_coo_to_scipy_csc,
)


def cholesky_decomp(
    A: sparse.csc_matrix,
    factor: Optional["CholeskyDecomposition"] = None,
) -> CholeskyDecomposition:
    if factor is None:
        factor = analyze(A)

    # Try to compute the cholesky decomposition
    try:
        A_chol = factor.cholesky(A)
    except:
        print("Error: cholesky() failed. Adding values on diagonal.")
        max_diag = A.diagonal().max()
        try:
            A_chol = factor.cholesky(
                A + 1e-6 * max_diag * sparse.eye(A.shape[0], format="csc")
            )
        except:
            print("-> Error: cholesky() failed again. Returning cholesky of identity.")
            A_chol = factor.cholesky(sparse.eye(A.shape[0], format="csc"))
    return A_chol


def tensor_cholesky_decomp(
    A: torch.Tensor,
):
    A_csc = torch_coo_to_scipy_csc(A)
    A_chol = cholesky_decomp(A_csc)
    return A_chol


def tensor_solve_system(
    A_chol: CholeskyDecomposition,
    b: torch.Tensor,
):
    b_np = tensor_to_numpy(b)
    x_np = A_chol.solve_A(b)
    x = numpy_to_tensor(x_np)
    return x
