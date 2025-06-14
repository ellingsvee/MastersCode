"""
Inspired by theseus/optimizer/linear/cholmod_sparse_solver.py in https://github.com/facebookresearch/theseus.git
"""

from typing import Any, Tuple

import torch

try:
    from sksparse.cholmod import Factor as CholeskyDecomposition
    from sksparse.cholmod import analyze
except ModuleNotFoundError:
    import warnings

    warnings.warn("Couldn't import skparse.cholmod. Cholmod solver won't work.")

from custom_autograd.compute_cholesky_decomposition import cholesky_decomp
from custom_autograd.sparse_structure import SparseStructure
from partial_inverse import qinv
from utils.scipy_torch_conversion import scipy_csc_to_torch_coo, torch_coo_to_scipy_csc


class CholmodSolveFunction(torch.autograd.Function):
    """
    NB: The matrix A is assumed to be symmetric positive definite.
    """

    @staticmethod
    def forward(  # type: ignore
        ctx: Any,
        A: torch.Tensor,
        b: torch.Tensor,
        symbolic_decomposition: "CholeskyDecomposition",
    ) -> torch.Tensor:
        A_sp = torch_coo_to_scipy_csc(A)
        A_chol = cholesky_decomp(
            A_sp,
            factor=symbolic_decomposition,
        )

        x = torch.Tensor(A_chol(b)).double()

        ctx.save_for_backward(A, x)
        ctx.A_chol = A_chol
        ctx.A_sp = A_sp

        return x

    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, None]:
        A, x = ctx.saved_tensors
        grad_output_double = grad_output.double()

        gradb = torch.Tensor(ctx.A_chol(grad_output_double)).double()
        gradA = (-gradb @ x.T).sparse_mask(A)

        return gradA, gradb, None


def temp_compute_sparse_solve(A, b, symbolic_decomposition):
    return CholmodSolveFunction.apply(A, b, symbolic_decomposition)


class CholmodLogDetFunction(torch.autograd.Function):
    """
    Custom torch.autograd.Function to compute the log-determinant of a
    symmetric positive definite matrix A.
    """

    @staticmethod
    def forward(  # type: ignore
        ctx: Any,
        A: torch.Tensor,
        symbolic_decomposition: "CholeskyDecomposition",
        use_torch_qinv: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass: computes log-determinant of A.
        """

        # Create sparse matrix A using its values and the sparse structure
        # Perform Cholesky decomposition
        # cholesky_decomposition = symbolic_decomposition.cholesky(A_i, damping)
        A_sp = torch_coo_to_scipy_csc(A)
        A_chol = cholesky_decomp(
            A_sp,
            factor=symbolic_decomposition,
        )

        # Compute log-determinant: sum(log(diagonal of L))
        logdet = torch.tensor(A_chol.logdet()).double()

        # Save for backward
        ctx.save_for_backward(A)
        ctx.A_chol = A_chol
        ctx.A_sp = A_sp
        ctx.use_torch_qinv = use_torch_qinv

        return logdet

    @staticmethod
    def backward(  # type: ignore
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        A = ctx.saved_tensors[0]
        grad_output_double = grad_output.double()
        if ctx.use_torch_qinv:
            Ainv = torch.linalg.inv(A.to_dense()).sparse_mask(A)
        else:
            Ainv = scipy_csc_to_torch_coo(qinv(ctx.A_sp))
        gradA = grad_output_double * Ainv
        return gradA, None, None


def temp_compute_sparse_logdet(A, symbolic_decomposition):
    return CholmodLogDetFunction.apply(A, symbolic_decomposition)


class CholmodSolveAndLogdetFunction(torch.autograd.Function):
    """
    NB: The matrix A is assumed to be symmetric positive definite.
    """

    @staticmethod
    def forward(  # type: ignore
        ctx: Any,
        A: torch.Tensor,
        b: torch.Tensor,
        symbolic_decomposition: "CholeskyDecomposition",
        use_torch_qinv: bool = False,
    ) -> torch.Tensor:
        A_sp = torch_coo_to_scipy_csc(A)
        A_chol = cholesky_decomp(
            A_sp,
            factor=symbolic_decomposition,
        )

        x = torch.Tensor(A_chol(b)).double()
        logdet = torch.tensor(A_chol.logdet()).double()

        ctx.save_for_backward(A, x)
        ctx.A_chol = A_chol
        ctx.A_sp = A_sp
        ctx.use_torch_qinv = use_torch_qinv

        return x, logdet

    # NOTE: in the torch docs the backward is also marked as "staticmethod", I think it makes sense
    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_x: torch.Tensor, grad_logdet: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, None, None]:
        A, x = ctx.saved_tensors

        grad_x_double = grad_x.double()
        grad_logdet_double = grad_logdet.double()

        gradb = torch.Tensor(ctx.A_chol(grad_x_double)).double()
        gradA_solve_syst = (-gradb @ x.T).sparse_mask(A)

        if ctx.use_torch_qinv:
            Ainv = torch.linalg.inv(A.to_dense()).sparse_mask(A)
        else:
            Ainv = scipy_csc_to_torch_coo(qinv(ctx.A_sp))
        gradA_logdet = grad_logdet_double * Ainv

        gradA = gradA_solve_syst + gradA_logdet

        return gradA, gradb, None, None


class CholmodOperations:
    def __init__(
        self,
        sparse_structure: SparseStructure,
        use_torch_qinv: bool = False,
        **kwargs,
    ):
        self.sparse_structure = sparse_structure
        self.use_torch_qinv = use_torch_qinv

        # symbolic decomposition depending on the sparse structure, done with mock data
        self._symbolic_cholesky_decomposition: CholeskyDecomposition = analyze(
            sparse_structure.mock_csc_straight()
        )

    def solve_system(
        self,
        A: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        return CholmodSolveFunction.apply(
            A,
            b,
            self._symbolic_cholesky_decomposition,
        )

    def compute_logdet(
        self,
        A: torch.Tensor,
    ) -> torch.Tensor:
        return CholmodLogDetFunction.apply(
            A,
            self._symbolic_cholesky_decomposition,
            self.use_torch_qinv,
        )

    def solve_system_and_compute_logdet(
        self,
        A: torch.Tensor,
        b: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return CholmodSolveAndLogdetFunction.apply(
            A,
            b,
            self._symbolic_cholesky_decomposition,
            self.use_torch_qinv,
        )
