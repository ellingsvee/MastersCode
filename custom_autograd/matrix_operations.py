import time
import traceback
from typing import Optional, Tuple

import torch

from utils.scipy_torch_conversion import scipy_csc_to_torch_coo, torch_coo_to_scipy_csc

_MMBwdReturnType = Tuple[torch.Tensor, torch.Tensor, None]


class SPSPMM(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, A: torch.Tensor, B: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if not A.is_sparse or not B.is_sparse:
            raise ValueError("Both A and B must be sparse tensors.")

        A_coalesced = A.coalesce()
        B_coalesced = B.coalesce()
        C = torch.sparse.mm(A_coalesced, B_coalesced)
        ctx.save_for_backward(A_coalesced, B_coalesced, C)
        return C

    @staticmethod
    def backward(ctx, grad_C: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, None]:
        A, B, C = ctx.saved_tensors

        if not grad_C.is_sparse:
            raise ValueError("grad_C must be a sparse tensor.")

        # Compute sparse gradients
        grad_A = torch.sparse.mm(grad_C, B.T).coalesce()
        grad_B = torch.sparse.mm(A.T, grad_C).coalesce()

        # Apply the masks
        grad_A = grad_A.sparse_mask(A)
        grad_B = grad_B.sparse_mask(B)

        return grad_A, grad_B, None


@torch.compile
def spspmm(
    A: torch.Tensor, B: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if not A.is_sparse or not B.is_sparse:
        raise ValueError("Both A and B must be sparse tensors.")
    return SPSPMM.apply(A, B, mask)


class SPDMM(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, A: torch.Tensor, B: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if not A.is_sparse:
            raise ValueError("A must be a sparse tensor.")
        if B.is_sparse:
            raise ValueError("B must be a dense tensor.")

        A_coalesced = A.coalesce()
        C = torch.sparse.mm(A_coalesced, B)  # Compute sparse matrix multiplication
        ctx.save_for_backward(A_coalesced, B)
        return C

    @staticmethod
    def backward(ctx, grad_C: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, None]:
        A, B = ctx.saved_tensors

        if grad_C.is_sparse:
            raise ValueError("grad_C must be a dense tensor.")

        # Compute sparse gradients
        grad_A = torch.matmul(grad_C, B.T)
        grad_B = torch.matmul(A.T, grad_C)

        # Apply the mask to A
        grad_A = grad_A.sparse_mask(A)

        return grad_A, grad_B, None


@torch.compile
def spdmm(
    A: torch.Tensor, B: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if not A.is_sparse:
        raise ValueError("A must be a sparse tensor.")
    if B.is_sparse:
        raise ValueError("B must be a dense tensor.")
    return SPDMM.apply(A, B, mask)


class ADD(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(A, B)
        return A + B

    @staticmethod
    def backward(ctx, grad_C: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        A, B = ctx.saved_tensors

        # Compute sparse gradients
        grad_A = grad_C.sparse_mask(A)
        grad_B = grad_C.sparse_mask(B)

        return grad_A, grad_B


@torch.compile
def spspadd(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return ADD.apply(A, B)


def spspsub(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return spspadd(A, -B)
