from typing import Tuple

import torch

from custom_autograd.matrix_operations import spspadd


def create_diagonal_tensor_coo(values: torch.Tensor) -> torch.Tensor:
    n = values.size(0)

    # Create row and column indices for the diagonal
    indices = (
        torch.arange(n).unsqueeze(0).repeat(2, 1)
    )  # Shape (2, n), same for rows and cols

    # Create the sparse COO tensor
    return torch.sparse_coo_tensor(indices, values, size=(n, n))


def invert_diag_tensor_coo(D: torch.Tensor) -> torch.Tensor:
    D = D.coalesce()
    return torch.sparse_coo_tensor(
        D.indices(),
        1 / D.values(),
        D.shape,
    )


def compute_coo_power(D: torch.Tensor, power: float) -> torch.Tensor:
    D = D.coalesce()
    return torch.sparse_coo_tensor(
        D.indices(),
        D.values() ** power,
        D.shape,
    )


def tensor_to_sparse_coo(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_sparse_coo().coalesce()


def invert_sparse_coo(tensor: torch.Tensor) -> torch.Tensor:
    return tensor_to_sparse_coo(torch.linalg.inv(tensor.to_dense())).coalesce()


def init_empty_coo(size: tuple):
    # Create empty tensors for the indices and values
    indices = torch.empty(2, 0, dtype=torch.long)  # Empty index tensor
    values = torch.empty(0)  # Empty values tensor

    # Initialize the sparse COO matrix with the given size
    sparse_matrix = torch.sparse_coo_tensor(
        indices, values, size, dtype=torch.double
    ).coalesce()

    return sparse_matrix


def place_sparse_in_larger(
    sparse_large: torch.Tensor,
    sparse_small: torch.Tensor,
    target_indices: Tuple[int, int],
):
    """
    Place a small sparse tensor into a larger sparse tensor at specified positions.

    Parameters:
    - sparse_large (torch.sparse.CoalescedTensor): The larger sparse tensor (target).
    - sparse_small (torch.sparse.CoalescedTensor): The smaller sparse tensor to be placed inside the larger tensor.
    - target_indices (tuple): The (row, col) slice or range where to place the smaller tensor.

    Returns:
    - torch.sparse.CoalescedTensor: Updated sparse tensor with the small tensor placed at the specified position.
    """

    sparse_small = sparse_small.coalesce()
    sparse_large = sparse_large.coalesce()

    # Extract indices and values of the small tensor
    indices_small = sparse_small.indices()
    values_small = sparse_small.values()

    # Calculate the offset based on the target position
    row_offset, col_offset = target_indices

    if (
        row_offset + sparse_small.shape[0] > sparse_large.shape[0]
        or col_offset + sparse_small.shape[1] > sparse_large.shape[1]
    ):
        raise ValueError("The small matrix is placed outside the larger.")

    # Shift the indices of the small tensor to the target position in the large matrix
    shifted_indices = indices_small.clone()
    shifted_indices[0] += row_offset  # Apply row offset
    shifted_indices[1] += col_offset  # Apply column offset

    # Extract indices and values of the large tensor
    indices_large = sparse_large.indices()
    values_large = sparse_large.values()

    # Concatenate the indices and values from both tensors
    indices_combined = torch.cat((indices_large, shifted_indices), dim=1)
    values_combined = torch.cat((values_large, values_small), dim=0)

    # Create the updated sparse matrix
    updated_sparse = torch.sparse_coo_tensor(
        indices_combined, values_combined, sparse_large.size()
    ).coalesce()

    return updated_sparse


def enforce_symmetry(M: torch.Tensor) -> torch.Tensor:
    """
    Enforce symmetry on a matrix by averaging with its transpose.

    Parameters:
    - M (torch.Tensor): The matrix to enforce symmetry on.

    Returns:
    - torch.Tensor: The symmetric matrix.
    """
    return 0.5 * spspadd(M, M.T)
