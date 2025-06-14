import numpy as np
import scipy.sparse as sparse
import torch


def tensor_to_numpy(torch_tensor: torch.Tensor) -> np.ndarray:
    return torch_tensor.detach().numpy()


def numpy_to_tensor(numpy_array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(numpy_array)


def scipy_csc_to_torch_coo(scipy_mat: sparse.csc_matrix) -> torch.Tensor:
    coo_mat = scipy_mat.tocoo()

    indices = np.vstack((coo_mat.row, coo_mat.col))
    indices = torch.from_numpy(indices).long()
    values = torch.from_numpy(coo_mat.data).double()

    return torch.sparse_coo_tensor(
        indices, values, torch.Size(coo_mat.shape)
    ).coalesce()


def torch_coo_to_scipy_csc(torch_sparse_tensor: torch.Tensor) -> sparse.csc_matrix:
    indices = torch_sparse_tensor.detach().coalesce().indices().numpy()
    values = torch_sparse_tensor.detach().coalesce().values().numpy()
    shape = torch_sparse_tensor.shape

    return sparse.coo_matrix(
        (values, (indices[0], indices[1])), shape=shape, dtype=np.double
    ).tocsc()


def scipy_sparse_csc_to_torch_csc(A_scipy: sparse.csc_matrix) -> torch.Tensor:
    values = torch.tensor(A_scipy.data, dtype=torch.double)
    indices = torch.tensor(A_scipy.indices, dtype=torch.int32)
    indptr = torch.tensor(A_scipy.indptr, dtype=torch.int32)
    A_torch = torch.sparse_csc_tensor(indices, indptr, values, size=A_scipy.shape)
    return A_torch


def torch_sparse_csc_to_scipy_csc(A_csc: torch.Tensor) -> sparse.csc_matrix:
    A_scipy = sparse.csc_matrix(
        (
            A_csc.values().numpy(),
            A_csc.row_indices().numpy(),
            A_csc.ccol_indices().numpy(),
        ),
        shape=A_csc.shape,
    )
    return A_scipy
