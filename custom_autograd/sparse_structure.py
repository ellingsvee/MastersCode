import abc
import numpy as np
import torch
import scipy.sparse as sparse


class SparseStructure(abc.ABC):
    def __init__(
        self,
        col_ind: np.ndarray,
        row_ptr: np.ndarray,
        num_rows: int,
        num_cols: int,
        dtype: np.dtype = np.float64,  # type: ignore
    ):
        self.col_ind = col_ind
        self.row_ptr = row_ptr
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.dtype = dtype

        # Tensor properties
        self.col_ind_tensor = torch.from_numpy(self.col_ind).long()
        self.row_ptr_tensor = torch.from_numpy(self.row_ptr).long()

    def csr_straight(self, val: torch.Tensor) -> sparse.csr_matrix:
        return sparse.csr_matrix(
            (val, self.col_ind, self.row_ptr),
            (self.num_rows, self.num_cols),
            dtype=self.dtype,
        )

    def csc_straight(self, val: torch.Tensor) -> sparse.csc_matrix:
        return self.csr_straight(val).tocsc()

    def csc_transpose(self, val: torch.Tensor) -> sparse.csc_matrix:
        return sparse.csc_matrix(
            (val, self.col_ind, self.row_ptr),
            (self.num_cols, self.num_rows),
            dtype=self.dtype,
        )

    def mock_csr_straight(self) -> sparse.csr_matrix:
        return sparse.csr_matrix(
            (np.ones(len(self.col_ind), dtype=self.dtype), self.col_ind, self.row_ptr),
            (self.num_rows, self.num_cols),
            dtype=self.dtype,
        )

    def mock_csc_straight(self) -> sparse.csc_matrix:
        return self.mock_csr_straight().tocsc()

    def mock_csc_transpose(self) -> sparse.csc_matrix:
        return sparse.csc_matrix(
            (np.ones(len(self.col_ind), dtype=self.dtype), self.col_ind, self.row_ptr),
            (self.num_cols, self.num_rows),
            dtype=self.dtype,
        )


def torch_sparse_coo_to_sparse_structure(
    A: torch.Tensor,
) -> SparseStructure:
    A_csr = A.to_sparse_csr()
    return SparseStructure(
        col_ind=A_csr.col_indices().numpy(),
        row_ptr=A_csr.crow_indices().numpy(),
        num_rows=A_csr.shape[0],
        num_cols=A_csr.shape[1],
    )
