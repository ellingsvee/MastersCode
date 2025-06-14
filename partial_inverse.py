import os
import platform
import subprocess
import tempfile

import numpy as np
import scipy.sparse as sparse
import torch

from utils.scipy_torch_conversion import scipy_csc_to_torch_coo, torch_coo_to_scipy_csc


def run_qinv(Q: sparse.csc_matrix) -> sparse.csc_matrix:
    """
    Calculate sparse inverse subset of symmetric, positive definite matrix Q.
    If the Panua Pardiso solver stops working, the old_qinc/GA_qinv_implemetaion can by used instead.

    Args:
        Q: Symmetric, positive definite sparse matrix

    Returns:
        Qinv: Resulting partial inverse
    """
    # Get row, columns and values (convert to 0-based indexing)
    rows, cols = Q.nonzero()
    values = Q.data

    # Create header
    # (version, elems, nrow, ncol, datatype, valuetp, matrixtype, storagetype)
    header = np.zeros(8, dtype=np.int32)
    header[1] = len(rows)
    header[2] = Q.shape[0]
    header[3] = Q.shape[1]
    header[4] = 1
    header[5] = 1
    header[7] = 1
    h_length = 8

    # Create temporary files
    tmp_to_inla = tempfile.mktemp()
    tmp_from_inla = f"{tmp_to_inla}fromInla"
    tmp_const = tempfile.mktemp()  # Unused but required to make INLA stop complaining

    try:
        # Write to binary file
        with open(tmp_to_inla, "wb") as f:
            # Write header length and header
            np.array([h_length], dtype=np.int32).tofile(f)
            header.tofile(f)

            # Write row, col and values
            rows.astype(np.int32).tofile(f)
            cols.astype(np.int32).tofile(f)
            values.astype(np.float64).tofile(f)

        # Determine executable name based on platform
        # file_path = "/Users/ellingsvee/Documents/GitHub/SPDE2025/INLA/"
        file_path = os.path.abspath(os.curdir) + "/INLA/"
        if platform.system() == "Windows":
            exe_name = file_path + "inla64.exe"
        elif platform.system() == "Linux":
            # exe_name = file_path + "inla64"
            exe_name = file_path + "inla.mkl.run"  # Needed to use this
        elif platform.system() == "Darwin":  # macOS
            exe_name = file_path + "inla.run"
        else:
            raise RuntimeError("Unsupported platform")

        # Run INLA command
        cmd = f"{exe_name} -s -m qinv {tmp_to_inla} {tmp_const} {tmp_from_inla}"

        # Use this to suppress errors:
        # Redirect stdout and stderr to DEVNULL to suppress messages
        with open(os.devnull, "w") as devnull:
            _ = subprocess.run(
                cmd, shell=True, check=True, stdout=devnull, stderr=devnull
            )

        # Read results
        with open(tmp_from_inla, "rb") as f:
            # Read header length and header
            h_length = np.fromfile(f, dtype=np.int32, count=1)[0]
            header = np.fromfile(f, dtype=np.int32, count=8)

            # Read row, column and values
            r_out = np.fromfile(f, dtype=np.int32, count=len(rows))
            c_out = np.fromfile(f, dtype=np.int32, count=len(cols))
            v_out = np.fromfile(f, dtype=np.float64, count=len(values))

        # Create sparse matrix
        Qinv = sparse.coo_matrix((v_out, (r_out, c_out)), shape=Q.shape).tocsc()
        return Qinv

    finally:
        # Clean up temporary files
        for file in [tmp_to_inla, tmp_from_inla, tmp_const]:
            if os.path.exists(file):
                os.remove(file)


def qinv(Q: sparse.csc_matrix) -> sparse.csc_matrix:
    """
    The same as the qinv_run, but with try-catching
    """
    try:
        return run_qinv(Q)
    except Exception as e:
        print("Error in qinv: Returning crazy.")
        return Q * 1e6


def qinv_tensor(A: torch.Tensor, error_handling: bool = True) -> torch.Tensor:
    if error_handling:
        qinv_func = qinv
    else:
        qinv_func = run_qinv
    return scipy_csc_to_torch_coo(qinv_func(torch_coo_to_scipy_csc(A.detach())))


def qinv_tensor_non_R(A: torch.Tensor) -> torch.Tensor:
    return torch.linalg.inv(A.to_dense()).sparse_mask(A)
