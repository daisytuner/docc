# Sparse Matrix-Vector Multiplication (SpMV)
import pytest
import numpy as np
from benchmarks.npbench.harness import SDFGVerification, run_benchmark, run_pytest

PARAMETERS = {
    "S": {"M": 4096, "N": 4096, "nnz": 8192},
    "M": {"M": 32768, "N": 32768, "nnz": 65536},
    "L": {"M": 262144, "N": 262144, "nnz": 262144},
    "paper": {"M": 131072, "N": 131072, "nnz": 262144},
}


def initialize(M, N, nnz):
    from numpy.random import default_rng

    rng = default_rng(42)

    x = rng.random((N,))

    from scipy.sparse import random

    matrix = random(
        M, N, density=nnz / (M * N), format="csr", dtype=np.float64, random_state=rng
    )
    A_row = np.int64(matrix.indptr)  # Originally np.int32
    A_col = np.int64(matrix.indices)  # Originally np.int32
    A_val = matrix.data

    return A_row, A_col, A_val, x


# Matrix-Vector Multiplication with the matrix given in Compressed Sparse Row
# (CSR) format
def kernel(A_row, A_col, A_val, x):
    y = np.empty(A_row.shape[0] - 1, np.float64)

    for i in range(A_row.shape[0] - 1):
        cols = A_col[A_row[i] : A_row[i + 1]]
        vals = A_val[A_row[i] : A_row[i + 1]]
        y[i] = vals @ x[cols]

    return y


@pytest.mark.parametrize(
    "target",
    [
        "none",
        "sequential",
        "openmp",
        # "cuda"
    ],
)
def test_spmv(target):
    if target == "none":
        verifier = SDFGVerification(
            verification={"DOT": 0, "MAP": 4, "SEQUENTIAL": 4, "FOR": 6, "Malloc": 4}
        )
    elif target == "sequential":
        verifier = SDFGVerification(
            verification={
                "DOT": 0,
                "HIGHWAY": 3,
                "MAP": 4,
                "SEQUENTIAL": 1,
                "FOR": 6,
                "Malloc": 4,
            }
        )
    elif target == "openmp":
        verifier = SDFGVerification(
            verification={"DOT": 0, "CPU_PARALLEL": 1, "HIGHWAY": 2 , "SEQUENTIAL": 1, "MAP": 4, "FOR": 6, "Malloc": 4}
        )
    else:  # cuda
        verifier = SDFGVerification(
            verification={
                "CMath": 14,
                "CUDA": 12,
                "MAP": 18,
                "CUDAOffloading": 34,
                "Memcpy": 4,
                "SEQUENTIAL": 6,
                "FOR": 22,
                "Memset": 1,
                "Malloc": 8,
            }
        )
    run_pytest(initialize, kernel, PARAMETERS, target, verifier=verifier)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "spmv")
