from docc.python import native
import ctypes
import numpy as np


def test_ndarray_shape():
    def test_ndarray_shape_1d():
        @native
        def ndarray_shape_1d(A, B, C):
            for i in range(A.shape[0]):
                C[i] = A[i] + B[i]

        N = 1024
        A = np.random.rand(N).astype(np.float64)
        B = np.random.rand(N).astype(np.float64)
        C = np.zeros(N, dtype=np.float64)

        ndarray_shape_1d(A, B, C)
        assert np.allclose(C, A + B)

        # Check shape arguments
        compiled = ndarray_shape_1d.compile(A, B, C)
        int64_count = sum(1 for t in compiled.arg_types if t == ctypes.c_int64)
        assert int64_count == 1
        assert len(compiled.arg_types) == 4

    def test_ndarray_shape_2d_uniform():
        @native
        def ndarray_shape_2d_uniform(A, B, C):
            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    C[i, j] = A[i, j] + B[i, j]

        N, M = 32, 32
        A = np.random.rand(N, M).astype(np.float64)
        B = np.random.rand(N, M).astype(np.float64)
        C = np.zeros((N, M), dtype=np.float64)

        ndarray_shape_2d_uniform(A, B, C)
        assert np.allclose(C, A + B)

        # Check shape arguments
        compiled = ndarray_shape_2d_uniform.compile(A, B, C)
        int64_count = sum(1 for t in compiled.arg_types if t == ctypes.c_int64)
        assert int64_count == 1
        assert len(compiled.arg_types) == 4

    def test_ndarray_shape_mixed():
        @native
        def ndarray_shape_mixed(A, B):
            for i in range(A.shape[0]):
                B[i] = A[i]

        N = 512
        M = 1024
        A = np.random.rand(N).astype(np.float64)
        B = np.zeros(M, dtype=np.float64)

        ndarray_shape_mixed(A, B)
        assert np.allclose(B[:N], A)
        assert np.allclose(B[N:], 0)

        # Check shape arguments
        compiled = ndarray_shape_mixed.compile(A, B)
        int64_count = sum(1 for t in compiled.arg_types if t == ctypes.c_int64)
        assert int64_count == 2
        assert len(compiled.arg_types) == 4

    test_ndarray_shape_1d()
    test_ndarray_shape_2d_uniform()
    test_ndarray_shape_mixed()


def test_ndarray_strides():
    def test_f_contiguous_rhs():
        @native
        def f_contiguous_rhs(A, B):
            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    B[i, j] = A[i, j]

        N, M = 32, 64
        A = np.asfortranarray(np.random.rand(N, M).astype(np.float64))
        B = np.zeros((N, M), dtype=np.float64, order="C")

        assert not A.flags["C_CONTIGUOUS"]
        assert A.flags["F_CONTIGUOUS"]

        f_contiguous_rhs(A, B)
        assert np.allclose(B, A)

    def test_f_contiguous_lhs():
        """Test writing to a column-major (Fortran order) array"""

        @native
        def f_contiguous_lhs(A, B):
            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    B[i, j] = A[i, j]

        N, M = 32, 64
        A = np.random.rand(N, M).astype(np.float64)
        B = np.asfortranarray(np.zeros((N, M), dtype=np.float64))

        assert not B.flags["C_CONTIGUOUS"]
        assert B.flags["F_CONTIGUOUS"]

        f_contiguous_lhs(A, B)
        assert np.allclose(B, A)

    def test_f_contiguous():
        """Test both input and output in column-major order"""

        @native
        def f_contiguous(A, B, C):
            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    C[i, j] = A[i, j] + B[i, j]

        N, M = 32, 64
        A = np.asfortranarray(np.random.rand(N, M).astype(np.float64))
        B = np.asfortranarray(np.random.rand(N, M).astype(np.float64))
        C = np.asfortranarray(np.zeros((N, M), dtype=np.float64))

        f_contiguous(A, B, C)
        assert np.allclose(C, A + B)

    def test_transposed_array():
        """Test using a transposed view (swapped strides)"""

        @native
        def transposed_array(A, B):
            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    B[i, j] = A[i, j]

        N, M = 32, 64
        A_base = np.random.rand(M, N).astype(np.float64)
        A = A_base.T  # Shape (N, M), but strides are swapped
        B = np.zeros((N, M), dtype=np.float64)

        assert A.shape == (N, M)
        assert A.strides[0] < A.strides[1]  # Row stride < column stride

        transposed_array(A, B)
        assert np.allclose(B, A)

    def test_reversed_1d():
        """Test 1D array with negative stride (reversed view)"""

        @native
        def reversed_1d(A, B):
            for i in range(A.shape[0]):
                B[i] = A[i]

        N = 256
        A_base = np.random.rand(N).astype(np.float64)
        A = A_base[::-1]  # Reversed view with negative stride
        B = np.zeros(N, dtype=np.float64)

        assert A.strides[0] < 0

        reversed_1d(A, B)
        assert np.allclose(B, A)

    def test_reversed_2d_rows():
        """Test 2D array with rows reversed (negative row stride)"""

        @native
        def reversed_2d_rows(A, B):
            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    B[i, j] = A[i, j]

        N, M = 32, 64
        A_base = np.random.rand(N, M).astype(np.float64)
        A = A_base[::-1, :]  # Rows reversed
        B = np.zeros((N, M), dtype=np.float64)

        assert A.strides[0] < 0

        reversed_2d_rows(A, B)
        assert np.allclose(B, A)

    def test_reversed_2d_cols():
        """Test 2D array with columns reversed (negative column stride)"""

        @native
        def reversed_2d_cols(A, B):
            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    B[i, j] = A[i, j]

        N, M = 32, 64
        A_base = np.random.rand(N, M).astype(np.float64)
        A = A_base[:, ::-1]  # Columns reversed
        B = np.zeros((N, M), dtype=np.float64)

        assert A.strides[1] < 0

        reversed_2d_cols(A, B)
        assert np.allclose(B, A)

    def test_reversed_2d_both():
        """Test 2D array with both dimensions reversed"""

        @native
        def reversed_2d_both(A, B):
            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    B[i, j] = A[i, j]

        N, M = 32, 64
        A_base = np.random.rand(N, M).astype(np.float64)
        A = A_base[::-1, ::-1]  # Both reversed
        B = np.zeros((N, M), dtype=np.float64)

        assert A.strides[0] < 0
        assert A.strides[1] < 0

        reversed_2d_both(A, B)
        assert np.allclose(B, A)

    def test_sliced_non_contiguous():
        """Test sliced array with non-contiguous memory (every 2nd element)"""

        @native
        def sliced_non_continguous(A, B):
            for i in range(A.shape[0]):
                B[i] = A[i]

        N = 256
        A_base = np.random.rand(N * 2).astype(np.float64)
        A = A_base[::2]  # Every 2nd element
        B = np.zeros(N, dtype=np.float64)

        assert A.strides[0] == 16  # 2 * sizeof(float64)
        assert not A.flags["C_CONTIGUOUS"]

        sliced_non_continguous(A, B)
        assert np.allclose(B, A)

    def test_sliced_2d_rows():
        """Test 2D array sliced to every 2nd row"""

        @native
        def sliced_2d_rows(A, B):
            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    B[i, j] = A[i, j]

        N, M = 64, 32
        A_base = np.random.rand(N, M).astype(np.float64)
        A = A_base[::2, :]  # Every 2nd row
        B = np.zeros((N // 2, M), dtype=np.float64)

        assert not A.flags["C_CONTIGUOUS"]

        sliced_2d_rows(A, B)
        assert np.allclose(B, A)

    def test_sliced_2d_cols():
        """Test 2D array sliced to every 3rd column"""

        @native
        def sliced_2d_cols(A, B):
            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    B[i, j] = A[i, j]

        N, M = 32, 63
        A_base = np.random.rand(N, M).astype(np.float64)
        A = A_base[:, ::3]  # Every 3rd column
        B = np.zeros((N, M // 3), dtype=np.float64)

        assert not A.flags["C_CONTIGUOUS"]

        sliced_2d_cols(A, B)
        assert np.allclose(B, A)

    def test_mixed_strides_operation():
        """Test operation between arrays with different stride patterns"""

        @native
        def mixed_strides_operation(A, B, C):
            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    C[i, j] = A[i, j] + B[i, j]

        N, M = 32, 64
        A = np.random.rand(N, M).astype(np.float64)  # C-order
        B = np.asfortranarray(np.random.rand(N, M).astype(np.float64))  # F-order
        C = np.zeros((N, M), dtype=np.float64)

        assert A.flags["C_CONTIGUOUS"]
        assert B.flags["F_CONTIGUOUS"]

        mixed_strides_operation(A, B, C)
        assert np.allclose(C, A + B)

    def test_3d_non_standard_strides():
        """Test 3D array with non-standard strides"""

        @native
        def non_standard_strides_3d(A, B):
            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    for k in range(A.shape[2]):
                        B[i, j, k] = A[i, j, k]

        D1, D2, D3 = 8, 16, 32
        A = np.asfortranarray(np.random.rand(D1, D2, D3).astype(np.float64))
        B = np.zeros((D1, D2, D3), dtype=np.float64, order="C")

        assert A.flags["F_CONTIGUOUS"]

        non_standard_strides_3d(A, B)
        assert np.allclose(B, A)

    def test_3d_transposed():
        """Test 3D transposed array with permuted axes"""

        @native
        def transposed_3d(A, B):
            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    for k in range(A.shape[2]):
                        B[i, j, k] = A[i, j, k]

        D1, D2, D3 = 8, 16, 32
        A_base = np.random.rand(D3, D2, D1).astype(np.float64)
        A = A_base.transpose(2, 1, 0)  # Permute axes
        B = np.zeros((D1, D2, D3), dtype=np.float64)

        assert A.shape == (D1, D2, D3)

        transposed_3d(A, B)
        assert np.allclose(B, A)

    # Run all sub-tests
    test_f_contiguous_rhs()
    test_f_contiguous_lhs()
    test_f_contiguous()
    test_transposed_array()
    test_reversed_1d()
    test_reversed_2d_rows()
    test_reversed_2d_cols()
    test_reversed_2d_both()
    test_sliced_non_contiguous()
    test_sliced_2d_rows()
    test_sliced_2d_cols()
    test_mixed_strides_operation()
    test_3d_non_standard_strides()
    test_3d_transposed()
