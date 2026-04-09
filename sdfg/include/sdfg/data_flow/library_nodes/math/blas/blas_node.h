/**
 * @file blas_node.h
 * @brief BLAS (Basic Linear Algebra Subprograms) library node definitions
 *
 * This file defines base classes and enumerations for BLAS operations.
 * BLAS nodes are mathematical library nodes that can be dispatched to various
 * BLAS implementations (CPU BLAS, CUBLAS, etc.) based on their implementation type.
 *
 * ## BLAS Implementation Types
 *
 * BLAS nodes support multiple implementation types:
 * - ImplementationType_BLAS: Standard CPU BLAS (e.g., OpenBLAS, MKL)
 * - ImplementationType_CUDAWithTransfers: CUDA with automatic data transfers
 * - ImplementationType_CUDAWithoutTransfers: CUDA assuming data is on GPU
 *
 * The implementation type is used by dispatchers to generate appropriate library calls.
 *
 * ## BLAS Operations
 *
 * Common BLAS operations include:
 * - Level 1: Vector operations (dot product)
 * - Level 2: Matrix-vector operations
 * - Level 3: Matrix-matrix operations (GEMM)
 *
 * @see math::blas::GemmNode for matrix multiplication
 * @see math::blas::DotNode for dot product
 * @see math::MathNode for expansion interface
 */

#pragma once

#include "sdfg/data_flow/library_nodes/math/math_node.h"

namespace sdfg {
namespace math {
namespace blas {

/**
 * @enum BLAS_Precision
 * @brief Precision/data type for BLAS operations
 *
 * Indicates the floating-point precision of the operation:
 * - h: Half precision (16-bit)
 * - s: Single precision (32-bit float)
 * - d: Double precision (64-bit double)
 * - c: Complex single precision
 * - z: Complex double precision
 */
enum BLAS_Precision {
    h = 'h',
    s = 's',
    d = 'd',
    c = 'c',
    z = 'z',
};

constexpr std::string_view BLAS_Precision_to_string(BLAS_Precision precision) {
    switch (precision) {
        case BLAS_Precision::h:
            return "h";
        case BLAS_Precision::s:
            return "s";
        case BLAS_Precision::d:
            return "d";
        case BLAS_Precision::c:
            return "c";
        case BLAS_Precision::z:
            return "z";
        default:
            throw std::runtime_error("Invalid BLAS_Precision value");
    }
}

/**
 * @enum BLAS_Transpose
 * @brief Transpose operation for BLAS matrices
 *
 * Specifies whether to use a matrix as-is or transposed:
 * - No: No transpose (111)
 * - Trans: Transpose (112)
 * - ConjTrans: Conjugate transpose (113)
 */
enum BLAS_Transpose {
    No = 111,
    Trans = 112,
    ConjTrans = 113,
};

constexpr std::string_view BLAS_Transpose_to_string(BLAS_Transpose transpose) {
    switch (transpose) {
        case BLAS_Transpose::No:
            return "CblasNoTrans";
        case BLAS_Transpose::Trans:
            return "CblasTrans";
        case BLAS_Transpose::ConjTrans:
            return "CblasConjTrans";
        default:
            throw std::runtime_error("Invalid BLAS_Transpose value");
    }
}

inline constexpr char BLAS_Transpose_to_char(BLAS_Transpose transpose) {
    switch (transpose) {
        case BLAS_Transpose::No:
            return 'N';
        case BLAS_Transpose::Trans:
            return 'T';
        case BLAS_Transpose::ConjTrans:
            return 'C';
        default:
            throw std::runtime_error("Invalid BLAS_Transpose value");
    }
}

/**
 * @enum BLAS_Layout
 * @brief Memory layout for BLAS matrices
 *
 * Specifies how matrix elements are stored in memory:
 * - RowMajor: Rows are contiguous (C-style, 101)
 * - ColMajor: Columns are contiguous (Fortran-style, 102)
 */
enum BLAS_Layout {
    RowMajor = 101,
    ColMajor = 102,
};

constexpr std::string_view BLAS_Layout_to_string(BLAS_Layout layout) {
    switch (layout) {
        case BLAS_Layout::RowMajor:
            return "CblasRowMajor";
        case BLAS_Layout::ColMajor:
            return "CblasColMajor";
    }
}

inline constexpr std::string_view BLAS_Layout_to_short_string(BLAS_Layout layout) {
    switch (layout) {
        case BLAS_Layout::RowMajor:
            return "RowM";
        case BLAS_Layout::ColMajor:
            return "ColM";
        default:
            throw std::runtime_error("Invalid BLAS_Layout value");
    }
}

/**
 * @brief BLAS implementation type
 * Uses standard CPU BLAS libraries (OpenBLAS, Intel MKL, etc.)
 */
inline data_flow::ImplementationType ImplementationType_BLAS{"BLAS"};

/**
 * @class BLASNode
 * @brief Base class for BLAS operation nodes
 *
 * BLASNode extends MathNode with BLAS-specific properties like precision.
 * All BLAS nodes have a precision type that determines the data type and
 * which BLAS routine variant to call (e.g., dgemm for double, sgemm for single).
 *
 * BLAS nodes typically do not expand into primitive operations but instead
 * are dispatched to library calls based on their implementation_type.
 */
class BLASNode : public math::MathNode {
protected:
    BLAS_Precision precision_; ///< Floating-point precision for the operation

public:
    /**
     * @brief Construct a BLAS node
     * @param element_id Unique element identifier
     * @param debug_info Debug information
     * @param vertex Graph vertex
     * @param parent Parent dataflow graph
     * @param code Operation code
     * @param outputs Output connector names
     * @param inputs Input connector names
     * @param implementation_type Implementation type (BLAS, CUBLAS, etc.)
     * @param precision Floating-point precision
     */
    BLASNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const data_flow::LibraryNodeCode& code,
        const std::vector<std::string>& outputs,
        const std::vector<std::string>& inputs,
        const data_flow::ImplementationType& implementation_type,
        const BLAS_Precision& precision
    );

    /**
     * @brief Get the precision/data type
     * @return BLAS precision
     */
    BLAS_Precision precision() const;

    /**
     * @brief Get the corresponding scalar primitive type
     * @return Primitive type for this precision
     */
    types::PrimitiveType scalar_primitive() const;
};

} // namespace blas
} // namespace math
} // namespace sdfg
