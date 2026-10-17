#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include "sdfg/data_flow/library_node.h"
#include "sdfg/fp_math/contractions/kernel_builder.h"
#include "sdfg/passes/expansion/lib_node_expander.h"

namespace sdfg {
namespace fp_math {
namespace detail {

/// Largest K whose int8 products, each at most @p magnitude^2, sum exactly in Int32.
int64_t exact_int32_reduction(int64_t magnitude);

/// Whether @p node is a contraction these transformations handle at all: a MatMul or a GEMM node.
bool is_contraction(const data_flow::LibraryNode& node);

/// Why @p node cannot be computed by a low-precision contraction with a reduction of at most @p max_k; empty if it can.
std::string unsupported_contraction(const data_flow::LibraryNode& node, int64_t max_k);

/// K of a contraction, null if @p node is none.
symbolic::Expression contraction_k(const data_flow::LibraryNode& node);

/// K of a supported contraction.
int64_t constant_k(const data_flow::LibraryNode& node);

/// The Float or Double element type of a contraction.
types::PrimitiveType contraction_type(const data_flow::LibraryNode& node);

/// The smallest positive normal value of @p type.
double smallest_normal(types::PrimitiveType type);

/// Reads A and B and writes the result of a contraction under expansion, as matrices indexed [row, column].
class ContractionOperands {
public:
    virtual ~ContractionOperands() = default;

    symbolic::Expression m, n, k;
    types::PrimitiveType element;

    virtual KernelValue a(BlockBuilder& block, const symbolic::Expression& row, const symbolic::Expression& column) = 0;
    virtual KernelValue b(BlockBuilder& block, const symbolic::Expression& row, const symbolic::Expression& column) = 0;

    /// Stores the product @p value (Double) of A and B at [row, column], e.g. as `alpha value + beta C` for a GEMM.
    virtual void store(
        BlockBuilder& block,
        const KernelValue& value,
        const symbolic::Expression& row,
        const symbolic::Expression& column
    ) = 0;
};

using MatrixRead = std::function<KernelValue(BlockBuilder&, const symbolic::Expression&, const symbolic::Expression&)>;

/// `S[o] = max(max_i |X(o, i)|, tiny)` in Double over @p outer x @p inner, where @p read gives X(o, i).
std::string absmax(
    KernelBuilder& kb,
    const std::string& prefix,
    const MatrixRead& read,
    const symbolic::Expression& outer,
    const symbolic::Expression& inner,
    double tiny
);

/// Powers of two around the magnitudes @p magnitudes: `P[o] = 2^e` and `I[o] = 2^-e` with
/// `e = floor(log2(magnitudes[o])) + 1`, so that `|x| < P[o]` for every |x| <= magnitudes[o].
std::pair<std::string, std::string> power_of_two_scales(
    KernelBuilder& kb, const std::string& prefix, const std::string& magnitudes, const symbolic::Expression& count
);

/// The representative of @p value modulo the odd @p modulus in [-(modulus-1)/2, (modulus-1)/2], computed in @p type.
KernelValue symmetric_mod(BlockBuilder& block, const KernelValue& value, int64_t modulus, types::PrimitiveType type);

/// `P = A @ B` for the row-major Int8 buffers A[m, k] and B[k, n] into the Int32 buffer P[m, n], as a MatMul node.
void int8_matmul(
    KernelBuilder& kb,
    const std::string& a,
    const std::string& b,
    const std::string& p,
    const symbolic::Expression& m,
    const symbolic::Expression& n,
    const symbolic::Expression& k
);

/// The MatMul or GEMM node a transformation description refers to as its subgraph node "0".
data_flow::LibraryNode& contraction_from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);

/// Runs @p build as the expansion of the contraction @p node, replacing it with the sequence it builds.
void expand_contraction(
    builder::StructuredSDFGBuilder& builder,
    data_flow::LibraryNode& node,
    const std::function<void(KernelBuilder&, ContractionOperands&)>& build
);

} // namespace detail
} // namespace fp_math
} // namespace sdfg
