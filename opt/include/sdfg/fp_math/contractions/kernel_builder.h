#pragma once

#include <memory>
#include <string>
#include <vector>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace fp_math {

/// A value in a block under construction: an access node with the subset and base type of the memlet reading it.
struct KernelValue {
    data_flow::AccessNode* node = nullptr;
    data_flow::Subset subset;
    std::shared_ptr<types::IType> base_type;
};

/**
 * @brief Builds the dataflow of one block from expressions.
 *
 * Every operation writes its result to a new transient scalar, so values combine freely, e.g.
 * `store(op(fp_mul, {read("x", {i}), constant(2.0, Double)}, Double), "y", {i})`.
 */
class BlockBuilder {
private:
    builder::StructuredSDFGBuilder& builder_;
    structured_control_flow::Block& block_;
    DebugInfo debug_info_;

    KernelValue result(types::PrimitiveType type);
    void connect(const KernelValue& value, data_flow::CodeNode& node, const std::string& connector);

public:
    BlockBuilder(builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& block, DebugInfo debug_info);

    structured_control_flow::Block& block() {
        return block_;
    }

    /// Reads @p node at @p subset through a memlet of @p base_type.
    KernelValue read(data_flow::AccessNode& node, const data_flow::Subset& subset, const types::IType& base_type);

    /// Reads a scalar container, or the element at @p subset of a pointer container.
    KernelValue read(const std::string& container, const data_flow::Subset& subset = {});

    KernelValue constant(double value, types::PrimitiveType type);

    /// Converts @p value to @p type; a value already of that type is returned as is.
    KernelValue convert(const KernelValue& value, types::PrimitiveType type);

    KernelValue op(data_flow::TaskletCode code, const std::vector<KernelValue>& inputs, types::PrimitiveType type);

    KernelValue
    cmath(math::cmath::CMathFunction function, const std::vector<KernelValue>& inputs, types::PrimitiveType type);

    /// Assigns @p value, converted to the target's element type, to @p node at @p subset.
    void store(
        const KernelValue& value,
        data_flow::AccessNode& node,
        const data_flow::Subset& subset,
        const types::IType& base_type
    );

    /// Assigns @p value to a scalar container, or to the element at @p subset of a pointer container.
    void store(const KernelValue& value, const std::string& container, const data_flow::Subset& subset = {});
};

/**
 * @brief Builds loop nests over transient buffers into a sequence.
 *
 * Buffers are allocated where they are requested, in order, and freed by finish().
 */
class KernelBuilder {
private:
    builder::StructuredSDFGBuilder& builder_;
    structured_control_flow::Sequence& scope_;
    DebugInfo debug_info_;
    std::vector<std::string> buffers_;

public:
    KernelBuilder(builder::StructuredSDFGBuilder& builder, structured_control_flow::Sequence& scope, DebugInfo debug_info);

    builder::StructuredSDFGBuilder& builder() {
        return builder_;
    }

    structured_control_flow::Sequence& scope() {
        return scope_;
    }

    const DebugInfo& debug_info() const {
        return debug_info_;
    }

    /// A new transient scalar.
    std::string scalar(const std::string& prefix, types::PrimitiveType type);

    /// A new transient buffer of @p count elements, allocated in the scope right away.
    std::string buffer(const std::string& prefix, types::PrimitiveType type, const symbolic::Expression& count);

    /// A sequential map counting from zero up to @p bound in @p scope.
    structured_control_flow::StructuredLoop&
    map(structured_control_flow::Sequence& scope, const std::string& prefix, const symbolic::Expression& bound);

    /// A for loop counting from zero up to @p bound in @p scope.
    structured_control_flow::StructuredLoop&
    for_loop(structured_control_flow::Sequence& scope, const std::string& prefix, const symbolic::Expression& bound);

    BlockBuilder block(structured_control_flow::Sequence& scope);

    /// Frees all buffers at the end of the scope.
    void finish();
};

} // namespace fp_math
} // namespace sdfg
