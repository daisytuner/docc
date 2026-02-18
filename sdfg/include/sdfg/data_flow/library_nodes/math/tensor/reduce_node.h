/**
 * @file reduce_node.h
 * @brief Tensor reduction operation nodes
 *
 * This file defines the base class for tensor reduction operations. Reductions
 * are operations that aggregate values along one or more tensor dimensions.
 *
 * ## Tensor Library Nodes
 *
 * ReduceNode expects **scalars or flat pointers of scalars** as inputs.
 * The tensor operation is performed with **linearized indices**.
 *
 * ## Reduction Operations
 *
 * Reduction operations include:
 * - Sum, mean, std: Statistical reductions
 * - Min, max: Value reductions
 * - Softmax: Exponential normalization
 *
 * Reductions can operate on:
 * - Specific axes (e.g., reduce along dimension 1)
 * - Multiple axes (e.g., reduce along dimensions 0 and 2)
 * - All axes (full reduction to scalar)
 *
 * The keepdims parameter controls whether reduced dimensions are kept with size 1
 * or removed from the output shape.
 *
 * ## Example
 *
 * Creating a sum reduction:
 * @code
 * // Reduce along last axis of shape [32, 64, 128]
 * std::vector<symbolic::Expression> shape = {
 *     symbolic::integer(32), symbolic::integer(64), symbolic::integer(128)
 * };
 * std::vector<int64_t> axes = {-1};  // Last axis
 * bool keepdims = false;
 *
 * auto& sum_node = builder.add_library_node<math::tensor::SumNode>(
 *     block, debug_info, shape, axes, keepdims
 * );
 *
 * // Output shape will be [32, 64] since last axis is reduced
 * @endcode
 *
 * @see math::tensor::ElementWiseUnaryNode for elementwise operations
 * @see math::MathNode for expansion interface
 */

#pragma once

#include "sdfg/data_flow/library_nodes/math/tensor/tensor_node.h"

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/serializer/json_serializer.h"

namespace sdfg {
namespace math {
namespace tensor {

/**
 * @class ReduceNode
 * @brief Base class for tensor reduction operations
 *
 * ReduceNode represents operations that reduce a tensor along specified axes
 * by aggregating values. The general form is:
 *   Y = reduce(X, axes, reduction_function)
 *
 * Where the reduction_function (sum, max, mean, etc.) is applied along the
 * specified axes. Input and output use linearized indexing.
 *
 * Derived classes implement specific reductions (sum, max, mean, etc.) by
 * providing:
 * - expand_reduction: Method to generate the reduction computation
 * - identity: Identity value for the reduction operation
 *
 * ## Input/Output Requirements
 * - Input connector: "X" (scalar or flat pointer to scalar)
 * - Output connector: "Y" (scalar or flat pointer to scalar)
 * - Input shape: Multi-dimensional logical shape
 * - Output shape: Input shape with reduced axes removed or set to 1
 * - Indexing: Linearized (flat) memory layout
 */
class ReduceNode : public TensorNode {
protected:
    std::vector<symbolic::Expression> shape_; ///< Input tensor shape
    std::vector<int64_t> axes_; ///< Axes to reduce over
    bool keepdims_; ///< Whether to keep reduced dimensions with size 1

public:
    /**
     * @brief Construct a reduction node
     * @param element_id Unique element identifier
     * @param debug_info Debug information
     * @param vertex Graph vertex
     * @param parent Parent dataflow graph
     * @param code Operation code
     * @param shape Input tensor shape
     * @param axes Axes to reduce (negative values index from end)
     * @param keepdims Whether to keep reduced dimensions with size 1
     */
    ReduceNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const data_flow::LibraryNodeCode& code,
        const std::vector<symbolic::Expression>& shape,
        const std::vector<int64_t>& axes,
        bool keepdims
    );

    void validate(const Function& function) const override;

    /**
     * @brief Get the input tensor shape
     * @return Input tensor shape
     */
    const std::vector<symbolic::Expression>& shape() const { return shape_; }

    /**
     * @brief Get the reduction axes
     * @return Axes to reduce over
     */
    const std::vector<int64_t>& axes() const { return axes_; }

    /**
     * @brief Check if reduced dimensions are kept
     * @return True if keepdims is enabled
     */
    bool keepdims() const { return keepdims_; }

    symbolic::SymbolSet symbols() const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    /**
     * @brief Expand into nested maps with reduction logic
     *
     * Creates maps over non-reduced dimensions and reduction loops over reduced
     * dimensions with appropriate initialization and accumulation.
     *
     * @param builder SDFG builder
     * @param analysis_manager Analysis manager
     * @return True if expansion succeeded
     */
    bool expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    /**
     * @brief Generate the actual reduction code
     *
     * Subclasses implement this to generate the specific reduction (sum, max, etc.)
     *
     * @param builder SDFG builder
     * @param analysis_manager Analysis manager
     * @param body Sequence to add the reduction to
     * @param input_name Input data name
     * @param output_name Output data name
     * @param input_type Input data type
     * @param output_type Output data type
     * @param input_subset Input data subset
     * @param output_subset Output data subset
     * @return True if reduction generation succeeded
     */
    virtual bool expand_reduction(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::Sequence& body,
        const std::string& input_name,
        const std::string& output_name,
        const types::Tensor& input_type,
        const types::Tensor& output_type,
        const data_flow::Subset& input_subset,
        const data_flow::Subset& output_subset
    ) = 0;

    /**
     * @brief Get the identity value for this reduction
     *
     * The identity value is used to initialize the accumulator.
     * Examples:
     * - Sum: "0"
     * - Product: "1"
     * - Max: "-inf" or minimum value
     * - Min: "inf" or maximum value
     *
     * @param primitive_type The primitive type of the reduction element
     * @return Identity value as string expression
     */
    virtual std::string identity(types::PrimitiveType primitive_type) const = 0;
};

template<typename T>
class ReduceNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override {
        const ReduceNode& reduce_node = static_cast<const ReduceNode&>(library_node);
        nlohmann::json j;

        j["code"] = reduce_node.code().value();

        serializer::JSONSerializer serializer;
        j["shape"] = nlohmann::json::array();
        for (auto& dim : reduce_node.shape()) {
            j["shape"].push_back(serializer.expression(dim));
        }

        j["axes"] = reduce_node.axes();
        j["keepdims"] = reduce_node.keepdims();

        return j;
    }

    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override {
        // Assertions for required fields
        assert(j.contains("element_id"));
        assert(j.contains("code"));
        assert(j.contains("debug_info"));
        assert(j.contains("shape"));
        assert(j.contains("axes"));
        assert(j.contains("keepdims"));

        auto code = j["code"].get<std::string>();

        std::vector<symbolic::Expression> shape;
        for (const auto& dim : j["shape"]) {
            shape.push_back(symbolic::parse(dim.get<std::string>()));
        }

        std::vector<int64_t> axes = j["axes"].get<std::vector<int64_t>>();
        bool keepdims = j["keepdims"].get<bool>();

        // Extract debug info using JSONSerializer
        sdfg::serializer::JSONSerializer serializer;
        DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

        return static_cast<ReduceNode&>(builder.add_library_node<T>(parent, debug_info, shape, axes, keepdims));
    }
};

} // namespace tensor
} // namespace math
} // namespace sdfg
