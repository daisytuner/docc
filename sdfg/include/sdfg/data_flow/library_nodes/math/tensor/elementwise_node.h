/**
 * @file elementwise_node.h
 * @brief Tensor elementwise operation nodes
 *
 * This file defines base classes for tensor elementwise operations. Elementwise
 * operations are mathematical operations applied independently to each element
 * of a tensor or pair of tensors.
 *
 * ## Tensor Library Nodes
 *
 * Tensor library nodes expect **scalars or flat pointers of scalars** as inputs.
 * The tensor operation is performed with **linearized indices**. This means:
 * - Multi-dimensional tensor operations are represented using 1D indexing
 * - The shape parameter specifies the logical dimensions
 * - Data access uses linearized (flat) memory layout
 *
 * For example, a 2D tensor of shape [M, N] is accessed using index `i*N + j`
 * where `i` and `j` are the row and column indices.
 *
 * ## Elementwise Operations
 *
 * Elementwise operations include:
 * - Unary operations: abs, sqrt, exp, tanh, sigmoid, relu, etc.
 * - Binary operations: add, sub, mul, div, pow, etc.
 *
 * These operations are expanded into maps that iterate over the tensor shape
 * with linearized indexing.
 *
 * ## Example
 *
 * Creating an elementwise addition:
 * @code
 * // Create tensor addition node for shape [32, 64]
 * std::vector<symbolic::Expression> shape = {
 *     symbolic::integer(32), symbolic::integer(64)
 * };
 * auto& add_node = builder.add_library_node<math::tensor::AddNode>(
 *     block, debug_info, shape
 * );
 *
 * // Connect flat pointer inputs
 * types::Scalar element_type(types::PrimitiveType::Float);
 * types::Pointer ptr_type(element_type);
 * builder.add_computational_memlet(block, input_a, add_node, "A", {}, ptr_type, debug_info);
 * builder.add_computational_memlet(block, input_b, add_node, "B", {}, ptr_type, debug_info);
 * builder.add_computational_memlet(block, add_node, "Y", output, {}, ptr_type, debug_info);
 *
 * // Expand into map with linearized indexing
 * analysis::AnalysisManager analysis_manager(sdfg);
 * add_node.expand(builder, analysis_manager);
 * @endcode
 *
 * @see math::tensor::ReduceNode for reduction operations
 * @see math::MathNode for expansion interface
 */

#pragma once

#include "sdfg/data_flow/library_nodes/math/tensor/tensor_node.h"

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/structured_control_flow/block.h"

namespace sdfg {

namespace math {
namespace tensor {

/**
 * @class ElementWiseDataflowTensorNode
 * Defined by each element of the output tensor (defined by shape) can be defined independently of any other output.
 * Inputs may have lower dimensionality then the output, but must never overlap with the output if its not a 1:1
 * mapping. For now only scalar or output shape is supported.
 *
 * The inputs are required to be listed as (dest_ptr, [other_inputs...]). input_.at(0) always is the tensor that is
 * written to
 */
class ElementWiseDataflowTensorNode : public TensorNode {
protected:
    QuantizationType fixed_quantization_;
    std::vector<symbolic::Expression> shape_; ///< Logical tensor shape

public:
    struct ElementOutput {
        CodeNode* producer = nullptr;
        int output_conn_index = -1;
        types::PrimitiveType type = types::Void;
    };
    struct ElementInput {
        CodeNode* consumer = nullptr;
        int input_conn_index = -1;
        types::PrimitiveType required_type = types::Void;
    };

    ElementWiseDataflowTensorNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const data_flow::LibraryNodeCode& code,
        const std::vector<symbolic::Expression>& shape,
        const std::string& modified_tensor_conn,
        const std::vector<std::string>& tensor_inputs,
        QuantizationType quantization = QUANTIZATION_MATCH_INPUTS,
        const data_flow::ImplementationType& impl_type = data_flow::ImplementationType_NONE
    );

    static std::vector<std::string>
    build_input_conns(const std::string& modified_tensor_conn, const std::vector<std::string>& inputs);

    /**
     * How many inputs, starting with 0 are full tensors?
     * input0 always has to be.
     * Further inputs that are not tensors are by default required to be scalar-only
     * Override validate_non_tensor_inputs() to change this.
     * [expand_operation_dataflow] is given the inputs that exist and expected to handle whatever is valid without
     * errors
     */
    virtual int tensor_input_count() const { return inputs_.size(); }

    /**
     * How many inputs, starting with 0 are optional?
     * input0 always has to be present.
     * Further inputs will be stripped out of the available inputs if not connected
     * Override validate_non_tensor_inputs() to match this
     * [expand_operation_dataflow] is given the inputs that exist and expected to handle whatever is valid without
     * errors
     */
    virtual int mandatory_input_count() const { return inputs_.size(); }

    QuantizationType quantization() const { return quantization(get_parent()); }

    /**
     * type of the math calculations. May be inferred or fixed.
     */
    QuantizationType quantization(const data_flow::DataFlowGraph& dataflow) const;

    /**
     * Same result as quantization if it matches all the inputs. None if its impossible to use the same types
     * for input & output and math
     */
    std::optional<QuantizationType> uniform_quantization(const data_flow::DataFlowGraph& dataflow) const;

    /**
     * configuration of the type for the math calculations, independent of current input types etc.
     * 'Void' indicates auto-inferring from inputs
     */
    QuantizationType fixed_quantization() const;

    void set_fixed_quantization(const QuantizationType quant);

    void validate_target_tensor(const data_flow::DataFlowGraph& graph) const;

    void validate_all_input_tensors(const data_flow::DataFlowGraph& graph) const;

    /**
     * by default, everything after the tensors is mandated to be scalar. Override to change this.
     */
    virtual void validate_non_tensor_inputs(const data_flow::DataFlowGraph& graph) const;

    void validate(const Function& function) const override;

    /**
     * @brief Get the tensor shape
     * @return Logical tensor shape
     */
    const std::vector<symbolic::Expression>& shape() const { return shape_; }

    symbolic::SymbolSet symbols() const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    void replace(const symbolic::ExpressionMapping& replacements) override;

    static std::pair<structured_control_flow::Sequence*, std::vector<symbolic::Expression>> add_eltwise_scope(
        builder::StructuredSDFGBuilder& builder,
        const DebugInfo& scope_deb_info,
        Sequence& parent,
        const std::vector<symbolic::Expression>& shape
    );

    static std::unique_ptr<types::IType> access_type(const std::pair<types::PrimitiveType, const TensorLayout*>& pair);

    static bool create_input(
        builder::StructuredSDFGBuilder& builder,
        structured_control_flow::Block& block,
        const data_flow::AccessNode& org_src,
        const std::pair<types::PrimitiveType, const TensorLayout*>& src_type,
        const ElementInput& needed_input,
        const std::vector<symbolic::Expression>& eltwise_subset,
        std::unordered_map<const data_flow::AccessNode*, data_flow::AccessNode*>& new_node_mapping
    );

    static void create_output(
        builder::StructuredSDFGBuilder& builder,
        structured_control_flow::Block& block,
        const data_flow::AccessNode& org_dst,
        const types::Tensor& dst_type,
        const ElementOutput& provided_output,
        const std::vector<symbolic::Expression>& eltwise_subset
    );

    /**
     * @brief Expand into map with linearized indexing
     *
     * Creates nested maps over each dimension with linearized index computation
     * for accessing the flat input/output arrays.
     *
     * @param builder SDFG builder
     * @param analysis_manager Analysis manager
     * @return True if expansion succeeded
     */
    bool expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    data_flow::PointerAccessType pointer_access_type(int input_idx) const override;

protected:
    /**
     * Models the pure dataflow of the operation as applied to each element of the output as a function of the inputs.
     * Must fit into a single Dataflow block.
     * All inputs will be single elements. The caller must handle potential broadcasts etc.
     *
     * @param builder
     * @param analysis_manager
     * @param block
     * @param input_types access nodes for the current element of each of the input tensors in their respective order
     * @param needed_inputs list of inputs and their targets to connect with which scalar type
     * @param expected_type
     * @return (output producer, output-conn index), nullptr for producer to signal abort
     */
    virtual ElementOutput expand_operation_dataflow(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::Block& block,
        std::vector<ElementInput>& needed_inputs,
        types::PrimitiveType expected_type
    ) = 0;

    data_flow::AccessNode& create_tmp_access_node(
        builder::StructuredSDFGBuilder& builder,
        structured_control_flow::Block& block,
        const std::string& prefix,
        const types::IType& type
    ) const;
};

class BaseElementWiseDataflowTensorNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override;

    struct BaseDeser {
        std::vector<symbolic::Expression> shape;
        QuantizationType quantization;
        DebugInfo debug_info;
    };

    BaseDeser deserialize_base_values(const nlohmann::json& j);
};

template<typename T>
class SimpleElementWiseDataflowTensorNodeSerializer : public BaseElementWiseDataflowTensorNodeSerializer {
public:
    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override {
        // Assertions for required fields
        auto base = deserialize_base_values(j);

        return static_cast<ElementWiseDataflowTensorNode&>(builder.add_library_node<
                                                           T>(parent, base.debug_info, base.shape, base.quantization));
    }
};

} // namespace tensor
} // namespace math
} // namespace sdfg
