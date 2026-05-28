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

    void validate_shape_matches(
        const std::vector<symbolic::Expression>& required_shape, const TensorLayout& layout, const std::string& name
    ) const;

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
        const std::vector<symbolic::Expression>& eltwise_subset
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

/**
 * @class ElementWiseUnaryNode
 * @brief Base class for elementwise unary tensor operations
 *
 * ElementWiseUnaryNode represents operations that apply a unary function to
 * each element of a tensor independently. The operation is:
 *   Y[i] = f(X[i]) for all i in 0..product(shape)
 *
 * Where indexing is linearized across all dimensions.
 *
 * Derived classes implement specific operations (abs, exp, sqrt, etc.) by
 * providing the expand_operation method that generates the actual computation.
 *
 * ## Input/Output Requirements
 * - Input connector: "X" (scalar or flat pointer to scalar)
 * - Output connector: "Y" (scalar or flat pointer to scalar)
 * - Shape: Multi-dimensional logical shape
 * - Indexing: Linearized (flat) memory layout
 */
class ElementWiseUnaryNode : public TensorNode {
protected:
    QuantizationType quantization_;
    std::vector<symbolic::Expression> shape_; ///< Logical tensor shape

public:
    /**
     * @brief Construct an elementwise unary node
     * @param element_id Unique element identifier
     * @param debug_info Debug information
     * @param vertex Graph vertex
     * @param parent Parent dataflow graph
     * @param code Operation code
     * @param shape Logical tensor shape
     */
    ElementWiseUnaryNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const data_flow::LibraryNodeCode& code,
        const std::vector<symbolic::Expression>& shape,
        QuantizationType quantization = QUANTIZATION_MATCH_INPUTS,
        const data_flow::ImplementationType& impl_type = data_flow::ImplementationType_NONE
    );

    void validate(const Function& function) const override;

    /**
     * @brief Get the tensor shape
     * @return Logical tensor shape
     */
    const std::vector<symbolic::Expression>& shape() const { return shape_; }

    symbolic::SymbolSet symbols() const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

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

    /**
     * @brief Generate the actual operation code
     *
     * Subclasses implement this to generate the specific operation (abs, exp, etc.)
     *
     * @param builder SDFG builder
     * @param analysis_manager Analysis manager
     * @param body Sequence to add the operation to
     * @param input_name Input data name
     * @param output_name Output data name
     * @param input_type Input data type
     * @param output_type Output data type
     * @param subset Data subset for the operation
     * @return True if operation generation succeeded
     */
    virtual bool expand_operation(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::Sequence& body,
        const std::string& input_name,
        const std::string& output_name,
        const types::Tensor& input_type,
        const types::Tensor& output_type,
        const data_flow::Subset& subset
    ) = 0;
};

template<typename T>
class ElementWiseUnaryNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override {
        const ElementWiseUnaryNode& elem_node = static_cast<const ElementWiseUnaryNode&>(library_node);
        nlohmann::json j;

        j["code"] = elem_node.code().value();

        serializer::JSONSerializer serializer;
        j["shape"] = nlohmann::json::array();
        for (auto& dim : elem_node.shape()) {
            j["shape"].push_back(serializer.expression(dim));
        }

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

        auto code = j["code"].get<std::string>();

        std::vector<symbolic::Expression> shape;
        for (const auto& dim : j["shape"]) {
            shape.push_back(symbolic::parse(dim.get<std::string>()));
        }

        // Extract debug info using JSONSerializer
        sdfg::serializer::JSONSerializer serializer;
        DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

        return static_cast<ElementWiseUnaryNode&>(builder.add_library_node<T>(parent, debug_info, shape));
    }
};

/**
 * @class ElementWiseBinaryNode
 * @brief Base class for elementwise binary tensor operations
 *
 * ElementWiseBinaryNode represents operations that apply a binary function to
 * corresponding elements of two tensors independently. The operation is:
 *   Y[i] = f(A[i], B[i]) for all i in 0..product(shape)
 *
 * Where indexing is linearized across all dimensions.
 *
 * Derived classes implement specific operations (add, sub, mul, div, etc.) by
 * providing the expand_operation method that generates the actual computation.
 *
 * ## Input/Output Requirements
 * - Input connectors: "A", "B" (scalars or flat pointers to scalars)
 * - Output connector: "Y" (scalar or flat pointer to scalar)
 * - Shape: Multi-dimensional logical shape
 * - Indexing: Linearized (flat) memory layout
 */
class ElementWiseBinaryNode : public TensorNode {
protected:
    std::vector<symbolic::Expression> shape_; ///< Logical tensor shape

public:
    /**
     * @brief Construct an elementwise binary node
     * @param element_id Unique element identifier
     * @param debug_info Debug information
     * @param vertex Graph vertex
     * @param parent Parent dataflow graph
     * @param code Operation code
     * @param shape Logical tensor shape
     */
    ElementWiseBinaryNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const data_flow::LibraryNodeCode& code,
        const std::vector<symbolic::Expression>& shape
    );

    void validate(const Function& function) const override;

    /**
     * @brief Get the tensor shape
     * @return Logical tensor shape
     */
    const std::vector<symbolic::Expression>& shape() const { return shape_; }

    symbolic::SymbolSet symbols() const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

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

    /**
     * @brief Create an input memlet for a binary operation
     *
     * Handles both regular data containers and constants, as well as scalar
     * vs tensor inputs. This is a common pattern shared by multiple binary
     * elementwise operations (add, sub, mul, div, etc.).
     *
     * @param builder SDFG builder
     * @param input_conn Input connector name on the tasklet (e.g. "_in1", "_in2")
     * @param input_name Input data name
     * @param input_type Input tensor type
     * @param subset Data subset for the operation
     * @param code_block Block to add the memlet to
     * @param tasklet Tasklet to connect the input to
     */
    static void create_input_memlet(
        builder::StructuredSDFGBuilder& builder,
        const std::string& input_conn,
        const std::string& input_name,
        const types::Tensor& input_type,
        const data_flow::Subset& subset,
        structured_control_flow::Block& code_block,
        data_flow::CodeNode& code_node
    );

    /**
     * @brief Generate the actual operation code
     *
     * Subclasses implement this to generate the specific operation (add, sub, etc.)
     *
     * @param builder SDFG builder
     * @param analysis_manager Analysis manager
     * @param body Sequence to add the operation to
     * @param input_name_a First input data name
     * @param input_name_b Second input data name
     * @param output_name Output data name
     * @param input_type_a First input data type
     * @param input_type_b Second input data type
     * @param output_type Output data type
     * @param subset Data subset for the operation
     * @return True if operation generation succeeded
     */
    virtual bool expand_operation(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::Sequence& body,
        const std::string& input_name_a,
        const std::string& input_name_b,
        const std::string& output_name,
        const types::Tensor& input_type_a,
        const types::Tensor& input_type_b,
        const types::Tensor& output_type,
        const data_flow::Subset& subset
    ) = 0;
};

template<typename T>
class ElementWiseBinaryNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override {
        const ElementWiseBinaryNode& elem_node = static_cast<const ElementWiseBinaryNode&>(library_node);
        nlohmann::json j;

        j["code"] = elem_node.code().value();

        serializer::JSONSerializer serializer;
        j["shape"] = nlohmann::json::array();
        for (auto& dim : elem_node.shape()) {
            j["shape"].push_back(serializer.expression(dim));
        }

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

        auto code = j["code"].get<std::string>();

        std::vector<symbolic::Expression> shape;
        for (const auto& dim : j["shape"]) {
            shape.push_back(symbolic::parse(dim.get<std::string>()));
        }

        // Extract debug info using JSONSerializer
        sdfg::serializer::JSONSerializer serializer;
        DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

        return static_cast<ElementWiseBinaryNode&>(builder.add_library_node<T>(parent, debug_info, shape));
    }
};

} // namespace tensor
} // namespace math
} // namespace sdfg
