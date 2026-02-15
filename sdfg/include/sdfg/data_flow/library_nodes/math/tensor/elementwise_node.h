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

namespace sdfg {
namespace math {
namespace tensor {

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
