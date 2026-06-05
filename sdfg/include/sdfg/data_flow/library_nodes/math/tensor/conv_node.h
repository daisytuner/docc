/**
 * @file conv_node.h
 * @brief Convolution operation node compatible with ONNX Conv operator
 *
 * This file defines the ConvNode class which implements a tensor convolution
 * operation following the ONNX Conv operator specification. The node is expanded
 * using the im2col transformation into a GEMM operation.
 *
 * ## ONNX Conv Operator Compatibility
 *
 * The ConvNode implements the ONNX Conv operator with the following parameters:
 * - Input tensor X: [N, C_in, D1, D2, ..., Dn] for n-dimensional convolution
 * - Weight tensor W: [C_out, C_in/group, k1, k2, ..., kn]
 * - Optional bias tensor B: [C_out]
 * - Output tensor Y: [N, C_out, D1_out, D2_out, ..., Dn_out]
 *
 * Supported attributes:
 * - kernel_shape: Shape of the convolution kernel
 * - strides: Stride along each spatial axis
 * - pads: Padding for the beginning and ending along each spatial axis
 * - dilations: Dilation along each spatial axis
 * - group: Number of groups for grouped convolutions
 *
 * ## Expansion via im2col
 *
 * The convolution is expanded into nested maps using a direct convolution approach:
 * 1. Create outer maps for parallel iteration over batch and output dimensions
 * 2. Create inner loops for sequential accumulation over input channels and kernel
 * 3. Compute convolution using FMA (fused multiply-add) operations
 * 4. Add bias (if present)
 * 5. Write results to output tensor
 *
 * The expansion supports n-dimensional convolutions (1D, 2D, 3D, etc.) with
 * configurable strides and padding for each spatial dimension.
 */

#pragma once

#include "sdfg/data_flow/library_nodes/math/tensor/tensor_node.h"

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/data_flow/library_nodes/math/blas/blas_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/spatial_tensor_node.h"
#include "sdfg/serializer/json_serializer.h"

namespace sdfg {
namespace math {
namespace tensor {

inline data_flow::LibraryNodeCode LibraryNodeType_Conv("ml::Conv");

/**
 * @class ConvNode
 * @brief Convolution operation following ONNX Conv operator specification
 *
 * ConvNode represents a convolution operation that is compatible with the
 * ONNX Conv operator. The operation is expanded using nested maps for
 * n-dimensional convolutions (1D, 2D, 3D, etc.).
 *
 * ## Input/Output Requirements
 * - Input connector "X": Input tensor [N, C_in, D1, ..., Dn]
 * - Input connector "W": Weight tensor [C_out, C_in/group, k1, ..., kn]
 * - Input connector "B" (optional): Bias tensor [C_out]
 * - Output connector "Y": Output tensor [N, C_out, D1_out, ..., Dn_out]
 *
 * ## Expansion Support
 * - ✅ 1D, 2D, 3D, and higher-dimensional convolutions
 * - ✅ Configurable strides for each spatial dimension
 * - ✅ Configurable padding (start and end) for each spatial dimension
 * - ✅ Optional bias addition
 * - ⚠️ Grouped convolutions and dilations not yet expanded (returns false)
 *
 * ## Example
 *
 * Creating a 2D convolution:
 * @code
 * std::vector<symbolic::Expression> kernel_shape = {symbolic::integer(3), symbolic::integer(3)};
 * std::vector<symbolic::Expression> strides = {symbolic::integer(1), symbolic::integer(1)};
 * std::vector<symbolic::Expression> pads = {symbolic::integer(1), symbolic::integer(1),
 *                                            symbolic::integer(1), symbolic::integer(1)};
 * std::vector<symbolic::Expression> dilations = {symbolic::integer(1), symbolic::integer(1)};
 * auto group = symbolic::integer(1);
 *
 * auto& conv_node = builder.add_library_node<math::tensor::ConvNode>(
 *     block, debug_info, kernel_shape, strides, pads, dilations, group
 * );
 * @endcode
 */
class ConvNode : public SpatialTensorNode {
protected:
    symbolic::Expression output_channels_; ///< Number of output channels (C_out)
    symbolic::Expression group_; ///< Number of groups for grouped convolution
    bool with_bias_;

public:
    /**
     * @brief Construct a convolution node
     * @param element_id Unique element identifier
     * @param debug_info Debug information
     * @param vertex Graph vertex
     * @param parent Parent dataflow graph
     * @param shape Input tensor shape [N, C_in, D1, ..., Dn]
     * @param kernel_shape Shape of the convolution kernel
     * @param strides Stride along each spatial axis (defaults to 1 for each axis)
     * @param pads Padding for start and end of each axis (defaults to 0)
     * @param dilations Dilation along each spatial axis (defaults to 1)
     * @param output_channels Number of output channels
     * @param group Number of groups for grouped convolution (defaults to 1)
     */
    ConvNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const std::vector<symbolic::Expression>& shape,
        const std::vector<symbolic::Expression>& kernel_shape,
        const std::vector<symbolic::Expression>& strides,
        const std::vector<symbolic::Expression>& pads,
        const std::vector<symbolic::Expression>& dilations,
        symbolic::Expression output_channels,
        symbolic::Expression group,
        bool with_bias = false,
        QuantizationType quantization = QUANTIZATION_MATCH_INPUTS,
        const data_flow::ImplementationType& impl_type = data_flow::ImplementationType_NONE
    );

    /**
     * @brief Get the output channels
     * @return Number of output channels
     */
    symbolic::Expression output_channels() const { return output_channels_; }

    /**
     * @brief Get the group count
     * @return Number of groups for grouped convolution
     */
    symbolic::Expression group() const { return group_; }

    void validate(const Function& function) const override;

    static blas::BLAS_Precision get_blas_precision(types::Scalar base_type);

    symbolic::MultiExpression get_out_shape();

    bool has_bias() const;

    /**
     * @brief Expand convolution into nested maps for n-dimensional convolution
     *
     * Expands the convolution operation by:
     * 1. Creating outer maps for parallel iteration over batch and output dimensions
     * 2. Creating inner for loops for sequential accumulation over input channels and kernel
     * 3. Computing convolution using FMA (fused multiply-add) tasklets
     * 4. Adding bias if present
     * 5. Writing results to output tensor
     *
     * Supports n-dimensional convolutions (1D, 2D, 3D, and higher) with
     * configurable strides and padding for each spatial dimension.
     *
     * @param builder SDFG builder
     * @param analysis_manager Analysis manager
     * @return True if expansion succeeded, false for unsupported configurations
     */
    bool expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    symbolic::SymbolSet symbols() const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    bool supports_integer_types() const override { return false; }

    std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;

    std::string toStr() const override;

    /**
     * @brief Total number of output elements: N * C_out * prod(output_spatial_dim(i))
     */
    symbolic::Expression num_output_elements() const;

    /**
     * @brief Number of multiply-accumulate iterations per output element:
     *        (C_in / group) * prod(kernel_shape[i])
     */
    symbolic::Expression kernel_iteration_count() const;

    symbolic::Expression flop() const override;

    data_flow::PointerAccessType pointer_access_type(int input_idx) const override;

    struct ConvExpandPrerequisits {
        const data_flow::Memlet* iedge_X;
        const data_flow::Memlet* iedge_W;
        const data_flow::Memlet* iedge_B;
        const data_flow::Memlet* iedge_Y;
        const data_flow::AccessNode* access_X;
        const data_flow::AccessNode* access_W;
        const data_flow::AccessNode* access_B;
        const data_flow::AccessNode* access_Y;
        bool has_bias;
        structured_control_flow::Block* block;
        structured_control_flow::Sequence* block_parent;
        size_t block_index;
    };

    bool check_expandable(
        data_flow::DataFlowGraph& dfg, analysis::AnalysisManager& analysis_manager, ConvExpandPrerequisits& boundary
    ) const;
};

/**
 * @class ConvNodeSerializer
 * @brief Serializer for ConvNode
 */
class ConvNodeSerializer : public SpatialTensorNodeBaseSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override;

    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override;
};

} // namespace tensor
} // namespace math
} // namespace sdfg
