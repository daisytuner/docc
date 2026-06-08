#pragma once

#include <cassert>
#include <cstddef>
#include <vector>
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_node.h"
#include "sdfg/serializer/json_serializer.h"

#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace math {
namespace tensor {

/**
 * @brief Specialization of TensorNode for spatial operations such as Convolution and Pooling
 *
 * Both ConvNode and PoolingNode operate on tensors with layout
 * [N, C, D1, …, Dn] and share identical kernel_shape / strides / pads /
 * dilations semantics.  This mixin extracts the common helpers so they
 * are written — and tested — in one place.
 *
 * ## Requirements on `Derived`
 *
 * `Derived` must expose the following **public** const accessors (both
 * ConvNode and PoolingNode already do):
 *
 * | Accessor          | Return type                                      |
 * |-------------------|--------------------------------------------------|
 * | `shape()`         | `const std::vector<symbolic::Expression>&`       |
 * | `kernel_shape()`  | `const std::vector<symbolic::Expression>&`       |
 * | `strides()`       | `const std::vector<symbolic::Expression>&`       |
 * | `pads()`          | `const std::vector<symbolic::Expression>&`       |
 * | `dilations()`     | `const std::vector<symbolic::Expression>&`       |
 */
class SpatialTensorNode : public TensorNode {
protected:
    QuantizationType fixed_quantization_;
    std::vector<symbolic::Expression> shape_; ///< Input shape [N, C, D1, ..., Dn]
    std::vector<symbolic::Expression> kernel_shape_; ///< Pooling window shape [k1, ..., kn]
    std::vector<symbolic::Expression> strides_; ///< Stride along each spatial axis
    std::vector<symbolic::Expression> pads_; ///< Padding (start and end for each axis)
    std::vector<symbolic::Expression> dilations_; ///< Dilation along each spatial axis

public:
    SpatialTensorNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const data_flow::LibraryNodeCode& code,
        const std::vector<std::string>& outputs,
        const std::vector<std::string>& inputs,
        const data_flow::ImplementationType& impl_type,
        QuantizationType quantization,
        const std::vector<symbolic::Expression>& shape,
        const std::vector<symbolic::Expression>& kernel_shape,
        const std::vector<symbolic::Expression>& strides,
        const std::vector<symbolic::Expression>& pads,
        const std::vector<symbolic::Expression>& dilations
    );

    const std::vector<symbolic::Expression>& shape() const { return shape_; }
    const std::vector<symbolic::Expression>& kernel_shape() const { return kernel_shape_; }
    const std::vector<symbolic::Expression>& strides() const { return strides_; }
    const std::vector<symbolic::Expression>& pads() const { return pads_; }
    const std::vector<symbolic::Expression>& dilations() const { return dilations_; }

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

    symbolic::SymbolSet symbols() const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;
    /**
     * @brief Number of spatial dimensions.
     *
     * For a tensor with shape [N, C, D1, …, Dn] this returns n.
     */
    size_t num_spatial_dims() const;

    /**
     * @brief Compute the output size for spatial dimension @p i.
     *
     * Uses the standard convolution / pooling output-size formula:
     * @code
     *   floor((D_i + pad_begin_i + pad_end_i
     *          - dilation_i * (k_i - 1) - 1) / stride_i) + 1
     * @endcode
     *
     * @param i  Spatial dimension index (0-based, relative to the spatial axes).
     */
    symbolic::Expression output_spatial_dim(size_t i) const;

    /**
     * @brief Total number of output spatial elements: prod(output_spatial_dim(i)).
     *
     * Does **not** include the batch (N) or channel (C / C_out) dimensions —
     * those are node-specific.
     */
    symbolic::Expression output_spatial_volume() const;

    /**
     * @brief Total number of kernel / window elements: prod(kernel_shape[i]).
     *
     * For convolutions this is the per-channel kernel volume;
     * for pooling this is the window volume.
     */
    symbolic::Expression kernel_volume() const;

    std::basic_ostream<char>& operator<<(std::basic_ostream<char>& os) const;
};

class SpatialTensorNodeBaseSerializer : public serializer::LibraryNodeSerializer {
public:
    struct BaseDeser {
        std::vector<symbolic::Expression> shape;
        std::vector<symbolic::Expression> kernel_shape;
        std::vector<symbolic::Expression> strides;
        std::vector<symbolic::Expression> pads;
        std::vector<symbolic::Expression> dilations;
        QuantizationType quantization;
        DebugInfo debug_info;
    };

    void fill_base_values(const SpatialTensorNode& node, nlohmann::json& j);

    BaseDeser deserialize_base_values(const nlohmann::json& j);
};

} // namespace tensor
} // namespace math
} // namespace sdfg
