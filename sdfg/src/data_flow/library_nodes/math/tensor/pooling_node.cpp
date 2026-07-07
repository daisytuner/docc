#include "sdfg/data_flow/library_nodes/math/tensor/pooling_node.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/spatial_tensor_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_node.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace math {
namespace tensor {

PoolingNode::PoolingNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    PoolingMode mode,
    const std::vector<symbolic::Expression>& shape,
    const std::vector<symbolic::Expression>& kernel_shape,
    const std::vector<symbolic::Expression>& strides,
    const std::vector<symbolic::Expression>& pads,
    const std::vector<symbolic::Expression>& dilations,
    QuantizationType quantization,
    const data_flow::ImplementationType& impl_type
)
    : SpatialTensorNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_Pooling,
          {},
          {"Y", "X"},
          impl_type,
          quantization,
          shape,
          kernel_shape,
          strides,
          pads,
          dilations
      ),
      mode_(mode) {}

void PoolingNode::validate(const Function& function) const {
    TensorNode::validate(function);

    if (kernel_shape_.empty()) {
        throw InvalidSDFGException("PoolingNode kernel_shape cannot be empty");
    }

    size_t spatial_dims = kernel_shape_.size();

    if (!strides_.empty() && strides_.size() != spatial_dims) {
        throw InvalidSDFGException("PoolingNode strides must match kernel spatial dimensions");
    }

    if (!pads_.empty() && pads_.size() != 2 * spatial_dims) {
        throw InvalidSDFGException("PoolingNode pads must have 2 * spatial dimensions");
    }

    if (!dilations_.empty() && dilations_.size() != spatial_dims) {
        throw InvalidSDFGException("PoolingNode dilations must match kernel spatial dimensions");
    }
}

passes::LibNodeExpander::ExpandOutcome PoolingNode::
    expand(passes::LibNodeExpander::ExpandContext& context, structured_control_flow::Block& block) {
    auto& dataflow = this->get_parent();

    auto primitive_type = this->primitive_type(dataflow);
    types::Scalar scalar_type(primitive_type);

    auto x_edge = dataflow.in_edge_for_connector(*this, "X");
    if (!x_edge) {
        return context.unable();
    }

    auto y_edge = dataflow.in_edge_for_connector(*this, "Y");
    if (!y_edge) {
        return context.unable();
    }

    size_t spatial_dims = kernel_shape_.size();
    if (spatial_dims == 0) {
        return context.unable();
    }

    // Get strides (default to 1)
    std::vector<symbolic::Expression> strides_vec;
    for (size_t i = 0; i < spatial_dims; ++i) {
        if (i < strides_.size()) {
            strides_vec.push_back(strides_[i]);
        } else {
            strides_vec.push_back(symbolic::one());
        }
    }

    // Get padding (default to 0)
    std::vector<symbolic::Expression> pads_begin_vec, pads_end_vec;
    for (size_t i = 0; i < spatial_dims; ++i) {
        if (i < pads_.size()) {
            pads_begin_vec.push_back(pads_[i]);
        } else {
            pads_begin_vec.push_back(symbolic::zero());
        }
        if (spatial_dims + i < pads_.size()) {
            pads_end_vec.push_back(pads_[spatial_dims + i]);
        } else {
            pads_end_vec.push_back(symbolic::zero());
        }
    }

    // Get dilations (default to 1)
    std::vector<symbolic::Expression> dilations_vec;
    for (size_t i = 0; i < spatial_dims; ++i) {
        if (i < dilations_.size()) {
            dilations_vec.push_back(dilations_[i]);
        } else {
            dilations_vec.push_back(symbolic::one());
        }
    }

    // Input shape: [N, C, D0, D1, ..., Dn]
    symbolic::Expression N = shape_[0];
    symbolic::Expression C = shape_[1];
    std::vector<symbolic::Expression> input_spatial_dims;
    for (size_t i = 0; i < spatial_dims; ++i) {
        input_spatial_dims.push_back(shape_[2 + i]);
    }

    // Output spatial dimensions
    std::vector<symbolic::Expression> output_spatial_dims;
    for (size_t i = 0; i < spatial_dims; ++i) {
        auto d_in = input_spatial_dims[i];
        auto pad = symbolic::add(pads_begin_vec[i], pads_end_vec[i]);
        auto dk = symbolic::mul(dilations_vec[i], symbolic::sub(kernel_shape_[i], symbolic::one()));
        auto num = symbolic::sub(symbolic::add(d_in, pad), symbolic::add(dk, symbolic::one()));
        auto d_out = symbolic::add(symbolic::div(num, strides_vec[i]), symbolic::one());
        output_spatial_dims.push_back(d_out);
    }

    using Use = passes::LibNodeExpander::InputUse;
    auto standalone = context.replacement_requires_access_nodes({Use::IndirectWrite, Use::IndirectRead});

    if (!standalone) {
        return context.unable();
    }

    auto& new_sequence = standalone->replace_with_sequence();
    auto& builder = standalone->builder();

    structured_control_flow::Sequence* current_scope = &new_sequence;
    std::vector<symbolic::Expression> output_indices;
    std::vector<symbolic::Expression> output_spatial_vars;

    // Map over batch
    std::string n_str = builder.find_new_name("n");
    builder.add_container(n_str, types::Scalar(types::PrimitiveType::UInt64));
    auto n_var = symbolic::symbol(n_str);
    auto& map_n = builder.add_map(
        *current_scope,
        n_var,
        symbolic::Lt(n_var, N),
        symbolic::zero(),
        symbolic::add(n_var, symbolic::one()),
        structured_control_flow::ScheduleType_Sequential::create(),
        {},
        block.debug_info()
    );
    current_scope = &map_n.root();
    output_indices.push_back(n_var);

    // Map over channel
    std::string c_str = builder.find_new_name("c");
    builder.add_container(c_str, types::Scalar(types::PrimitiveType::UInt64));
    auto c_var = symbolic::symbol(c_str);
    auto& map_c = builder.add_map(
        *current_scope,
        c_var,
        symbolic::Lt(c_var, C),
        symbolic::zero(),
        symbolic::add(c_var, symbolic::one()),
        structured_control_flow::ScheduleType_Sequential::create(),
        {},
        block.debug_info()
    );
    current_scope = &map_c.root();
    output_indices.push_back(c_var);

    // Map over each output spatial dimension
    for (size_t i = 0; i < spatial_dims; ++i) {
        std::string od_str = builder.find_new_name("od" + std::to_string(i));
        builder.add_container(od_str, types::Scalar(types::PrimitiveType::UInt64));
        auto od_var = symbolic::symbol(od_str);
        auto& map_od = builder.add_map(
            *current_scope,
            od_var,
            symbolic::Lt(od_var, output_spatial_dims[i]),
            symbolic::zero(),
            symbolic::add(od_var, symbolic::one()),
            structured_control_flow::ScheduleType_Sequential::create(),
            {},
            block.debug_info()
        );
        current_scope = &map_od.root();
        output_indices.push_back(od_var);
        output_spatial_vars.push_back(od_var);
    }

    // Create accumulator
    std::string accum_var = builder.find_new_name("_pool_accum");
    builder.add_container(accum_var, scalar_type);

    // Initialize accumulator
    std::string init_value;
    if (mode_ == PoolingMode::Max) {
        // Use -INFINITY for float, type-min for integers
        if (types::is_integer(primitive_type)) {
            switch (primitive_type) {
                case types::PrimitiveType::Int8:
                    init_value = "INT8_MIN";
                    break;
                case types::PrimitiveType::Int16:
                    init_value = "INT16_MIN";
                    break;
                case types::PrimitiveType::Int32:
                    init_value = "INT32_MIN";
                    break;
                case types::PrimitiveType::Int64:
                    init_value = "INT64_MIN";
                    break;
                default:
                    init_value = "0";
                    break;
            }
        } else {
            init_value = "-INFINITY";
        }
    } else {
        // Sum / Avg: init to 0
        init_value = types::is_integer(primitive_type) ? "0" : "0.0";
    }

    auto& init_block = builder.add_block(*current_scope, {}, block.debug_info());
    auto& accum_init = builder.add_access(init_block, accum_var, block.debug_info());
    auto& zero_const = builder.add_constant(init_block, init_value, scalar_type, block.debug_info());
    auto& init_tasklet = builder.add_tasklet(init_block, data_flow::assign, "_out", {"_in"}, block.debug_info());
    builder.add_computational_memlet(init_block, zero_const, init_tasklet, "_in", {}, scalar_type, block.debug_info());
    builder.add_computational_memlet(init_block, init_tasklet, "_out", accum_init, {}, scalar_type, block.debug_info());

    // For loops over kernel spatial dimensions
    auto* loop_scope = current_scope;
    std::vector<symbolic::Expression> kernel_vars;
    for (size_t i = 0; i < spatial_dims; ++i) {
        std::string k_str = builder.find_new_name("k" + std::to_string(i));
        builder.add_container(k_str, types::Scalar(types::PrimitiveType::UInt64));
        auto k_var = symbolic::symbol(k_str);
        auto& for_k = builder.add_for(
            *loop_scope,
            k_var,
            symbolic::Lt(k_var, kernel_shape_[i]),
            symbolic::zero(),
            symbolic::add(k_var, symbolic::one()),
            {},
            block.debug_info()
        );
        loop_scope = &for_k.root();
        kernel_vars.push_back(k_var);
    }

    // Compute input spatial indices
    std::vector<symbolic::Expression> input_spatial_indices;
    for (size_t i = 0; i < spatial_dims; ++i) {
        auto k_dilated = symbolic::mul(kernel_vars[i], dilations_vec[i]);
        auto input_idx = symbolic::
            add(symbolic::sub(symbolic::mul(output_spatial_vars[i], strides_vec[i]), pads_begin_vec[i]), k_dilated);
        input_spatial_indices.push_back(input_idx);
    }

    // Add branching if padding is non-zero
    bool has_padding = false;
    for (auto padding : this->pads_) {
        if (!symbolic::eq(padding, symbolic::zero())) {
            has_padding = true;
            break;
        }
    }
    if (has_padding) {
        symbolic::Condition comp_condition = symbolic::__true__();
        for (size_t i = 0; i < spatial_dims; ++i) {
            comp_condition = symbolic::
                And(comp_condition,
                    symbolic::
                        And(symbolic::Lt(input_spatial_indices[i], input_spatial_dims[i]),
                            symbolic::Ge(input_spatial_indices[i], symbolic::zero())));
        }
        auto& branch = builder.add_if_else(*loop_scope, {}, block.debug_info());
        auto& comp_case = builder.add_case(branch, comp_condition, block.debug_info());
        loop_scope = &comp_case;
    }

    // Build X indices: [n, c, input_spatial...]
    std::vector<symbolic::Expression> x_indices_vec = {n_var, c_var};
    x_indices_vec.insert(x_indices_vec.end(), input_spatial_indices.begin(), input_spatial_indices.end());
    data_flow::Subset x_subset(x_indices_vec);

    // Computation block: accumulate
    auto& comp_block = builder.add_block(*loop_scope, {}, block.debug_info());
    auto& x_access = standalone->add_indirect_read_access(comp_block, 1);
    auto& accum_read = builder.add_access(comp_block, accum_var, block.debug_info());
    auto& accum_write = builder.add_access(comp_block, accum_var, block.debug_info());

    if (mode_ == PoolingMode::Max) {
        bool is_int = types::is_integer(primitive_type);
        if (is_int) {
            auto tasklet_code = TensorNode::get_integer_minmax_tasklet(primitive_type, true);
            auto& tasklet = builder.add_tasklet(comp_block, tasklet_code, "_out", {"_in1", "_in2"}, block.debug_info());
            builder.add_computational_memlet(
                comp_block, x_access, tasklet, "_in1", x_subset, x_edge->base_type(), block.debug_info()
            );
            builder
                .add_computational_memlet(comp_block, accum_read, tasklet, "_in2", {}, scalar_type, block.debug_info());
            builder
                .add_computational_memlet(comp_block, tasklet, "_out", accum_write, {}, scalar_type, block.debug_info());
        } else {
            auto& libnode = builder.add_library_node<
                math::cmath::CMathNode>(comp_block, block.debug_info(), cmath::CMathFunction::fmax, primitive_type);
            builder.add_computational_memlet(
                comp_block, x_access, libnode, "_in1", x_subset, x_edge->base_type(), block.debug_info()
            );
            builder
                .add_computational_memlet(comp_block, accum_read, libnode, "_in2", {}, scalar_type, block.debug_info());
            builder
                .add_computational_memlet(comp_block, libnode, "_out", accum_write, {}, scalar_type, block.debug_info());
        }
    } else {
        // Sum or Avg: accumulate with addition
        bool is_int = types::is_integer(primitive_type);
        data_flow::TaskletCode opcode = is_int ? data_flow::TaskletCode::int_add : data_flow::TaskletCode::fp_add;
        auto& tasklet = builder.add_tasklet(comp_block, opcode, "_out", {"_in1", "_in2"}, block.debug_info());
        builder.add_computational_memlet(
            comp_block, x_access, tasklet, "_in1", x_subset, x_edge->base_type(), block.debug_info()
        );
        builder.add_computational_memlet(comp_block, accum_read, tasklet, "_in2", {}, scalar_type, block.debug_info());
        builder.add_computational_memlet(comp_block, tasklet, "_out", accum_write, {}, scalar_type, block.debug_info());
    }

    // After kernel loops: write result to output
    data_flow::Subset y_subset(output_indices);

    auto& output_block = builder.add_block(*current_scope, {}, block.debug_info());
    auto& accum_final = builder.add_access(output_block, accum_var, block.debug_info());
    auto& y_access = standalone->add_indirect_write_access(output_block, 0);

    if (mode_ == PoolingMode::Avg) {
        // Divide by window size: product of kernel_shape dimensions
        // Create a temporary for the divisor
        std::string divisor_var = builder.find_new_name("_pool_divisor");
        builder.add_container(divisor_var, scalar_type);

        // Compute window size as product of kernel dimensions
        symbolic::Expression window_size = kernel_shape_[0];
        for (size_t i = 1; i < spatial_dims; ++i) {
            window_size = symbolic::mul(window_size, kernel_shape_[i]);
        }

        auto& divisor_const =
            builder.add_constant(output_block, window_size->__str__(), scalar_type, block.debug_info());
        auto& divisor_access = builder.add_access(output_block, divisor_var, block.debug_info());
        auto& divisor_assign =
            builder.add_tasklet(output_block, data_flow::assign, "_out", {"_in"}, block.debug_info());
        builder.add_computational_memlet(
            output_block, divisor_const, divisor_assign, "_in", {}, scalar_type, block.debug_info()
        );
        builder.add_computational_memlet(
            output_block, divisor_assign, "_out", divisor_access, {}, scalar_type, block.debug_info()
        );

        bool is_int = types::is_integer(primitive_type);
        data_flow::TaskletCode div_opcode = is_int ? data_flow::TaskletCode::int_sdiv : data_flow::TaskletCode::fp_div;
        auto& div_tasklet = builder.add_tasklet(output_block, div_opcode, "_out", {"_in1", "_in2"}, block.debug_info());
        builder
            .add_computational_memlet(output_block, accum_final, div_tasklet, "_in1", {}, scalar_type, block.debug_info());
        builder.add_computational_memlet(
            output_block, divisor_access, div_tasklet, "_in2", {}, scalar_type, block.debug_info()
        );
        builder.add_computational_memlet(
            output_block, div_tasklet, "_out", y_access, y_subset, y_edge->base_type(), y_edge->debug_info()
        );
    } else {
        // Max or Sum: just assign
        auto& assign_tasklet =
            builder.add_tasklet(output_block, data_flow::assign, "_out", {"_in"}, block.debug_info());
        builder.add_computational_memlet(
            output_block, accum_final, assign_tasklet, "_in", {}, scalar_type, block.debug_info()
        );
        builder.add_computational_memlet(
            output_block, assign_tasklet, "_out", y_access, y_subset, y_edge->base_type(), y_edge->debug_info()
        );
    }

    return standalone->successfully_expanded();
}

std::unique_ptr<data_flow::DataFlowNode> PoolingNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(new PoolingNode(
        element_id,
        this->debug_info(),
        vertex,
        parent,
        mode_,
        shape_,
        kernel_shape_,
        strides_,
        pads_,
        dilations_,
        fixed_quantization_,
        implementation_type_
    ));
}

std::string PoolingNode::mode_to_string(PoolingMode mode) {
    switch (mode) {
        case PoolingMode::Max:
            return "max";
        case PoolingMode::Sum:
            return "sum";
        case PoolingMode::Avg:
            return "avg";
    }
    return "unknown";
}

PoolingMode PoolingNode::string_to_mode(const std::string& str) {
    if (str == "max") return PoolingMode::Max;
    if (str == "sum") return PoolingMode::Sum;
    if (str == "avg") return PoolingMode::Avg;
    throw InvalidSDFGException("Unknown pooling mode: " + str);
}

symbolic::Expression PoolingNode::flop() const {
    // Total output elements: N * C * prod(output_spatial_dim(i))
    auto output_elems = symbolic::mul(symbolic::mul(shape_[0], shape_[1]), output_spatial_volume());

    // Each output element reduces a full kernel window.
    auto kv = kernel_volume();

    switch (mode_) {
        case PoolingMode::Max:
            // max pooling: (kv - 1) comparisons per output element
            return symbolic::mul(output_elems, symbolic::sub(kv, symbolic::one()));
        case PoolingMode::Sum:
            // sum pooling: (kv - 1) additions per output element
            return symbolic::mul(output_elems, symbolic::sub(kv, symbolic::one()));
        case PoolingMode::Avg:
            // avg pooling: (kv - 1) additions + 1 division per output element
            return symbolic::mul(output_elems, kv);
        default:
            return symbolic::symbol("UnknownFlops_Pool_n" + std::to_string(element_id_));
    }
}

data_flow::PointerAccessType PoolingNode::pointer_access_type(int input_idx) const {
    if (input_idx == 0) {
        return data_flow::PointerAccessMeta::create_full_write_only(symbolic::__nullptr__(), true);
    } else if (input_idx == 1) {
        return data_flow::PointerAccessMeta::create_read_only(symbolic::__nullptr__(), true);
    } else {
        return TensorNode::pointer_access_type(input_idx);
    }
}

std::string PoolingNode::toStr() const {
    std::stringstream ss;
    ss << "Pooling(mode=" << mode_to_string(mode_) << ", ";
    SpatialTensorNode::operator<<(ss);
    ss << ")";
    return ss.str();
}

nlohmann::json PoolingNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const PoolingNode& node = static_cast<const PoolingNode&>(library_node);
    nlohmann::json j;

    j["mode"] = PoolingNode::mode_to_string(node.mode());

    fill_base_values(node, j);

    return j;
}

data_flow::LibraryNode& PoolingNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    assert(j.contains("mode"));

    auto base = deserialize_base_values(j);

    auto mode = PoolingNode::string_to_mode(j["mode"].get<std::string>());

    return builder.add_library_node<PoolingNode>(
        parent,
        base.debug_info,
        mode,
        base.shape,
        base.kernel_shape,
        base.strides,
        base.pads,
        base.dilations,
        base.quantization
    );
}

} // namespace tensor
} // namespace math
} // namespace sdfg
