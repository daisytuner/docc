#include "sdfg/data_flow/library_nodes/math/tensor/pooling_node.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"
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
    const std::vector<symbolic::Expression>& dilations
)
    : TensorNode(
          element_id, debug_info, vertex, parent, LibraryNodeType_Pooling, {"Y"}, {"X"}, data_flow::ImplementationType_NONE
      ),
      mode_(mode), shape_(shape), kernel_shape_(kernel_shape), strides_(strides), pads_(pads), dilations_(dilations) {}

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

bool PoolingNode::expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();

    auto& dataflow = this->get_parent();
    auto& block = static_cast<structured_control_flow::Block&>(*dataflow.get_parent());
    auto& parent = static_cast<structured_control_flow::Sequence&>(*scope_analysis.parent_scope(&block));
    int index = parent.index(block);
    auto& transition = parent.at(index).second;

    auto primitive_type = this->primitive_type(dataflow);
    types::Scalar scalar_type(primitive_type);

    auto in_edges = dataflow.in_edges(*this);
    data_flow::Memlet* x_edge = nullptr;
    auto in_edges_it = in_edges.begin();
    while (in_edges_it != in_edges.end()) {
        auto& edge = *in_edges_it;
        if (edge.dst_conn() == "X") {
            x_edge = &edge;
        }
        ++in_edges_it;
    }
    if (!x_edge) {
        return false;
    }

    auto& y_edge = *dataflow.out_edges(*this).begin();

    auto* x_node = static_cast<data_flow::AccessNode*>(&x_edge->src());
    auto* y_node = static_cast<data_flow::AccessNode*>(&y_edge.dst());

    if (!x_node || dataflow.in_degree(*x_node) != 0 || !y_node || dataflow.out_degree(*y_node) != 0) {
        return false;
    }

    size_t spatial_dims = kernel_shape_.size();
    if (spatial_dims == 0) {
        return false;
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

    auto& X_var = x_node->data();
    auto& Y_var = y_node->data();

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

    auto& new_sequence = builder.add_sequence_before(parent, block, transition.assignments(), block.debug_info());

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
    auto& x_access = builder.add_access(comp_block, X_var, x_node->debug_info());
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
    auto& y_access = builder.add_access(output_block, Y_var, y_node->debug_info());

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
            output_block, div_tasklet, "_out", y_access, y_subset, y_edge.base_type(), y_edge.debug_info()
        );
    } else {
        // Max or Sum: just assign
        auto& assign_tasklet =
            builder.add_tasklet(output_block, data_flow::assign, "_out", {"_in"}, block.debug_info());
        builder.add_computational_memlet(
            output_block, accum_final, assign_tasklet, "_in", {}, scalar_type, block.debug_info()
        );
        builder.add_computational_memlet(
            output_block, assign_tasklet, "_out", y_access, y_subset, y_edge.base_type(), y_edge.debug_info()
        );
    }

    // Clean up original block
    builder.remove_memlet(block, *x_edge);
    builder.remove_memlet(block, y_edge);
    builder.remove_node(block, *x_node);
    builder.remove_node(block, *y_node);
    builder.remove_node(block, *this);
    builder.remove_child(parent, index + 1);

    return true;
}

symbolic::SymbolSet PoolingNode::symbols() const {
    symbolic::SymbolSet syms;
    for (auto& expr : shape_) {
        for (auto& atom : symbolic::atoms(expr)) {
            syms.insert(atom);
        }
    }
    for (auto& expr : kernel_shape_) {
        for (auto& atom : symbolic::atoms(expr)) {
            syms.insert(atom);
        }
    }
    for (auto& expr : strides_) {
        for (auto& atom : symbolic::atoms(expr)) {
            syms.insert(atom);
        }
    }
    for (auto& expr : pads_) {
        for (auto& atom : symbolic::atoms(expr)) {
            syms.insert(atom);
        }
    }
    for (auto& expr : dilations_) {
        for (auto& atom : symbolic::atoms(expr)) {
            syms.insert(atom);
        }
    }
    return syms;
}

void PoolingNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    for (auto& expr : shape_) {
        expr = symbolic::subs(expr, old_expression, new_expression);
    }
    for (auto& expr : kernel_shape_) {
        expr = symbolic::subs(expr, old_expression, new_expression);
    }
    for (auto& expr : strides_) {
        expr = symbolic::subs(expr, old_expression, new_expression);
    }
    for (auto& expr : pads_) {
        expr = symbolic::subs(expr, old_expression, new_expression);
    }
    for (auto& expr : dilations_) {
        expr = symbolic::subs(expr, old_expression, new_expression);
    }
}

std::unique_ptr<data_flow::DataFlowNode> PoolingNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(new PoolingNode(
        element_id, this->debug_info(), vertex, parent, mode_, shape_, kernel_shape_, strides_, pads_, dilations_
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

std::string PoolingNode::toStr() const {
    std::string result = "Pooling(mode=" + mode_to_string(mode_) + ", shape=[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        if (i > 0) result += ", ";
        result += shape_[i]->__str__();
    }
    result += "], kernel_shape=[";
    for (size_t i = 0; i < kernel_shape_.size(); ++i) {
        if (i > 0) result += ", ";
        result += kernel_shape_[i]->__str__();
    }
    result += "], strides=[";
    for (size_t i = 0; i < strides_.size(); ++i) {
        if (i > 0) result += ", ";
        result += strides_[i]->__str__();
    }
    result += "])";
    return result;
}

nlohmann::json PoolingNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const PoolingNode& node = static_cast<const PoolingNode&>(library_node);
    nlohmann::json j;

    j["code"] = node.code().value();
    j["mode"] = PoolingNode::mode_to_string(node.mode());

    serializer::JSONSerializer serializer;

    j["shape"] = nlohmann::json::array();
    for (auto& dim : node.shape()) {
        j["shape"].push_back(serializer.expression(dim));
    }

    j["kernel_shape"] = nlohmann::json::array();
    for (auto& dim : node.kernel_shape()) {
        j["kernel_shape"].push_back(serializer.expression(dim));
    }

    j["strides"] = nlohmann::json::array();
    for (auto& stride : node.strides()) {
        j["strides"].push_back(serializer.expression(stride));
    }

    j["pads"] = nlohmann::json::array();
    for (auto& pad : node.pads()) {
        j["pads"].push_back(serializer.expression(pad));
    }

    j["dilations"] = nlohmann::json::array();
    for (auto& dilation : node.dilations()) {
        j["dilations"].push_back(serializer.expression(dilation));
    }

    return j;
}

data_flow::LibraryNode& PoolingNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    assert(j.contains("element_id"));
    assert(j.contains("code"));
    assert(j.contains("debug_info"));
    assert(j.contains("mode"));
    assert(j.contains("kernel_shape"));

    auto mode = PoolingNode::string_to_mode(j["mode"].get<std::string>());

    std::vector<symbolic::Expression> shape;
    if (j.contains("shape")) {
        for (const auto& dim : j["shape"]) {
            shape.push_back(symbolic::parse(dim.get<std::string>()));
        }
    }

    std::vector<symbolic::Expression> kernel_shape;
    for (const auto& dim : j["kernel_shape"]) {
        kernel_shape.push_back(symbolic::parse(dim.get<std::string>()));
    }

    std::vector<symbolic::Expression> strides;
    if (j.contains("strides")) {
        for (const auto& stride : j["strides"]) {
            strides.push_back(symbolic::parse(stride.get<std::string>()));
        }
    }

    std::vector<symbolic::Expression> pads;
    if (j.contains("pads")) {
        for (const auto& pad : j["pads"]) {
            pads.push_back(symbolic::parse(pad.get<std::string>()));
        }
    }

    std::vector<symbolic::Expression> dilations;
    if (j.contains("dilations")) {
        for (const auto& dilation : j["dilations"]) {
            dilations.push_back(symbolic::parse(dilation.get<std::string>()));
        }
    }

    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    return builder
        .add_library_node<PoolingNode>(parent, debug_info, mode, shape, kernel_shape, strides, pads, dilations);
}

} // namespace tensor
} // namespace math
} // namespace sdfg
