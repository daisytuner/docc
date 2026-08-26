#include "sdfg/data_flow/library_nodes/math/tensor/layernorm_node.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/data_flow/data_flow_node.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_layout.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_node.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/data_flow/pointer_metadata.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/element.h"
#include "sdfg/exceptions.h"
#include "sdfg/graph/graph.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/reduce.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/tensor.h"
#include "symengine/add.h"
#include "symengine/mul.h"

namespace sdfg {
namespace math {
namespace tensor {

static std::string shape_to_str(const symbolic::MultiExpression& shape) {
    std::stringstream stream;
    stream << "[";
    long long dims = shape.size();
    for (long long i = 0; i < dims; i++) {
        if (i > 0) {
            stream << ", ";
        }
        stream << shape[i]->__str__();
    }
    stream << "]";
    return stream.str();
}

void LayerNormNode::validate_equal_shapes(
    const std::string& msg, const symbolic::MultiExpression& shape1, const symbolic::MultiExpression& shape2
) const {
    long long size1 = shape1.size();
    long long size2 = shape2.size();
    if (size1 == 0 && size2 == 1 && symbolic::eq(shape2[0], symbolic::one())) {
        return;
    }
    if (size1 == 1 && size2 == 0 && symbolic::eq(shape1[0], symbolic::one())) {
        return;
    }
    if (size1 != size2) {
        throw InvalidSDFGException(
            "LayerNormNode: " + msg + " shapes mismatch: " + shape_to_str(shape1) + " != " + shape_to_str(shape2)
        );
    }
    for (long long i = 0; i < size1; i++) {
        if (!symbolic::eq(shape1[i], shape2[i])) {
            throw InvalidSDFGException(
                "LayerNormNode: " + msg + " shapes mismatch: " + shape_to_str(shape1) + " != " + shape_to_str(shape2)
            );
        }
    }
}

LayerNormNode::LayerNormNode(
    size_t element_id,
    const DebugInfo& debug_info,
    graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const symbolic::MultiExpression& normalized_shape,
    const TensorLayout& y_layout,
    const TensorLayout& mean_layout,
    const TensorLayout& rstd_layout,
    const TensorLayout& x_layout,
    QuantizationType quantization,
    data_flow::ImplementationType impl_type
)
    : TensorNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_LayerNorm,
          {},
          {"_y", "_mean", "_rstd", "_x", "_eps"},
          impl_type
      ),
      normalized_shape_(normalized_shape), elementwise_affine_(false), bias_(false), y_layout_(y_layout),
      mean_layout_(mean_layout), rstd_layout_(rstd_layout), x_layout_(x_layout), gamma_layout_(std::nullopt),
      beta_layout_(std::nullopt), fixed_quantization_(quantization) {}

LayerNormNode::LayerNormNode(
    size_t element_id,
    const DebugInfo& debug_info,
    graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const symbolic::MultiExpression& normalized_shape,
    const TensorLayout& y_layout,
    const TensorLayout& mean_layout,
    const TensorLayout& rstd_layout,
    const TensorLayout& x_layout,
    const TensorLayout& gamma_layout,
    QuantizationType quantization,
    data_flow::ImplementationType impl_type
)
    : TensorNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_LayerNorm,
          {},
          {"_y", "_mean", "_rstd", "_x", "_eps", "_gamma"},
          impl_type
      ),
      normalized_shape_(normalized_shape), elementwise_affine_(true), bias_(false), y_layout_(y_layout),
      mean_layout_(mean_layout), rstd_layout_(rstd_layout), x_layout_(x_layout), gamma_layout_(gamma_layout),
      beta_layout_(std::nullopt), fixed_quantization_(quantization) {}

LayerNormNode::LayerNormNode(
    size_t element_id,
    const DebugInfo& debug_info,
    graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const symbolic::MultiExpression& normalized_shape,
    const TensorLayout& y_layout,
    const TensorLayout& mean_layout,
    const TensorLayout& rstd_layout,
    const TensorLayout& x_layout,
    const TensorLayout& gamma_layout,
    const TensorLayout& beta_layout,
    QuantizationType quantization,
    data_flow::ImplementationType impl_type
)
    : TensorNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_LayerNorm,
          {},
          {"_y", "_mean", "_rstd", "_x", "_eps", "_gamma", "_beta"},
          impl_type
      ),
      normalized_shape_(normalized_shape), elementwise_affine_(true), bias_(true), y_layout_(y_layout),
      mean_layout_(mean_layout), rstd_layout_(rstd_layout), x_layout_(x_layout), gamma_layout_(gamma_layout),
      beta_layout_(beta_layout), fixed_quantization_(quantization) {}

const symbolic::MultiExpression& LayerNormNode::normalized_shape() const { return this->normalized_shape_; }

symbolic::MultiExpression LayerNormNode::non_normalized_shape() const {
    long long dims = this->x_layout_.dims() - this->normalized_shape_.size();
    symbolic::MultiExpression result;
    result.reserve(dims);
    for (long long i = 0; i < dims; i++) {
        result.push_back(this->x_layout_.get_dim(i));
    }
    return result;
}

bool LayerNormNode::elementwise_affine() const { return this->elementwise_affine_; }

bool LayerNormNode::bias() const { return this->bias_; }

const TensorLayout& LayerNormNode::y_layout() const { return this->y_layout_; }

const TensorLayout& LayerNormNode::mean_layout() const { return this->mean_layout_; }

const TensorLayout& LayerNormNode::rstd_layout() const { return this->rstd_layout_; }

const TensorLayout& LayerNormNode::x_layout() const { return this->x_layout_; }

const std::optional<TensorLayout>& LayerNormNode::gamma_layout() const { return this->gamma_layout_; }

const std::optional<TensorLayout>& LayerNormNode::beta_layout() const { return this->beta_layout_; }

QuantizationType LayerNormNode::quantization() const { return this->fixed_quantization_; }

void LayerNormNode::set_quantization(const QuantizationType quant) { this->fixed_quantization_ = quant; }

void LayerNormNode::validate(const Function& function) const {
    auto& graph = this->get_parent();

    // Checks that all edges have the same primitive type
    TensorNode::validate(function);

    // Check presence of in and out edges
    if (graph.out_degree(*this) != 0) {
        throw InvalidSDFGException(
            "LayerNormNode: Expected no outputs but got: " + std::to_string(graph.out_degree(*this))
        );
    }
    long long in_degree = 5;
    if (this->elementwise_affine_) {
        in_degree++;
        if (this->bias_) {
            in_degree++;
        }
    }
    if (graph.in_degree(*this) != in_degree) {
        throw InvalidSDFGException(
            "LayerNormNode: Expexted " + std::to_string(in_degree) +
            " inputs but got: " + std::to_string(graph.in_degree(*this))
        );
    }
    const auto* y_edge = graph.in_edge_for_connector(*this, "_y");
    if (!y_edge) {
        throw InvalidSDFGException("LayerNormNode: No memlet connected at connector: _y");
    }
    const auto* mean_edge = graph.in_edge_for_connector(*this, "_mean");
    if (!mean_edge) {
        throw InvalidSDFGException("LayerNormNode: No memlet connected at connector: _mean");
    }
    const auto* rstd_edge = graph.in_edge_for_connector(*this, "_rstd");
    if (!rstd_edge) {
        throw InvalidSDFGException("LayerNormNode: No memlet connected at connector: _rstd");
    }
    const auto* x_edge = graph.in_edge_for_connector(*this, "_x");
    if (!x_edge) {
        throw InvalidSDFGException("LayerNormNode: No memlet connected at connector: _x");
    }
    const auto* eps_edge = graph.in_edge_for_connector(*this, "_eps");
    if (!eps_edge) {
        throw InvalidSDFGException("LayerNormNode: No memlet connected at connector: _eps");
    }
    const data_flow::Memlet* gamma_edge = nullptr;
    const data_flow::Memlet* beta_edge = nullptr;
    if (this->elementwise_affine_) {
        gamma_edge = graph.in_edge_for_connector(*this, "_gamma");
        if (!gamma_edge) {
            throw InvalidSDFGException("LayerNormNode: No memlet connected at connector: _gamma");
        }
        if (this->bias_) {
            beta_edge = graph.in_edge_for_connector(*this, "_beta");
            if (!beta_edge) {
                throw InvalidSDFGException("LayerNormNode: No memlet connected at connector: _beta");
            }
        }
    }

    // Check that the in edges have tensor types as base types
    if (y_edge->base_type().type_id() != types::TypeID::Tensor) {
        throw InvalidSDFGException(
            "LayerNormNode: Expected tensor type at connector '_y' but got: " + y_edge->base_type().print()
        );
    }
    const types::Tensor& y_tensor = static_cast<const types::Tensor&>(y_edge->base_type());
    if (mean_edge->base_type().type_id() != types::TypeID::Tensor) {
        throw InvalidSDFGException(
            "LayerNormNode: Expected tensor type at connector '_mean' but got: " + mean_edge->base_type().print()
        );
    }
    const types::Tensor& mean_tensor = static_cast<const types::Tensor&>(mean_edge->base_type());
    if (rstd_edge->base_type().type_id() != types::TypeID::Tensor) {
        throw InvalidSDFGException(
            "LayerNormNode: Expected tensor type at connector '_rstd' but got: " + rstd_edge->base_type().print()
        );
    }
    const types::Tensor& rstd_tensor = static_cast<const types::Tensor&>(rstd_edge->base_type());
    if (x_edge->base_type().type_id() != types::TypeID::Tensor) {
        throw InvalidSDFGException(
            "LayerNormNode: Expected tensor type at connector '_x' but got: " + x_edge->base_type().print()
        );
    }
    const types::Tensor& x_tensor = static_cast<const types::Tensor&>(x_edge->base_type());
    const types::Tensor* gamma_tensor = nullptr;
    const types::Tensor* beta_tensor = nullptr;
    if (this->elementwise_affine_) {
        if (gamma_edge->base_type().type_id() != types::TypeID::Tensor) {
            throw InvalidSDFGException(
                "LayerNormNode: Expected tensor type at connector '_gamma' but got: " + gamma_edge->base_type().print()
            );
        }
        gamma_tensor = static_cast<const types::Tensor*>(&gamma_edge->base_type());
        if (this->bias_) {
            if (beta_edge->base_type().type_id() != types::TypeID::Tensor) {
                throw InvalidSDFGException(
                    "LayerNormNode: Expected tensor type at connector '_beta' but got: " +
                    beta_edge->base_type().print()
                );
            }
            beta_tensor = static_cast<const types::Tensor*>(&beta_edge->base_type());
        }
    }

    // Check that the tensor layouts match with the tensor types on the edges
    if (y_tensor.layout() != this->y_layout_) {
        throw InvalidSDFGException(
            "LayerNormNode: Provided tensor layout does not match the memlet tensor type for connector '_y': " +
            y_tensor.layout().toStr() + " != " + this->y_layout_.toStr()
        );
    }
    if (mean_tensor.layout() != this->mean_layout_) {
        throw InvalidSDFGException(
            "LayerNormNode: Provided tensor layout does not match the memlet tensor type for connector '_mean': " +
            mean_tensor.layout().toStr() + " != " + this->mean_layout_.toStr()
        );
    }
    if (rstd_tensor.layout() != this->rstd_layout_) {
        throw InvalidSDFGException(
            "LayerNormNode: Provided tensor layout does not match the memlet tensor type for connector '_rstd': " +
            rstd_tensor.layout().toStr() + " != " + this->rstd_layout_.toStr()
        );
    }
    if (x_tensor.layout() != this->x_layout_) {
        throw InvalidSDFGException(
            "LayerNormNode: Provided tensor layout does not match the memlet tensor type for connector '_x': " +
            x_tensor.layout().toStr() + " != " + this->x_layout_.toStr()
        );
    }
    if (this->elementwise_affine_) {
        if (gamma_tensor->layout() != this->gamma_layout_) {
            throw InvalidSDFGException(
                "LayerNormNode: Provided tensor layout does not match the memlet tensor type for connector '_gamma': " +
                gamma_tensor->layout().toStr() + " != " + this->gamma_layout_->toStr()
            );
        }
        if (this->bias_) {
            if (beta_tensor->layout() != this->beta_layout_) {
                throw InvalidSDFGException(
                    "LayerNormNode: Provided tensor layout does not match the memlet tensor type for connector "
                    "'_beta': " +
                    beta_tensor->layout().toStr() + " != " + this->beta_layout_->toStr()
                );
            }
        }
    }

    // Check the tensor shapes
    auto non_normalized_shape = this->non_normalized_shape();
    if (this->x_layout_.dims() != this->normalized_shape_.size() + non_normalized_shape.size()) {
        throw InvalidSDFGException(
            "LayerNormNode: '_x' layout dims (" + std::to_string(this->x_layout_.dims()) +
            ") != size of normalized shape (" + std::to_string(this->normalized_shape_.size()) +
            ") + size of non normalized shape (" + std::to_string(non_normalized_shape.size()) + ")"
        );
    }
    symbolic::MultiExpression full_shape;
    full_shape.insert(full_shape.end(), non_normalized_shape.begin(), non_normalized_shape.end());
    full_shape.insert(full_shape.end(), this->normalized_shape_.begin(), this->normalized_shape_.end());
    symbolic::MultiExpression empty_shape;
    this->validate_equal_shapes("y layout & full", this->y_layout_.shape(), full_shape);
    this->validate_equal_shapes("mean layout & non-normalized", this->mean_layout_.shape(), non_normalized_shape);
    this->validate_equal_shapes("rstd layout & non-normalized", this->rstd_layout_.shape(), non_normalized_shape);
    this->validate_equal_shapes("x layout & full", this->x_layout_.shape(), full_shape);
    if (this->elementwise_affine_) {
        this->validate_equal_shapes("gamma layout & normalized", this->gamma_layout_->shape(), this->normalized_shape_);
        if (this->bias_) {
            this->validate_equal_shapes("beta layout & normalized", this->beta_layout_->shape(), this->normalized_shape_);
        }
    }
}

bool LayerNormNode::supports_integer_types() const { return false; }

using Dir = passes::LibNodeExpander::InputUse;

passes::LibNodeExpander::ExpandOutcome LayerNormNode::
    expand(passes::LibNodeExpander::ExpandContext& context, structured_control_flow::Block& block) {
    // y, mean, rstd, x, eps, ?(gamma, ?(beta))
    std::vector<Dir>
        access_dirs({Dir::IndirectWrite, Dir::IndirectReadWrite, Dir::IndirectReadWrite, Dir::IndirectRead, Dir::Scalar}
        );
    if (this->elementwise_affine_) {
        access_dirs.push_back(Dir::IndirectRead);
        if (this->bias_) {
            access_dirs.push_back(Dir::IndirectRead);
        }
    }
    auto standalone = context.replacement_requires_access_nodes(access_dirs);
    if (!standalone) {
        return context.unable();
    }

    auto& new_sequence = standalone->replace_with_sequence();
    auto& builder = standalone->builder();

    auto& graph = this->get_parent();
    const auto* y_edge = graph.in_edge_for_connector(*this, "_y");
    const auto* mean_edge = graph.in_edge_for_connector(*this, "_mean");
    const auto* rstd_edge = graph.in_edge_for_connector(*this, "_rstd");
    const auto* x_edge = graph.in_edge_for_connector(*this, "_x");
    const auto* eps_edge = graph.in_edge_for_connector(*this, "_eps");
    const data_flow::Memlet* gamma_edge = nullptr;
    const data_flow::Memlet* beta_edge = nullptr;
    if (this->elementwise_affine_) {
        gamma_edge = graph.in_edge_for_connector(*this, "_gamma");
        if (this->bias_) {
            beta_edge = graph.in_edge_for_connector(*this, "_beta");
        }
    }
    auto prim_type = this->primitive_type(graph);
    types::Scalar base_type(prim_type);
    types::Scalar indvar_type(types::PrimitiveType::UInt64);

    // _ln_norm_dim_sum_int = Mul_{d in normalized_shape} d
    auto norm_dim_sum_int_container = builder.find_new_name("_ln_norm_dim_sum_int");
    builder.add_container(norm_dim_sum_int_container, indvar_type);
    builder.add_assignments(
        new_sequence, {{symbolic::symbol(norm_dim_sum_int_container), SymEngine::mul(this->normalized_shape_)}}
    );

    auto norm_dim_sum_container = builder.find_new_name("_ln_norm_dim_sum");
    builder.add_container(norm_dim_sum_container, base_type);
    {
        // _ln_norm_dim_sum = _ln_norm_dim_sum_int (cast to floating point)
        auto& block = builder.add_block(new_sequence, this->debug_info_);
        auto& norm_dim_sum_int_access = builder.add_access(block, norm_dim_sum_int_container, this->debug_info_);
        auto& norm_dim_sum_access = builder.add_access(block, norm_dim_sum_container, this->debug_info_);
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"}, this->debug_info_);
        builder.add_computational_memlet(block, norm_dim_sum_int_access, tasklet, "_in", {}, this->debug_info_);
        builder.add_computational_memlet(block, tasklet, "_out", norm_dim_sum_access, {}, this->debug_info_);
    }

    // Add map nest over non normalized shape
    auto non_normalized_shape = this->non_normalized_shape();
    structured_control_flow::Sequence* current_seq = &new_sequence;
    data_flow::Subset outer_subset;
    outer_subset.reserve(non_normalized_shape.size());
    for (auto dim : non_normalized_shape) {
        auto indvar_container = builder.find_new_name("_i");
        builder.add_container(indvar_container, indvar_type);
        auto indvar = symbolic::symbol(indvar_container);
        outer_subset.push_back(indvar);
        auto& map = builder.add_map(
            *current_seq,
            indvar,
            symbolic::Lt(indvar, dim),
            symbolic::zero(),
            symbolic::add(indvar, symbolic::one()),
            structured_control_flow::ScheduleType_Sequential::create(),
            this->debug_info_
        );
        current_seq = &map.root();
    }
    data_flow::Subset mean_rsqrt_subset;
    if (outer_subset.empty()) {
        mean_rsqrt_subset.push_back(symbolic::zero());
    } else {
        mean_rsqrt_subset.insert(mean_rsqrt_subset.end(), outer_subset.begin(), outer_subset.end());
    }

    // Compute mean over normalized shape
    auto local_sum_container = builder.find_new_name("_ln_sum");
    builder.add_container(local_sum_container, base_type);
    {
        // _ln_sum = 0.0
        auto& block = builder.add_block(*current_seq, this->debug_info_);
        auto& constant_zero = builder.add_constant(block, "0.0", base_type, this->debug_info_);
        auto& local_sum_access = builder.add_access(block, local_sum_container, this->debug_info_);
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"}, this->debug_info_);
        builder.add_computational_memlet(block, constant_zero, tasklet, "_in", {}, this->debug_info_);
        builder.add_computational_memlet(block, tasklet, "_out", local_sum_access, {}, this->debug_info_);
    }
    structured_control_flow::Sequence* mean_seq = current_seq;
    data_flow::Subset mean_subset(outer_subset);
    {
        // Add reduction nest over normalized shape
        for (auto dim : this->normalized_shape_) {
            auto indvar_container = builder.find_new_name("_i");
            builder.add_container(indvar_container, indvar_type);
            auto indvar = symbolic::symbol(indvar_container);
            mean_subset.push_back(indvar);
            auto& reduce = builder.add_reduce(
                *mean_seq,
                indvar,
                symbolic::Lt(indvar, dim),
                symbolic::zero(),
                symbolic::add(indvar, symbolic::one()),
                {{ReductionOperation::Add, local_sum_container}},
                structured_control_flow::ScheduleType_Sequential::create(),
                this->debug_info_
            );
            mean_seq = &reduce.root();
        }
    }
    {
        // _ln_sum = _ln_sum + x[non_normalized_shape, normalized_shape]
        auto& block = builder.add_block(*mean_seq, this->debug_info_);
        auto& x_access = standalone->add_indirect_read_access(block, X_INPUT_IDX);
        auto& local_sum_access_in = builder.add_access(block, local_sum_container, this->debug_info_);
        auto& local_sum_access_out = builder.add_access(block, local_sum_container, this->debug_info_);
        auto& tasklet =
            builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"}, this->debug_info_);
        builder.add_computational_memlet(
            block, x_access, tasklet, "_in1", mean_subset, x_edge->base_type(), x_edge->debug_info()
        );
        builder.add_computational_memlet(block, local_sum_access_in, tasklet, "_in2", {}, this->debug_info_);
        builder.add_computational_memlet(block, tasklet, "_out", local_sum_access_out, {}, this->debug_info_);
    }
    {
        // mean[non_normalized_shape] = _ln_sum / _ln_norm_dim_sum
        auto& block = builder.add_block(*current_seq, this->debug_info_);
        auto& local_sum_access = builder.add_access(block, local_sum_container, this->debug_info_);
        auto& norm_dim_sum_access = builder.add_access(block, norm_dim_sum_container, this->debug_info_);
        auto& mean_access = standalone->add_indirect_write_access(block, MEAN_INPUT_IDX);
        auto& tasklet =
            builder.add_tasklet(block, data_flow::TaskletCode::fp_div, "_out", {"_in1", "_in2"}, this->debug_info_);
        builder.add_computational_memlet(block, local_sum_access, tasklet, "_in1", {}, this->debug_info_);
        builder.add_computational_memlet(block, norm_dim_sum_access, tasklet, "_in2", {}, base_type, this->debug_info_);
        builder.add_computational_memlet(
            block, tasklet, "_out", mean_access, mean_rsqrt_subset, mean_edge->base_type(), mean_edge->debug_info()
        );
    }

    // Compute inversed standard deviation over normalized shape
    auto sum_of_squares_container = builder.find_new_name("_ln_sum_of_squares");
    builder.add_container(sum_of_squares_container, base_type);
    {
        // _ln_sum_of_squares = 0.0
        auto& block = builder.add_block(*current_seq, this->debug_info_);
        auto& constant_zero = builder.add_constant(block, "0.0", base_type, this->debug_info_);
        auto& sum_of_squares_access = builder.add_access(block, sum_of_squares_container, this->debug_info_);
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"}, this->debug_info_);
        builder.add_computational_memlet(block, constant_zero, tasklet, "_in", {}, this->debug_info_);
        builder.add_computational_memlet(block, tasklet, "_out", sum_of_squares_access, {}, this->debug_info_);
    }
    structured_control_flow::Sequence* std_seq = current_seq;
    data_flow::Subset std_subset(outer_subset);
    {
        // Add reduction nest over normalized shape
        for (auto dim : this->normalized_shape_) {
            auto indvar_container = builder.find_new_name("_i");
            builder.add_container(indvar_container, indvar_type);
            auto indvar = symbolic::symbol(indvar_container);
            std_subset.push_back(indvar);
            auto& reduce = builder.add_reduce(
                *std_seq,
                indvar,
                symbolic::Lt(indvar, dim),
                symbolic::zero(),
                symbolic::add(indvar, symbolic::one()),
                {{ReductionOperation::Add, sum_of_squares_container}},
                structured_control_flow::ScheduleType_Sequential::create(),
                this->debug_info_
            );
            std_seq = &reduce.root();
        }
    }
    auto diff_container = builder.find_new_name("_ln_diff");
    builder.add_container(diff_container, base_type);
    auto diff_squared_container = builder.find_new_name("_ln_diff_squared");
    builder.add_container(diff_squared_container, base_type);
    {
        // _ln_diff = x[non_normalized_shape, normalized_shape] - mean[non_normalized_shape]
        // _ln_diff_squared = _ln_diff * _ln_diff
        // _ln_sum_of_squared = _ln_sum_of_squared + _ln_diff_squared
        auto& block = builder.add_block(*std_seq, this->debug_info_);
        auto& x_access = standalone->add_indirect_read_access(block, X_INPUT_IDX);
        auto& mean_access = standalone->add_indirect_read_access(block, MEAN_INPUT_IDX);
        auto& diff_access = builder.add_access(block, diff_container, this->debug_info_);
        auto& diff_squared_access = builder.add_access(block, diff_squared_container, this->debug_info_);
        auto& sum_of_squares_access_in = builder.add_access(block, sum_of_squares_container, this->debug_info_);
        auto& sum_of_squares_access_out = builder.add_access(block, sum_of_squares_container, this->debug_info_);
        auto& sub_tasklet =
            builder.add_tasklet(block, data_flow::TaskletCode::fp_sub, "_out", {"_in1", "_in2"}, this->debug_info_);
        builder.add_computational_memlet(
            block, x_access, sub_tasklet, "_in1", std_subset, x_edge->base_type(), x_edge->debug_info()
        );
        builder.add_computational_memlet(
            block, mean_access, sub_tasklet, "_in2", mean_rsqrt_subset, mean_edge->base_type(), mean_edge->debug_info()
        );
        builder.add_computational_memlet(block, sub_tasklet, "_out", diff_access, {}, this->debug_info_);
        auto& mul_tasklet =
            builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"}, this->debug_info_);
        builder.add_computational_memlet(block, diff_access, mul_tasklet, "_in1", {}, this->debug_info_);
        builder.add_computational_memlet(block, diff_access, mul_tasklet, "_in2", {}, this->debug_info_);
        builder.add_computational_memlet(block, mul_tasklet, "_out", diff_squared_access, {}, this->debug_info_);
        auto& add_tasklet =
            builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"}, this->debug_info_);
        builder.add_computational_memlet(block, diff_squared_access, add_tasklet, "_in1", {}, this->debug_info_);
        builder.add_computational_memlet(block, sum_of_squares_access_in, add_tasklet, "_in2", {}, this->debug_info_);
        builder.add_computational_memlet(block, add_tasklet, "_out", sum_of_squares_access_out, {}, this->debug_info_);
    }
    auto var_container = builder.find_new_name("_ln_var");
    builder.add_container(var_container, base_type);
    auto var_eps_container = builder.find_new_name("_ln_var_eps");
    builder.add_container(var_eps_container, base_type);
    auto std_container = builder.find_new_name("_ln_std");
    builder.add_container(std_container, base_type);
    {
        // _ln_var = _ln_sum_of_squares / _ln_norm_dim_sum
        // _ln_var_eps = _ln_var + eps
        // _ln_std = sqrt(_ln_var_eps)
        // rstd[non_normalized_shape] = 1 / _ln_std
        auto& block = builder.add_block(*current_seq, this->debug_info_);
        auto& sum_of_squares_access = builder.add_access(block, sum_of_squares_container, this->debug_info_);
        auto& norm_dim_sum_access = builder.add_access(block, norm_dim_sum_container, this->debug_info_);
        auto& var_access = builder.add_access(block, var_container, this->debug_info_);
        auto& eps_access = standalone->add_scalar_input_access(block, EPS_INPUT_IDX);
        auto& var_eps_access = builder.add_access(block, var_eps_container, this->debug_info_);
        auto& std_access = builder.add_access(block, std_container, this->debug_info_);
        auto& constant_one = builder.add_constant(block, "1.0", base_type, this->debug_info_);
        auto& rstd_access = standalone->add_indirect_write_access(block, RSTD_INPUT_IDX);
        auto& div1_tasklet =
            builder.add_tasklet(block, data_flow::TaskletCode::fp_div, "_out", {"_in1", "_in2"}, this->debug_info_);
        builder.add_computational_memlet(block, sum_of_squares_access, div1_tasklet, "_in1", {}, this->debug_info_);
        builder.add_computational_memlet(block, norm_dim_sum_access, div1_tasklet, "_in2", {}, this->debug_info_);
        builder.add_computational_memlet(block, div1_tasklet, "_out", var_access, {}, this->debug_info_);
        auto& add_tasklet =
            builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"}, this->debug_info_);
        builder.add_computational_memlet(block, var_access, add_tasklet, "_in1", {}, this->debug_info_);
        builder.add_computational_memlet(
            block, eps_access, add_tasklet, "_in2", {}, eps_edge->base_type(), eps_edge->debug_info()
        );
        builder.add_computational_memlet(block, add_tasklet, "_out", var_eps_access, {}, this->debug_info_);
        auto& sqrt_libnode =
            builder.add_library_node<cmath::CMathNode>(block, this->debug_info_, cmath::CMathFunction::sqrt, prim_type);
        builder.add_computational_memlet(block, var_eps_access, sqrt_libnode, "_in1", {}, base_type, this->debug_info_);
        builder.add_computational_memlet(block, sqrt_libnode, "_out", std_access, {}, base_type, this->debug_info_);
        auto& div2_tasklet =
            builder.add_tasklet(block, data_flow::TaskletCode::fp_div, "_out", {"_in1", "_in2"}, this->debug_info_);
        builder.add_computational_memlet(block, constant_one, div2_tasklet, "_in1", {}, this->debug_info_);
        builder.add_computational_memlet(block, std_access, div2_tasklet, "_in2", {}, this->debug_info_);
        builder.add_computational_memlet(
            block, div2_tasklet, "_out", rstd_access, mean_rsqrt_subset, rstd_edge->base_type(), rstd_edge->debug_info()
        );
    }

    // Calculate y and apply affine and/or bias if available
    structured_control_flow::Sequence* y_seq = current_seq;
    data_flow::Subset y_subset(outer_subset), inner_subset;
    {
        // Add map nest over normalized shape
        for (auto dim : this->normalized_shape_) {
            auto indvar_container = builder.find_new_name("_i");
            builder.add_container(indvar_container, indvar_type);
            auto indvar = symbolic::symbol(indvar_container);
            y_subset.push_back(indvar);
            inner_subset.push_back(indvar);
            auto& map = builder.add_map(
                *y_seq,
                indvar,
                symbolic::Lt(indvar, dim),
                symbolic::zero(),
                symbolic::add(indvar, symbolic::one()),
                structured_control_flow::ScheduleType_Sequential::create(),
                this->debug_info_
            );
            y_seq = &map.root();
        }
    }
    auto diff2_container = builder.find_new_name("_ln_diff");
    builder.add_container(diff2_container, base_type);
    auto x_hat_container = builder.find_new_name("_ln_x_hat");
    builder.add_container(x_hat_container, base_type);
    {
        // _ln_diff = x[non_normalized_shape, normalized_shape] - mean[non_normalized_shape]
        // _ln_x_hat = _ln_diff * rstd[non_normalized_shape]
        auto& block = builder.add_block(*y_seq, this->debug_info_);
        auto& x_access = standalone->add_indirect_read_access(block, X_INPUT_IDX);
        auto& mean_access = standalone->add_indirect_read_access(block, MEAN_INPUT_IDX);
        auto& diff2_access = builder.add_access(block, diff2_container, this->debug_info_);
        auto& rstd_access = standalone->add_indirect_read_access(block, RSTD_INPUT_IDX);
        auto& x_hat_access = builder.add_access(block, x_hat_container, this->debug_info_);
        auto& sub_tasklet =
            builder.add_tasklet(block, data_flow::TaskletCode::fp_sub, "_out", {"_in1", "_in2"}, this->debug_info_);
        builder.add_computational_memlet(
            block, x_access, sub_tasklet, "_in1", y_subset, x_edge->base_type(), x_edge->debug_info()
        );
        builder.add_computational_memlet(
            block, mean_access, sub_tasklet, "_in2", mean_rsqrt_subset, mean_edge->base_type(), mean_edge->debug_info()
        );
        builder.add_computational_memlet(block, sub_tasklet, "_out", diff2_access, {}, this->debug_info_);
        auto& mul_tasklet =
            builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"}, this->debug_info_);
        builder.add_computational_memlet(block, diff2_access, mul_tasklet, "_in1", {}, this->debug_info_);
        builder.add_computational_memlet(
            block, rstd_access, mul_tasklet, "_in2", mean_rsqrt_subset, rstd_edge->base_type(), rstd_edge->debug_info()
        );
        builder.add_computational_memlet(block, mul_tasklet, "_out", x_hat_access, {}, this->debug_info_);

        if (this->bias_) {
            // y[non_normalized_shape, normalized_shape] = gamma[normalized_shape] * _ln_x_hat + beta[normalized_shape]
            auto& gamma_access = standalone->add_indirect_read_access(block, GAMMA_INPUT_IDX);
            auto& beta_access = standalone->add_indirect_read_access(block, BETA_INPUT_IDX);
            auto& y_access = standalone->add_indirect_write_access(block, Y_INPUT_IDX);
            auto& tasklet = builder.add_tasklet(
                block, data_flow::TaskletCode::fp_fma, "_out", {"_in1", "_in2", "_in3"}, this->debug_info_
            );
            builder.add_computational_memlet(
                block, gamma_access, tasklet, "_in1", inner_subset, gamma_edge->base_type(), gamma_edge->debug_info()
            );
            builder.add_computational_memlet(block, x_hat_access, tasklet, "_in2", {}, this->debug_info_);
            builder.add_computational_memlet(
                block, beta_access, tasklet, "_in3", inner_subset, beta_edge->base_type(), beta_edge->debug_info()
            );
            builder.add_computational_memlet(
                block, tasklet, "_out", y_access, y_subset, y_edge->base_type(), y_edge->debug_info()
            );
        } else if (this->elementwise_affine_) {
            // y[non_normalized_shape, normalized_shape] = gamma[normalized_shape] * _ln_x_hat
            auto& gamma_access = standalone->add_indirect_read_access(block, GAMMA_INPUT_IDX);
            auto& y_access = standalone->add_indirect_write_access(block, Y_INPUT_IDX);
            auto& tasklet =
                builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"}, this->debug_info_);
            builder.add_computational_memlet(
                block, gamma_access, tasklet, "_in1", inner_subset, gamma_edge->base_type(), gamma_edge->debug_info()
            );
            builder.add_computational_memlet(block, x_hat_access, tasklet, "_in2", {}, this->debug_info_);
            builder.add_computational_memlet(
                block, tasklet, "_out", y_access, y_subset, y_edge->base_type(), y_edge->debug_info()
            );
        } else {
            // y[non_normalized_shape, normalized_shape] = _ln_x_hat
            auto& y_access = standalone->add_indirect_write_access(block, Y_INPUT_IDX);
            auto& tasklet =
                builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"}, this->debug_info_);
            builder.add_computational_memlet(block, x_hat_access, tasklet, "_in", {}, this->debug_info_);
            builder.add_computational_memlet(
                block, tasklet, "_out", y_access, y_subset, y_edge->base_type(), y_edge->debug_info()
            );
        }
    }

    return standalone->successfully_expanded();
}

std::string LayerNormNode::toStr() const {
    std::stringstream stream;
    stream << "LayerNorm([";
    for (int i = 0; i < this->normalized_shape_.size(); i++) {
        if (i > 0) {
            stream << ",";
        }
        stream << this->normalized_shape_[i]->__str__();
    }
    stream << "], affine=" << this->elementwise_affine_ << ", bias=" << this->bias_ << ")";
    return stream.str();
}

symbolic::SymbolSet LayerNormNode::symbols() const {
    symbolic::SymbolSet syms;
    for (auto dim : this->normalized_shape_) {
        for (auto sym : symbolic::atoms(dim)) {
            syms.insert(sym);
        }
    }
    this->y_layout_.collect_symbols(syms);
    this->mean_layout_.collect_symbols(syms);
    this->rstd_layout_.collect_symbols(syms);
    this->x_layout_.collect_symbols(syms);
    if (this->elementwise_affine_) {
        this->gamma_layout_->collect_symbols(syms);
    }
    if (this->bias_) {
        this->beta_layout_->collect_symbols(syms);
    }
    return syms;
}

symbolic::Expression LayerNormNode::flop() const {
    constexpr int SQRT_FLOP_ESTIMATE = 10;

    auto outer = SymEngine::mul(this->non_normalized_shape());
    auto inner = SymEngine::mul(this->normalized_shape_);

    auto mean_flops = symbolic::add(inner, symbolic::one());
    auto std_flops =
        symbolic::add(symbolic::mul(symbolic::integer(3), inner), symbolic::integer(SQRT_FLOP_ESTIMATE + 3));
    int normalize_inner_flops = 2;
    if (this->elementwise_affine_) {
        normalize_inner_flops++;
        if (this->bias_) {
            normalize_inner_flops++;
        }
    }
    auto normalize_flops = symbolic::mul(symbolic::integer(normalize_inner_flops), inner);

    return symbolic::mul(outer, SymEngine::add({mean_flops, std_flops, normalize_flops}));
}

data_flow::PointerAccessType LayerNormNode::pointer_access_type(int input_idx) const {
    switch (input_idx) {
        case Y_INPUT_IDX:
            return data_flow::PointerAccessMeta::create_full_write_only(this->y_layout_.total_elements(), true);
        case MEAN_INPUT_IDX:
            return data_flow::PointerAccessMeta::create_full_write_only(this->mean_layout_.total_elements(), true);
        case RSTD_INPUT_IDX:
            return data_flow::PointerAccessMeta::create_full_write_only(this->rstd_layout_.total_elements(), true);
        case X_INPUT_IDX:
            return data_flow::PointerAccessMeta::create_read_only(this->x_layout_.total_elements(), true);
        case EPS_INPUT_IDX:
            return data_flow::PointerAccessMeta::create_read_only(symbolic::one(), true);
        case GAMMA_INPUT_IDX:
            if (this->elementwise_affine_) {
                return data_flow::PointerAccessMeta::create_read_only(this->gamma_layout_->total_elements(), true);
            } else {
                return nullptr;
            }
        case BETA_INPUT_IDX:
            if (this->bias_) {
                return data_flow::PointerAccessMeta::create_read_only(this->beta_layout_->total_elements(), true);
            } else {
                return nullptr;
            }
        default:
            return nullptr;
    }
}

std::unique_ptr<data_flow::DataFlowNode> LayerNormNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    if (this->bias_) {
        return std::make_unique<LayerNormNode>(
            element_id,
            this->debug_info_,
            vertex,
            parent,
            this->normalized_shape_,
            this->y_layout_,
            this->mean_layout_,
            this->rstd_layout_,
            this->x_layout_,
            *this->gamma_layout_,
            *this->beta_layout_,
            this->fixed_quantization_,
            this->implementation_type_
        );
    } else if (this->elementwise_affine_) {
        return std::make_unique<LayerNormNode>(
            element_id,
            this->debug_info_,
            vertex,
            parent,
            this->normalized_shape_,
            this->y_layout_,
            this->mean_layout_,
            this->rstd_layout_,
            this->x_layout_,
            *this->gamma_layout_,
            this->fixed_quantization_,
            this->implementation_type_
        );
    } else {
        return std::make_unique<LayerNormNode>(
            element_id,
            this->debug_info_,
            vertex,
            parent,
            this->normalized_shape_,
            this->y_layout_,
            this->mean_layout_,
            this->rstd_layout_,
            this->x_layout_,
            this->fixed_quantization_,
            this->implementation_type_
        );
    }
}

void LayerNormNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    for (long long i = 0; i < this->normalized_shape_.size(); i++) {
        this->normalized_shape_[i] = symbolic::subs(this->normalized_shape_[i], old_expression, new_expression);
    }
    this->y_layout_.replace_symbols(old_expression, new_expression);
    this->mean_layout_.replace_symbols(old_expression, new_expression);
    this->rstd_layout_.replace_symbols(old_expression, new_expression);
    this->x_layout_.replace_symbols(old_expression, new_expression);
    if (this->elementwise_affine_) {
        this->gamma_layout_->replace_symbols(old_expression, new_expression);
    }
    if (this->bias_) {
        this->beta_layout_->replace_symbols(old_expression, new_expression);
    }
}

void LayerNormNode::replace(const symbolic::ExpressionMapping& replacements) {
    for (long long i = 0; i < this->normalized_shape_.size(); i++) {
        this->normalized_shape_[i] = symbolic::subs(this->normalized_shape_[i], replacements);
    }
    this->y_layout_.replace_symbols(replacements);
    this->mean_layout_.replace_symbols(replacements);
    this->rstd_layout_.replace_symbols(replacements);
    this->x_layout_.replace_symbols(replacements);
    if (this->elementwise_affine_) {
        this->gamma_layout_->replace_symbols(replacements);
    }
    if (this->bias_) {
        this->beta_layout_->replace_symbols(replacements);
    }
}

nlohmann::json LayerNormNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    auto& node = static_cast<const LayerNormNode&>(library_node);
    nlohmann::json j;
    serializer::JSONSerializer serializer;

    j["code"] = node.code().value();

    j["normalized_shape"] = nlohmann::json::array();
    for (auto& dim : node.normalized_shape()) {
        j["normalized_shape"].push_back(serializer.expression(dim));
    }

    j["elementwise_affine"] = node.elementwise_affine();
    j["bias"] = node.bias();

    node.y_layout().serialize_to_json(j["y_layout"]);
    node.mean_layout().serialize_to_json(j["mean_layout"]);
    node.rstd_layout().serialize_to_json(j["rstd_layout"]);
    node.x_layout().serialize_to_json(j["x_layout"]);
    if (node.elementwise_affine()) {
        node.gamma_layout()->serialize_to_json(j["gamma_layout"]);
    }
    if (node.bias()) {
        node.beta_layout()->serialize_to_json(j["beta_layout"]);
    }

    j["quant"] = node.quantization();

    return j;
}

data_flow::LibraryNode& LayerNormNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    assert(j.contains("element_id"));
    assert(j.contains("code"));
    assert(j.contains("normalized_shape"));
    assert(j.contains("elementwise_affine"));
    assert(j.contains("bias"));
    assert(j.contains("y_layout"));
    assert(j.contains("mean_layout"));
    assert(j.contains("rstd_layout"));
    assert(j.contains("x_layout"));
    assert(j.contains("quant"));
    assert(j.contains("debug_info"));

    std::vector<symbolic::Expression> normalized_shape;
    if (j.contains("normalized_shape")) {
        for (const auto& dim : j["normalized_shape"]) {
            normalized_shape.push_back(symbolic::parse(dim.get<std::string>()));
        }
    }

    auto elementwise_affine = j.at("elementwise_affine").get<bool>();
    auto bias = j.at("bias").get<bool>();

    auto y_layout = TensorLayout::deserialize_from_json(j.at("y_layout"));
    auto mean_layout = TensorLayout::deserialize_from_json(j.at("mean_layout"));
    auto rstd_layout = TensorLayout::deserialize_from_json(j.at("rstd_layout"));
    auto x_layout = TensorLayout::deserialize_from_json(j.at("x_layout"));

    auto quant = j.at("quant").get<types::PrimitiveType>();

    serializer::JSONSerializer serializer;
    auto deb_info = serializer.json_to_debug_info(j.at("debug_info"));

    if (elementwise_affine) {
        assert(j.contains("gamma_layout"));
        auto gamma_layout = TensorLayout::deserialize_from_json(j.at("gamma_layout"));

        if (bias) {
            assert(j.contains("beta_layout"));
            auto beta_layout = TensorLayout::deserialize_from_json(j.at("beta_layout"));

            return builder.add_library_node<LayerNormNode>(
                parent,
                deb_info,
                normalized_shape,
                y_layout,
                mean_layout,
                rstd_layout,
                x_layout,
                gamma_layout,
                beta_layout,
                quant
            );
        }

        return builder.add_library_node<LayerNormNode>(
            parent, deb_info, normalized_shape, y_layout, mean_layout, rstd_layout, x_layout, gamma_layout, quant
        );
    }

    return builder.add_library_node<
        LayerNormNode>(parent, deb_info, normalized_shape, y_layout, mean_layout, rstd_layout, x_layout, quant);
}

} // namespace tensor
} // namespace math
} // namespace sdfg
