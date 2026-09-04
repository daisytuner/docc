#include "sdfg/data_flow/library_nodes/math/tensor/index_node.h"

#include <cstddef>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/data_flow/data_flow_node.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_layout.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_node.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/data_flow/pointer_metadata.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/element.h"
#include "sdfg/exceptions.h"
#include "sdfg/graph/graph.h"
#include "sdfg/passes/expansion/lib_node_expander.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/tensor.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace math {
namespace tensor {

static std::vector<std::string> make_index_inputs(const std::vector<long long>& indices) {
    std::vector<std::string> inputs = {"Y", "X"};
    for (long long i = 0; i < indices.size(); i++) {
        inputs.push_back("I" + std::to_string(indices[i]));
    }
    return inputs;
}

IndexNode::IndexNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::vector<long long>& indices,
    const TensorLayout& y_layout,
    const TensorLayout& x_layout,
    const std::vector<TensorLayout>& index_layouts,
    const data_flow::ImplementationType& impl_type
)
    : TensorNode(element_id, debug_info, vertex, parent, LibraryNodeType_Index, {}, make_index_inputs(indices), impl_type),
      indices_(indices), y_layout_(y_layout), x_layout_(x_layout), index_layouts_(index_layouts) {}

long long IndexNode::num_indices() const { return this->indices_.size(); }

const std::vector<long long>& IndexNode::indices() const { return this->indices_; }

bool IndexNode::contiguous_indices() const {
    long long num_indices = this->num_indices();
    for (long long i = 1; i < num_indices; i++) {
        if (this->indices_[i - 1] + 1 != this->indices_[i]) {
            return false;
        }
    }
    return true;
}

const TensorLayout& IndexNode::y_layout() const { return this->y_layout_; }

const TensorLayout& IndexNode::x_layout() const { return this->x_layout_; }

const std::vector<TensorLayout>& IndexNode::index_layouts() const { return this->index_layouts_; }

symbolic::MultiExpression IndexNode::common_indices_shape() const {
    long long num_indices = this->num_indices();
    symbolic::MultiExpression broadcast_shape = this->index_layouts_[0].shape();
    for (long long i = 1; i < num_indices; i++) {
        if (this->index_layouts_[i].dims() > broadcast_shape.size()) {
            long long offset = this->index_layouts_[i].dims() - broadcast_shape.size();
            symbolic::MultiExpression new_broadcast_shape = this->index_layouts_[i].shape();
            for (long long j = 0; j < new_broadcast_shape.size(); j++) {
                if (j < offset) {
                    continue;
                }
                if (symbolic::eq(new_broadcast_shape[j], broadcast_shape[j - offset]) ||
                    symbolic::eq(broadcast_shape[j - offset], symbolic::one())) {
                    continue;
                } else if (symbolic::eq(new_broadcast_shape[j], symbolic::one())) {
                    new_broadcast_shape[j] = broadcast_shape[j - offset];
                } else {
                    throw InvalidSDFGException(
                        "IndexNode: Could not broadcast shapes together with dimensions " +
                        broadcast_shape[j - offset]->__str__() + " and " + new_broadcast_shape[j]->__str__()
                    );
                }
            }
            broadcast_shape = new_broadcast_shape;
        } else {
            long long offset = static_cast<long long>(broadcast_shape.size()) - this->index_layouts_[i].dims();
            for (long long j = 0; j < broadcast_shape.size(); j++) {
                if (j < offset) {
                    continue;
                }
                auto index_dim = this->index_layouts_[i].get_dim(j - offset);
                if (symbolic::eq(broadcast_shape[j], index_dim) || symbolic::eq(index_dim, symbolic::one())) {
                    continue;
                } else if (symbolic::eq(broadcast_shape[j], symbolic::one())) {
                    broadcast_shape[j] = index_dim;
                } else {
                    throw InvalidSDFGException(
                        "IndexNode: Could not broadcast shapes together with dimensions " +
                        broadcast_shape[j]->__str__() + " and " + index_dim->__str__()
                    );
                }
            }
        }
    }
    return broadcast_shape;
}

void IndexNode::validate(const Function& function) const {
    auto& graph = this->get_parent();

    // NOTE: The base TensorNode::validate enforces that all connected memlets share one
    // primitive type. That does not hold here: the index tensors are integer-typed while
    // the data tensors may be floating-point. We therefore validate at the MathNode level
    // and add our own structural checks.
    MathNode::validate(function);

    // Check that indices and index layouts have the same size
    long long num_indices = this->num_indices();
    if (this->index_layouts_.size() != num_indices) {
        throw InvalidSDFGException(
            "IndexNode: Sizes of indices and index layouts mismatch: " + std::to_string(num_indices) +
            " != " + std::to_string(this->index_layouts_.size())
        );
    }

    // Check that there is at least one index
    if (num_indices <= 0) {
        throw InvalidSDFGException("IndexNode: Expected at least one index");
    }

    // Check that the indices are ordered ascending (from small to big)
    for (long long i = 1; i < num_indices; i++) {
        if (this->indices_[i - 1] >= this->indices_[i]) {
            throw InvalidSDFGException("IndexNode: Indices are not ordered ascending");
        }
    }

    // Check that all indices are valid (>= 0 and < x_layout.dims)
    int x_dims = this->x_layout_.dims();
    for (long long i = 0; i < num_indices; i++) {
        if (this->indices_[i] < 0 || this->indices_[i] >= x_dims) {
            throw InvalidSDFGException(
                "IndexNode: Index " + std::to_string(this->indices_[i]) + " at position " + std::to_string(i) +
                " is out of bounds [0, " + std::to_string(x_dims - 1) + "]"
            );
        }
    }

    // Check presence of in and out edges
    if (graph.out_degree(*this) != 0) {
        throw InvalidSDFGException("IndexNode: Expected no outputs but got: " + std::to_string(graph.out_degree(*this)));
    }
    long long in_degree = num_indices + 2;
    if (graph.in_degree(*this) != in_degree) {
        throw InvalidSDFGException(
            "IndexNode: Expexted " + std::to_string(in_degree) +
            " inputs but got: " + std::to_string(graph.in_degree(*this))
        );
    }
    const auto* y_edge = graph.in_edge_for_connector(*this, "Y");
    if (!y_edge) {
        throw InvalidSDFGException("IndexNode: No memlet connected at connector: Y");
    }
    const auto* x_edge = graph.in_edge_for_connector(*this, "X");
    if (!x_edge) {
        throw InvalidSDFGException("IndexNode: No memlet connected at connector: X");
    }
    std::vector<std::string> indices_conns;
    indices_conns.reserve(num_indices);
    for (long long i = 0; i < num_indices; i++) {
        indices_conns.push_back("I" + std::to_string(this->indices_[i]));
    }
    std::vector<const data_flow::Memlet*> indices_edges;
    indices_edges.reserve(num_indices);
    for (long long i = 0; i < num_indices; i++) {
        const auto* index_edge = graph.in_edge_for_connector(*this, indices_conns[i]);
        if (!index_edge) {
            throw InvalidSDFGException("IndexNode: No memlet connected at connector: " + indices_conns[i]);
        }
        indices_edges.push_back(index_edge);
    }

    // Check that the in edges have tensor types as base types
    if (y_edge->base_type().type_id() != types::TypeID::Tensor) {
        throw InvalidSDFGException(
            "IndexNode: Expected tensor type at connector 'Y' but got: " + y_edge->base_type().print()
        );
    }
    const types::Tensor& y_tensor = static_cast<const types::Tensor&>(y_edge->base_type());
    if (x_edge->base_type().type_id() != types::TypeID::Tensor) {
        throw InvalidSDFGException(
            "IndexNode: Expected tensor type at connector 'X' but got: " + x_edge->base_type().print()
        );
    }
    const types::Tensor& x_tensor = static_cast<const types::Tensor&>(x_edge->base_type());
    std::vector<const types::Tensor*> indices_tensors;
    indices_tensors.reserve(num_indices);
    for (long long i = 0; i < num_indices; i++) {
        if (indices_edges[i]->base_type().type_id() != types::TypeID::Tensor) {
            throw InvalidSDFGException(
                "IndexNode: Expected tensor type at connector '" + indices_conns[i] +
                "' but got: " + indices_edges[i]->base_type().print()
            );
        }
        indices_tensors.push_back(static_cast<const types::Tensor*>(&indices_edges[i]->base_type()));
    }

    // Check that the tensor layouts match with the tensor types on the edges
    if (y_tensor.layout() != this->y_layout_) {
        throw InvalidSDFGException(
            "IndexNode: Provided tensor layout does not match the memlet tensor type for connector 'Y': " +
            y_tensor.layout().toStr() + " != " + this->y_layout_.toStr()
        );
    }
    if (x_tensor.layout() != this->x_layout_) {
        throw InvalidSDFGException(
            "IndexNode: Provided tensor layout does not match the memlet tensor type for connector 'X': " +
            x_tensor.layout().toStr() + " != " + this->x_layout_.toStr()
        );
    }
    for (long long i = 0; i < num_indices; i++) {
        if (indices_tensors[i]->layout() != this->index_layouts_[i]) {
            throw InvalidSDFGException(
                "IndexNode: Provided tensor layout does not match the memlet tensor type for connector '" +
                indices_conns[i] + "': " + indices_tensors[i]->layout().toStr() +
                " != " + this->index_layouts_[i].toStr()
            );
        }
    }

    // Check that x and y edges have the same primitive type
    if (y_edge->base_type().primitive_type() != x_edge->base_type().primitive_type()) {
        throw InvalidSDFGException("IndexNode: Mismatching primitive types at memlet edges 'X' and 'Y'");
    }

    // Check that indices edges have an integer primitive type
    for (long long i = 0; i < num_indices; i++) {
        if (!types::is_integer(indices_edges[i]->base_type().primitive_type())) {
            throw InvalidSDFGException(
                "IndexNode: Expected integer primitive type at memlet edge '" + indices_conns[i] +
                "' but got: " + types::primitive_type_to_string(indices_edges[i]->base_type().primitive_type())
            );
        }
    }

    // Determine & check broadcast shape
    symbolic::MultiExpression broadcast_shape = this->common_indices_shape();

    // Check that shapes match
    bool contiguous_indices = this->contiguous_indices();
    symbolic::MultiExpression dummy_shape;
    if (contiguous_indices) {
        for (long long i = 0; i < this->indices_[0]; i++) {
            dummy_shape.push_back(this->x_layout_.get_dim(i));
        }
        dummy_shape.insert(dummy_shape.end(), broadcast_shape.begin(), broadcast_shape.end());
        for (long long i = this->indices_.back() + 1; i < x_dims; i++) {
            dummy_shape.push_back(this->x_layout_.get_dim(i));
        }
    } else {
        dummy_shape.insert(dummy_shape.end(), broadcast_shape.begin(), broadcast_shape.end());
        for (long long i = 0, j = 0; i < x_dims; i++) {
            if (j < num_indices && i == this->indices_[j]) {
                j++;
            } else {
                dummy_shape.push_back(this->x_layout_.get_dim(i));
            }
        }
    }
    this->validate_shape_matches(dummy_shape, this->y_layout_, "IndexNode");
}

bool IndexNode::supports_integer_types() const { return true; }

using Dir = passes::LibNodeExpander::InputUse;

passes::LibNodeExpander::ExpandOutcome IndexNode::
    expand(passes::LibNodeExpander::ExpandContext& context, structured_control_flow::Block& block) {
    std::vector<Dir> access_dirs = {Dir::IndirectWrite, Dir::IndirectRead};
    long long num_indices = this->num_indices();
    for (long long i = 0; i < num_indices; i++) {
        access_dirs.push_back(Dir::IndirectRead);
    }
    auto standalone = context.replacement_requires_access_nodes(access_dirs);
    if (!standalone) {
        return context.unable();
    }

    auto& new_sequence = standalone->replace_with_sequence();
    auto& builder = standalone->builder();

    auto& graph = this->get_parent();
    const auto* y_edge = graph.in_edge_for_connector(*this, "Y");
    const auto* x_edge = graph.in_edge_for_connector(*this, "X");
    std::vector<const data_flow::Memlet*> indices_edges;
    indices_edges.reserve(num_indices);
    for (long long i = 0; i < num_indices; i++) {
        indices_edges.push_back(graph.in_edge_for_connector(*this, "I" + std::to_string(this->indices_[i])));
    }
    auto prim_type = y_edge->base_type().primitive_type();
    types::Scalar base_type(prim_type);
    types::Scalar indvar_type(types::PrimitiveType::Int64);
    bool contiguous_indices = this->contiguous_indices();
    symbolic::MultiExpression broadcast_shape = this->common_indices_shape();
    long long broadcast_dims = broadcast_shape.size();

    // Create map nest over y dimensions
    structured_control_flow::Sequence* current_seq = &new_sequence;
    data_flow::Subset y_subset;
    int y_dims = this->y_layout_.dims();
    for (long long i = 0; i < y_dims; i++) {
        auto indvar_container = builder.find_new_name("_i");
        builder.add_container(indvar_container, indvar_type);
        auto indvar = symbolic::symbol(indvar_container);
        y_subset.push_back(indvar);

        auto& map = builder.add_map(
            *current_seq,
            indvar,
            symbolic::Lt(indvar, this->y_layout_.get_dim(i)),
            symbolic::zero(),
            symbolic::add(indvar, symbolic::one()),
            structured_control_flow::ScheduleType_Sequential::create(),
            this->debug_info_
        );
        current_seq = &map.root();
    }

    // Determine index tensors subset
    data_flow::Subset indices_subset;
    indices_subset.reserve(broadcast_dims);
    long long offset = 0;
    if (contiguous_indices) {
        offset = this->indices_[0];
    }
    for (long long i = 0; i < broadcast_dims; i++) {
        indices_subset.push_back(y_subset.at(i + offset));
    }

    // Extract x indvars from index tensors
    symbolic::MultiExpression index_indvars;
    index_indvars.reserve(num_indices);
    for (long long i = 0; i < num_indices; i++) {
        // Create new index container
        auto index_container = builder.find_new_name("_i");
        types::Scalar index_type(indices_edges[i]->base_type().primitive_type());
        builder.add_container(index_container, index_type);
        index_indvars.push_back(symbolic::symbol(index_container));

        // Create tensor type based on the broadcast shape
        symbolic::MultiExpression new_indices_strides;
        new_indices_strides.reserve(broadcast_dims);
        long long offset = broadcast_dims - this->index_layouts_[i].dims();
        for (long long j = 0; j < broadcast_dims; j++) {
            if (j < offset || (symbolic::eq(this->index_layouts_[i].get_dim(j - offset), symbolic::one()) &&
                               !symbolic::eq(broadcast_shape[j], symbolic::one()))) {
                new_indices_strides.push_back(symbolic::zero());
            } else {
                new_indices_strides.push_back(this->index_layouts_[i].get_stride(j - offset));
            }
        }
        TensorLayout new_indices_layout(broadcast_shape, new_indices_strides, this->index_layouts_[i].offset());
        types::Tensor new_indices_tensor(index_type, new_indices_layout);

        // Copy the indices value to this index container
        auto& block = builder.add_block(*current_seq, this->debug_info_);
        auto& indices_access = standalone->add_indirect_read_access(block, INDEX_INPUT_OFFSET + i);
        auto& index_access = builder.add_access(block, index_container, this->debug_info_);
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"}, this->debug_info_);
        builder.add_computational_memlet(
            block, indices_access, tasklet, "_in", indices_subset, new_indices_tensor, indices_edges[i]->debug_info()
        );
        builder.add_computational_memlet(block, tasklet, "_out", index_access, {}, this->debug_info_);
    }

    // Determine x subset
    long long x_dims = this->x_layout_.dims();
    data_flow::Subset x_subset;
    x_subset.reserve(x_dims);
    if (contiguous_indices) {
        for (long long i = 0; i < this->indices_[0]; i++) {
            x_subset.push_back(y_subset[i]);
        }
        x_subset.insert(x_subset.end(), index_indvars.begin(), index_indvars.end());
        for (long long i = this->indices_[0] + broadcast_dims; i < y_dims; i++) {
            x_subset.push_back(y_subset[i]);
        }
    } else {
        for (long long i = 0, j = 0, k = broadcast_dims; i < x_dims; i++) {
            if (j < num_indices && i == this->indices_[j]) {
                x_subset.push_back(index_indvars[j]);
                j++;
            } else {
                x_subset.push_back(y_subset[k]);
                k++;
            }
        }
    }

    // Create the copy from x to y
    auto& copy_block = builder.add_block(*current_seq, this->debug_info_);
    auto& x_access = standalone->add_indirect_read_access(copy_block, X_INPUT_IDX);
    auto& y_access = standalone->add_indirect_write_access(copy_block, Y_INPUT_IDX);
    auto& tasklet = builder.add_tasklet(copy_block, data_flow::TaskletCode::assign, "_out", {"_in"}, this->debug_info_);
    builder.add_computational_memlet(
        copy_block, x_access, tasklet, "_in", x_subset, x_edge->base_type(), x_edge->debug_info()
    );
    builder.add_computational_memlet(
        copy_block, tasklet, "_out", y_access, y_subset, y_edge->base_type(), y_edge->debug_info()
    );

    return standalone->successfully_expanded();
}

std::string IndexNode::toStr() const {
    std::stringstream stream;
    stream << "Index(indices: [";
    long long num_indices = this->num_indices();
    for (long long i = 0; i < num_indices; i++) {
        if (i > 0) {
            stream << ",";
        }
        stream << this->indices_[i];
    }
    stream << "], y_layout: " << this->y_layout_ << ", x_layout: " << this->x_layout_ << ", index_layouts: [";
    for (long long i = 0; i < num_indices; i++) {
        if (i > 0) {
            stream << ", ";
        }
        stream << this->index_layouts_[i];
    }
    stream << "])";
    return stream.str();
}

symbolic::SymbolSet IndexNode::symbols() const {
    symbolic::SymbolSet syms;
    this->y_layout_.collect_symbols(syms);
    this->x_layout_.collect_symbols(syms);
    long long num_indices = this->num_indices();
    for (long long i = 0; i < num_indices; i++) {
        this->index_layouts_[i].collect_symbols(syms);
    }
    return syms;
}

symbolic::Expression IndexNode::flop() const { return symbolic::zero(); }

data_flow::PointerAccessType IndexNode::pointer_access_type(int input_idx) const {
    if (input_idx == Y_INPUT_IDX) {
        return data_flow::PointerAccessMeta::create_full_write_only(symbolic::__nullptr__(), true);
    }
    if (input_idx == X_INPUT_IDX) {
        return data_flow::PointerAccessMeta::create_read_only(symbolic::__nullptr__(), true);
    }

    long long num_indices = this->num_indices();
    if (input_idx >= INDEX_INPUT_OFFSET && input_idx < num_indices + INDEX_INPUT_OFFSET) {
        return data_flow::PointerAccessMeta::create_read_only(symbolic::__nullptr__(), true);
    }

    return nullptr;
}

std::unique_ptr<data_flow::DataFlowNode> IndexNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::make_unique<IndexNode>(
        element_id,
        this->debug_info_,
        vertex,
        parent,
        this->indices_,
        this->y_layout_,
        this->x_layout_,
        this->index_layouts_,
        this->implementation_type_
    );
}

void IndexNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    this->y_layout_.replace_symbols(old_expression, new_expression);
    this->x_layout_.replace_symbols(old_expression, new_expression);
    long long num_indices = this->num_indices();
    for (long long i = 0; i < num_indices; i++) {
        this->index_layouts_[i].replace_symbols(old_expression, new_expression);
    }
}

void IndexNode::replace(const symbolic::ExpressionMapping& replacements) {
    this->y_layout_.replace_symbols(replacements);
    this->x_layout_.replace_symbols(replacements);
    long long num_indices = this->num_indices();
    for (long long i = 0; i < num_indices; i++) {
        this->index_layouts_[i].replace_symbols(replacements);
    }
}

nlohmann::json IndexNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const auto& index_node = static_cast<const IndexNode&>(library_node);
    nlohmann::json j;
    serializer::JSONSerializer serializer;

    j["code"] = index_node.code().value();

    j["indices"] = nlohmann::json::array();
    for (long long index : index_node.indices()) {
        j["indices"].push_back(index);
    }
    index_node.y_layout().serialize_to_json(j["y_layout"]);
    index_node.x_layout().serialize_to_json(j["x_layout"]);
    j["index_layouts"] = nlohmann::json::array();
    for (const auto& index_layout : index_node.index_layouts()) {
        nlohmann::json layoutj;
        index_layout.serialize_to_json(layoutj);
        j["index_layouts"].push_back(layoutj);
    }

    return j;
}

data_flow::LibraryNode& IndexNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    assert(j.contains("element_id"));
    assert(j.contains("code"));
    assert(j.contains("debug_info"));
    assert(j.contains("indices"));
    assert(j.contains("y_layout"));
    assert(j.contains("x_layout"));
    assert(j.contains("index_layouts"));
    sdfg::serializer::JSONSerializer serializer;

    std::vector<long long> indices;
    for (const auto& index : j.at("indices")) {
        indices.push_back(index.get<long long>());
    }
    TensorLayout y_layout = TensorLayout::deserialize_from_json(j.at("y_layout"));
    TensorLayout x_layout = TensorLayout::deserialize_from_json(j.at("x_layout"));
    std::vector<TensorLayout> index_layouts;
    for (const auto& index_layout : j.at("index_layouts")) {
        index_layouts.push_back(TensorLayout::deserialize_from_json(index_layout));
    }
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    return builder.add_library_node<IndexNode>(parent, debug_info, indices, y_layout, x_layout, index_layouts);
}

} // namespace tensor
} // namespace math
} // namespace sdfg
