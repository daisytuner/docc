#include "sdfg/data_flow/library_nodes/math/tensor/concat_node.h"

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
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/element.h"
#include "sdfg/exceptions.h"
#include "sdfg/function.h"
#include "sdfg/graph/graph.h"
#include "sdfg/passes/expansion/lib_node_expander.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/tensor.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace math {
namespace tensor {

ConcatNode::ConcatNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::string& result,
    const TensorLayout& result_layout,
    const std::vector<std::string>& tensors,
    const std::vector<TensorLayout>& tensor_layouts,
    long long dim,
    const data_flow::ImplementationType& impl_type
)
    : TensorNode(element_id, debug_info, vertex, parent, LibraryNodeType_TensorConcat, {}, tensors, impl_type),
      result_layout_(result_layout), tensor_layouts_(tensor_layouts), dim_(dim) {
    this->inputs_.push_back(result);
}

const std::string& ConcatNode::result() const { return this->inputs_.back(); }

const TensorLayout& ConcatNode::result_layout() const { return this->result_layout_; }

std::vector<std::string> ConcatNode::tensors() const {
    return std::vector<std::string>(this->inputs_.begin(), this->inputs_.end() - 1);
}

const std::vector<TensorLayout>& ConcatNode::tensor_layouts() const { return this->tensor_layouts_; }

long long ConcatNode::dim() const { return this->dim_; }

void ConcatNode::validate(const Function& function) const {
    TensorNode::validate(function);

    auto& graph = this->get_parent();

    if (graph.out_degree(*this) != 0) {
        throw InvalidSDFGException("ConcatNode: Expected no outputs but got: " + std::to_string(graph.out_degree(*this)));
    }

    // Check that the tensor layouts match with the tensor types on the edges
    const data_flow::Memlet* result_iedge = graph.in_edge_for_connector(*this, this->result());
    if (!result_iedge) {
        throw InvalidSDFGException("ConcatNode: No memlet connected at connector: " + this->result());
    }
    if (result_iedge->base_type().type_id() != types::TypeID::Tensor) {
        throw InvalidSDFGException(
            "ConcatNode: Expected tensor type at connector '" + this->result() +
            "' but got: " + result_iedge->base_type().print()
        );
    }
    const types::Tensor& result_tensor = static_cast<const types::Tensor&>(result_iedge->base_type());
    if (result_tensor.layout() != this->result_layout_) {
        throw InvalidSDFGException(
            "ConcatNode: Provided tensor layout does not match the tensor type on the edge for connector '" +
            this->result() + "': " + result_tensor.layout().toStr() + " != " + this->result_layout_.toStr()
        );
    }
    for (long long i = 0; i < this->tensor_layouts_.size(); i++) {
        const std::string& tensor = this->inputs_[i];
        const data_flow::Memlet* tensor_iedge = graph.in_edge_for_connector(*this, tensor);
        if (!tensor_iedge) {
            throw InvalidSDFGException("ConcatNode: No memlet connected at connector: " + tensor);
        }
        if (tensor_iedge->base_type().type_id() != types::TypeID::Tensor) {
            throw InvalidSDFGException(
                "ConcatNode: Expected tensor type at connector '" + tensor +
                "' but got: " + tensor_iedge->base_type().print()
            );
        }
        const types::Tensor& tensor_tensor = static_cast<const types::Tensor&>(tensor_iedge->base_type());
        if (tensor_tensor.layout() != this->tensor_layouts_[i]) {
            throw InvalidSDFGException(
                "ConcatNode: Provided tensor layout does not match the tensor type on the edge for connector '" +
                tensor + "': " + tensor_tensor.layout().toStr() + " != " + this->tensor_layouts_[i].toStr()
            );
        }
    }

    // Check that the cat dimension is < the tensor layout dimensions
    if (this->dim_ < 0 || this->dim_ >= this->result_layout_.dims()) {
        throw InvalidSDFGException(
            "ConcatNode: Cat dimension must be in [0, " + std::to_string(this->result_layout_.dims() - 1) +
            "] but got: " + std::to_string(this->dim_)
        );
    }

    // Check that all tensor layout have the same shape except in the cat dimension
    symbolic::Expression cat_dim = symbolic::zero();
    for (const TensorLayout& tensor_layout : this->tensor_layouts_) {
        int dims = tensor_layout.dims();
        if (dims != this->result_layout_.dims()) {
            throw InvalidSDFGException(
                "ConcatNode: Tensor layouts have different dimensions: " + tensor_layout.toStr() +
                " != " + this->result_layout_.toStr()
            );
        }
        for (long long i = 0; i < dims; i++) {
            if (i == this->dim_) {
                cat_dim = symbolic::add(cat_dim, tensor_layout.get_dim(i));
                continue; // Skip the cat dimension
            }
            if (!symbolic::eq(tensor_layout.get_dim(i), this->result_layout_.get_dim(i))) {
                throw InvalidSDFGException(
                    "ConcatNode: Shapes do not match at position " + std::to_string(i) + ": " + tensor_layout.toStr() +
                    " != " + this->result_layout_.toStr()
                );
            }
        }
    }
    if (!symbolic::eq(cat_dim, this->result_layout_.get_dim(this->dim_))) {
        throw InvalidSDFGException(
            "ConcatNode: Cat dimension does not match with the sum of tensor dimensions: " + cat_dim->__str__() +
            " != " + this->result_layout_.get_dim(this->dim_)->__str__()
        );
    }
}

bool ConcatNode::supports_integer_types() const { return true; }

using Dir = passes::LibNodeExpander::InputUse;

passes::LibNodeExpander::ExpandOutcome ConcatNode::
    expand(passes::LibNodeExpander::ExpandContext& context, structured_control_flow::Block& block) {
    std::vector<Dir> access_dirs(this->inputs_.size(), Dir::IndirectRead);
    access_dirs.back() = Dir::IndirectWrite;
    auto standalone = context.replacement_requires_access_nodes(access_dirs);

    if (!standalone) {
        return context.unable();
    }

    auto& builder = standalone->builder();

    // Add a graph after the current block
    auto& new_sequence = standalone->replace_with_sequence();

    auto& dfg = this->get_parent();

    types::Scalar indvar_type(types::PrimitiveType::UInt64);
    structured_control_flow::Sequence* current_seq = &new_sequence;
    data_flow::Subset subset;
    subset.reserve(this->result_layout_.dims());
    for (auto dim : this->result_layout_.shape()) {
        auto indvar_container = builder.find_new_name("_i");
        builder.add_container(indvar_container, indvar_type);
        auto indvar = symbolic::symbol(indvar_container);
        subset.push_back(indvar);
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

    auto& if_else = builder.add_if_else(*current_seq, this->debug_info_);
    auto dim_indvar = subset.at(this->dim_);
    const auto* iedge_result = dfg.in_edge_for_connector(*this, this->result());
    if (!iedge_result) {
        throw InvalidSDFGException("ConcatNode: Cannot get in edge for connector: " + this->result());
    }

    symbolic::Expression offset = symbolic::zero();
    for (size_t i = 0; i < this->tensor_layouts_.size(); i++) {
        auto new_offset = symbolic::add(offset, this->tensor_layouts_[i].get_dim(this->dim_));
        auto condition = symbolic::And(symbolic::Ge(dim_indvar, offset), symbolic::Lt(dim_indvar, new_offset));
        auto& case_seq = builder.add_case(if_else, condition, this->debug_info_);

        data_flow::Subset offset_subset(subset);
        offset_subset[this->dim_] = symbolic::sub(dim_indvar, offset);

        auto& block = builder.add_block(case_seq, this->debug_info_);
        auto& tensor_access = standalone->add_scalar_input_access(block, i);
        auto& result_access = standalone->add_scalar_input_access(block, this->tensor_layouts_.size());
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"}, this->debug_info_);

        const auto* iedge_tensor = dfg.in_edge_for_connector(*this, this->inputs_[i]);
        if (!iedge_tensor) {
            throw InvalidSDFGException("ConcatNode: Cannot get in edge for connector: " + this->inputs_[i]);
        }
        builder.add_computational_memlet(
            block, tensor_access, tasklet, "_in", offset_subset, iedge_tensor->base_type(), iedge_tensor->debug_info()
        );
        builder.add_computational_memlet(
            block, tasklet, "_out", result_access, subset, iedge_result->base_type(), iedge_result->debug_info()
        );

        offset = new_offset;
    }

    return standalone->successfully_expanded();
}

std::string ConcatNode::toStr() const {
    std::stringstream stream;
    stream << "ConcatNode(" << this->result() << ": [";
    for (long long i = 0; i < this->result_layout_.dims(); i++) {
        if (i > 0) {
            stream << ",";
        }
        stream << this->result_layout_.get_dim(i)->__str__();
    }
    stream << "], {";
    for (long long i = 0; i < this->tensor_layouts_.size(); i++) {
        if (i > 0) {
            stream << ", ";
        }
        stream << this->inputs_[i] << ": [";
        for (long long j = 0; j < this->tensor_layouts_[i].dims(); j++) {
            if (j > 0) {
                stream << ",";
            }
            stream << this->tensor_layouts_[i].get_dim(j)->__str__();
        }
        stream << "]";
    }
    stream << "}, dim: " << this->dim_ << ")";
    return stream.str();
}

symbolic::SymbolSet ConcatNode::symbols() const {
    symbolic::SymbolSet syms;
    this->result_layout_.collect_symbols(syms);
    for (const TensorLayout& tensor_layout : this->tensor_layouts_) {
        tensor_layout.collect_symbols(syms);
    }
    return syms;
}

symbolic::Expression ConcatNode::flop() const { return symbolic::zero(); }

std::unique_ptr<data_flow::DataFlowNode> ConcatNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::make_unique<ConcatNode>(
        element_id,
        this->debug_info_,
        vertex,
        parent,
        this->result(),
        this->result_layout_,
        this->tensors(),
        this->tensor_layouts_,
        this->dim_,
        this->implementation_type_
    );
}

void ConcatNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    this->result_layout_.replace_symbols(old_expression, new_expression);
    for (TensorLayout& tensor_layout : this->tensor_layouts_) {
        tensor_layout.replace_symbols(old_expression, new_expression);
    }
}

void ConcatNode::replace(const symbolic::ExpressionMapping& replacements) {
    this->result_layout_.replace_symbols(replacements);
    for (TensorLayout& tensor_layout : this->tensor_layouts_) {
        tensor_layout.replace_symbols(replacements);
    }
}

nlohmann::json ConcatNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const ConcatNode& concat_node = static_cast<const ConcatNode&>(library_node);
    nlohmann::json j;

    j["code"] = concat_node.code().value();

    j["result"] = concat_node.result();
    concat_node.result_layout().serialize_to_json(j["result_layout"]);

    j["tensors"] = nlohmann::json::array();
    for (std::string& tensor : concat_node.tensors()) {
        j["tensors"].push_back(tensor);
    }
    j["tensor_layouts"] = nlohmann::json::array();
    for (const TensorLayout& tensor_layout : concat_node.tensor_layouts()) {
        nlohmann::json tensor_layout_j;
        tensor_layout.serialize_to_json(tensor_layout_j);
        j["tensor_layouts"].push_back(tensor_layout_j);
    }

    j["dim"] = concat_node.dim();

    return j;
}

data_flow::LibraryNode& ConcatNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    assert(j.contains("element_id"));
    assert(j.contains("code"));
    assert(j.contains("result"));
    assert(j.contains("result_layout"));
    assert(j.contains("tensors"));
    assert(j.contains("tensor_layouts"));
    assert(j.contains("dim"));
    assert(j.contains("debug_info"));

    std::string result = j.at("result").get<std::string>();
    TensorLayout result_layout = TensorLayout::deserialize_from_json(j.at("result_layout"));

    std::vector<std::string> tensors = j.at("tensors").get<std::vector<std::string>>();
    std::vector<TensorLayout> tensor_layouts;
    tensor_layouts.reserve(j.at("tensor_layouts").size());
    for (long long i = 0; i < j.at("tensor_layouts").size(); i++) {
        tensor_layouts.push_back(TensorLayout::deserialize_from_json(j.at("tensor_layouts").at(i)));
    }

    long long dim = j.at("dim").get<long long>();

    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j.at("debug_info"));

    return builder.add_library_node<ConcatNode>(parent, debug_info, result, result_layout, tensors, tensor_layouts, dim);
}

} // namespace tensor
} // namespace math
} // namespace sdfg
