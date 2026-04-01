#include "sdfg/einsum/einsum.h"

#include <cstddef>
#include <memory>
#include <nlohmann/json_fwd.hpp>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/data_flow/data_flow_node.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/library_nodes/math/math_node.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/element.h"
#include "sdfg/exceptions.h"
#include "sdfg/function.h"
#include "sdfg/graph/graph.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/type.h"
#include "symengine/symbol.h"

namespace sdfg {
namespace einsum {

EinsumNode::EinsumNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::vector<std::string>& inputs,
    const std::vector<EinsumDimension>& dims,
    const data_flow::Subset& out_indices,
    const std::vector<data_flow::Subset>& in_indices,
    bool rename_indvars
)
    : math::MathNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_Einsum,
          {"__einsum_out"},
          inputs,
          data_flow::ImplementationType_NONE
      ),
      dims_(dims), out_indices_(out_indices), in_indices_(in_indices) {
    // Check list sizes
    if (inputs.size() != in_indices.size()) {
        throw InvalidSDFGException("EinsumNode: Number of input containers != number of input indices");
    }

    // Rename indvars to internal symbols (only for fresh construction, not clone/deserialize)
    if (rename_indvars) {
        // Build mapping from original indvars to internal symbols
        // Format: _einsum_node_{element_id}_{original_indvar_name}
        std::string prefix = "_einsum_node_" + std::to_string(element_id) + "_";
        std::vector<std::pair<symbolic::Symbol, symbolic::Symbol>> indvar_renames;
        for (const auto& dim : this->dims_) {
            auto old_indvar = dim.indvar;
            auto old_name = SymEngine::rcp_static_cast<const SymEngine::Symbol>(old_indvar)->get_name();
            auto new_indvar = symbolic::symbol(prefix + old_name);
            indvar_renames.push_back({old_indvar, new_indvar});
        }

        // Apply all substitutions
        for (size_t idx = 0; idx < indvar_renames.size(); idx++) {
            auto old_indvar = indvar_renames[idx].first;
            auto new_indvar = indvar_renames[idx].second;

            // Replace in all dims' init, bound, and indvar
            for (auto& d : this->dims_) {
                if (symbolic::eq(d.indvar, old_indvar)) {
                    d.indvar = new_indvar;
                }
                d.init = symbolic::subs(d.init, old_indvar, new_indvar);
                d.bound = symbolic::subs(d.bound, old_indvar, new_indvar);
            }

            // Replace in out_indices
            for (size_t i = 0; i < this->out_indices_.size(); i++) {
                this->out_indices_[i] = symbolic::subs(this->out_indices_[i], old_indvar, new_indvar);
            }

            // Replace in in_indices
            for (size_t i = 0; i < this->in_indices_.size(); i++) {
                for (size_t j = 0; j < this->in_indices_[i].size(); j++) {
                    this->in_indices_[i][j] = symbolic::subs(this->in_indices_[i][j], old_indvar, new_indvar);
                }
            }
        }
    }

    // Append output at the end
    this->inputs_.push_back("__einsum_out");
    this->in_indices_.push_back(this->out_indices_);
}

const std::vector<EinsumDimension>& EinsumNode::dims() const { return this->dims_; }

const EinsumDimension& EinsumNode::dim(size_t index) const { return this->dims_.at(index); }

const symbolic::Symbol& EinsumNode::indvar(size_t index) const { return this->dims_.at(index).indvar; }

const symbolic::Expression& EinsumNode::init(size_t index) const { return this->dims_.at(index).init; }

const symbolic::Expression& EinsumNode::bound(size_t index) const { return this->dims_.at(index).bound; }

const data_flow::Subset& EinsumNode::out_indices() const { return this->out_indices_; }

const symbolic::Expression& EinsumNode::out_index(size_t index) const { return this->out_indices_.at(index); }

const std::vector<data_flow::Subset>& EinsumNode::in_indices() const { return this->in_indices_; }

const data_flow::Subset& EinsumNode::in_indices(size_t index) const { return this->in_indices_.at(index); }

const symbolic::Expression& EinsumNode::in_index(size_t index1, size_t index2) const {
    return this->in_indices_.at(index1).at(index2);
}

symbolic::SymbolSet EinsumNode::internal_symbols() const {
    symbolic::SymbolSet result;
    for (auto& dim : this->dims()) {
        result.insert(dim.indvar);
    }
    return result;
}

bool EinsumNode::expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    // Get data flow graph and block
    auto& dfg = this->get_parent();
    auto* block = dynamic_cast<structured_control_flow::Block*>(dfg.get_parent());
    if (!block) {
        return false;
    }

    // Get parent sequence
    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto* sequence = dynamic_cast<structured_control_flow::Sequence*>(scope_analysis.parent_scope(block));
    if (!sequence) {
        return false;
    }

    // Create block after this block
    auto& block_after = builder.add_block_after(*sequence, *block, {}, block->debug_info());

    // Collect and transfer nodes after the EinsumNode
    bool before = true;
    std::unordered_map<data_flow::DataFlowNode*, data_flow::DataFlowNode*> nodes_after;
    for (auto* node : dfg.topological_sort()) {
        if (before) {
            if (node == this) {
                before = false;
            }
            continue;
        }
        data_flow::DataFlowNode* node_after = nullptr;
        if (auto* constant_node = dynamic_cast<data_flow::ConstantNode*>(node)) {
            node_after =
                &builder
                     .add_constant(block_after, constant_node->data(), constant_node->type(), constant_node->debug_info());
        } else if (auto* access_node = dynamic_cast<data_flow::AccessNode*>(node)) {
            if (dfg.out_degree(*access_node) == 0 && dfg.in_degree(*access_node) == 1 &&
                &(*dfg.in_edges(*access_node).begin()).src() == this) {
                continue;
            }
            node_after = &builder.add_access(block_after, access_node->data(), access_node->debug_info());
        } else if (auto* code_node = dynamic_cast<data_flow::CodeNode*>(node)) {
            node_after = &builder.copy_node(block_after, *code_node);
        } else {
            return false;
        }
        nodes_after.insert({node, node_after});
        if (dynamic_cast<data_flow::Tasklet*>(node) || dynamic_cast<data_flow::LibraryNode*>(node)) {
            for (auto& iedge : dfg.in_edges(*node)) {
                if (!nodes_after.contains(&iedge.src())) {
                    if (auto* constant_node = dynamic_cast<data_flow::ConstantNode*>(&iedge.src())) {
                        nodes_after.insert(
                            {constant_node,
                             &builder.add_constant(
                                 block_after, constant_node->data(), constant_node->type(), constant_node->debug_info()
                             )}
                        );
                    } else if (auto* access_node = dynamic_cast<data_flow::AccessNode*>(&iedge.src())) {
                        nodes_after.insert(
                            {access_node,
                             &builder.add_access(block_after, access_node->data(), access_node->debug_info())}
                        );
                    } else {
                        return false;
                    }
                }
            }
        }
    }

    // Transfer memlets after the EinsumNode
    for (auto& edge : dfg.edges()) {
        if (!nodes_after.contains(&edge.src()) || !nodes_after.contains(&edge.dst())) {
            continue;
        }
        builder.add_memlet(
            block_after,
            *nodes_after[&edge.src()],
            edge.src_conn(),
            *nodes_after[&edge.dst()],
            edge.dst_conn(),
            edge.subset(),
            edge.base_type(),
            edge.debug_info()
        );
    }

    // Delete transferred data flow in the original block
    std::unordered_set<data_flow::Memlet*> edges_for_removal;
    for (auto& edge : dfg.edges()) {
        if (nodes_after.contains(&edge.src()) && nodes_after.contains(&edge.dst())) {
            edges_for_removal.insert(&edge);
        }
    }
    for (auto* edge : edges_for_removal) {
        builder.remove_memlet(*block, *edge);
    }
    std::unordered_set<data_flow::DataFlowNode*> nodes_for_removal;
    for (auto& node : dfg.nodes()) {
        if (dfg.in_degree(node) == 0 && dfg.out_degree(node) == 0) {
            nodes_for_removal.insert(&node);
        }
    }
    for (auto* node : nodes_for_removal) {
        builder.remove_node(*block, *node);
    }

    // Add containers for loop induction variables (symbols already renamed in constructor)
    for (size_t i = 0; i < this->dims().size(); i++) {
        auto indvar = this->indvar(i);
        auto indvar_name = SymEngine::rcp_static_cast<const SymEngine::Symbol>(indvar)->get_name();
        if (builder.subject().exists(indvar_name)) {
            continue;
        }
        builder.add_container(indvar_name, types::Scalar(types::PrimitiveType::Int64));
    }

    // Add loops
    structured_control_flow::Sequence* current_sequence = nullptr;
    bool map = true;
    for (size_t i = 0; i < this->dims().size(); i++) {
        if (map) {
            if (i >= this->out_indices().size() || !symbolic::uses(this->out_index(i), this->indvar(i))) {
                map = false;
            } else {
                for (size_t j = 0; j < i; j++) {
                    if (symbolic::uses(this->init(i), this->indvar(j)) ||
                        symbolic::uses(this->bound(i), this->indvar(j))) {
                        map = false;
                        break;
                    }
                }
            }
        }
        auto indvar = this->indvar(i);
        auto condition = symbolic::Lt(indvar, this->bound(i));
        auto init = this->init(i);
        auto update = symbolic::add(indvar, symbolic::one());
        if (current_sequence) {
            structured_control_flow::StructuredLoop* loop;
            if (map) {
                loop = &builder.add_map(
                    *current_sequence,
                    indvar,
                    condition,
                    init,
                    update,
                    ScheduleType_Sequential::create(),
                    {},
                    this->debug_info()
                );
            } else {
                loop = &builder.add_for(*current_sequence, indvar, condition, init, update, {}, this->debug_info());
            }
            current_sequence = &loop->root();
        } else {
            structured_control_flow::StructuredLoop* loop;
            if (map) {
                loop = &builder.add_map_after(
                    *sequence,
                    *block,
                    indvar,
                    condition,
                    init,
                    update,
                    ScheduleType_Sequential::create(),
                    {},
                    this->debug_info()
                );
            } else {
                loop =
                    &builder.add_for_after(*sequence, *block, indvar, condition, init, update, {}, this->debug_info());
            }
            current_sequence = &loop->root();
        }
    }

    // Add new block
    structured_control_flow::Block* new_block;
    if (current_sequence) {
        new_block = &builder.add_block(*current_sequence);
    } else {
        new_block = &builder.add_block_after(*sequence, *block, {}, this->debug_info());
    }

    // Transfer the access nodes of the EinsumNode
    std::unordered_map<std::string, data_flow::AccessNode*> new_in_accesses;
    std::unordered_map<std::string, const types::IType&> in_types;
    for (auto& iedge : dfg.in_edges(*this)) {
        in_types.insert({iedge.dst_conn(), iedge.base_type()});
        if (auto* constant_node = dynamic_cast<data_flow::ConstantNode*>(&iedge.src())) {
            new_in_accesses.insert(
                {iedge.dst_conn(),
                 &builder
                      .add_constant(*new_block, constant_node->data(), constant_node->type(), constant_node->debug_info())
                }
            );
        } else if (auto* access_node = dynamic_cast<data_flow::AccessNode*>(&iedge.src())) {
            data_flow::AccessNode* new_access_node = nullptr;
            for (auto [conn, other_access_node] : new_in_accesses) {
                if (access_node->data() == other_access_node->data()) {
                    new_access_node = other_access_node;
                    break;
                }
            }
            if (!new_access_node) {
                new_access_node = &builder.add_access(*new_block, access_node->data(), access_node->debug_info());
            }
            new_in_accesses.insert({iedge.dst_conn(), new_access_node});
        } else {
            return false;
        }
    }
    data_flow::AccessNode* new_out_access;
    const types::IType* out_type;
    {
        auto& oedge = *dfg.out_edges(*this).begin();
        out_type = &oedge.base_type();
        if (auto* access_node = dynamic_cast<data_flow::AccessNode*>(&oedge.dst())) {
            new_out_access = &builder.add_access(*new_block, access_node->data(), access_node->debug_info());
        } else {
            return false;
        }
    }

    // Add computations to the block
    if (this->inputs().size() == 1) {
        auto& tasklet =
            builder.add_tasklet(*new_block, data_flow::TaskletCode::assign, {"_out"}, {"_in0"}, this->debug_info());
        builder.add_memlet(
            *new_block,
            *new_in_accesses.at(this->input(0)),
            "void",
            tasklet,
            "_in0",
            this->in_indices(0),
            in_types.at(this->input(0)),
            this->debug_info()
        );
        builder.add_memlet(
            *new_block, tasklet, "_out", *new_out_access, "void", this->out_indices(), *out_type, this->debug_info()
        );
    } else if (this->inputs().size() == 2) {
        auto& tasklet =
            builder
                .add_tasklet(*new_block, data_flow::TaskletCode::fp_add, {"_out"}, {"_in0", "_in1"}, this->debug_info());
        builder.add_memlet(
            *new_block,
            *new_in_accesses.at(this->input(0)),
            "void",
            tasklet,
            "_in0",
            this->in_indices(0),
            in_types.at(this->input(0)),
            this->debug_info()
        );
        builder.add_memlet(
            *new_block,
            *new_in_accesses.at(this->input(1)),
            "void",
            tasklet,
            "_in1",
            this->in_indices(1),
            in_types.at(this->input(1)),
            this->debug_info()
        );
        builder.add_memlet(
            *new_block, tasklet, "_out", *new_out_access, "void", this->out_indices(), *out_type, this->debug_info()
        );
    } else {
        // Build a mapping from original connector names to internal names and indices
        std::unordered_map<std::string, data_flow::Subset> in_indices;
        std::unordered_map<std::string, std::string> conn_to_internal;
        for (size_t i = 0; i < this->inputs().size(); i++) {
            in_indices.insert({this->input(i), this->in_indices(i)});
            conn_to_internal.insert({this->input(i), "_in" + std::to_string(i)});
        }
        long long inp;
        for (inp = 0; inp < (long long) this->inputs().size() - 3; inp++) {
            auto tmp = builder.find_new_name();
            auto& tmp_type = builder.add_container(tmp, types::Scalar(in_types.at(this->input(inp)).primitive_type()));
            auto& tmp_access = builder.add_access(*new_block, tmp);
            std::string int_conn0 = conn_to_internal.at(this->input(inp));
            std::string int_conn1 = conn_to_internal.at(this->input(inp + 1));
            auto& tasklet = builder.add_tasklet(
                *new_block, data_flow::TaskletCode::fp_mul, {"_out"}, {int_conn0, int_conn1}, this->debug_info()
            );
            builder.add_memlet(
                *new_block,
                *new_in_accesses.at(this->input(inp)),
                "void",
                tasklet,
                int_conn0,
                in_indices.at(this->input(inp)),
                in_types.at(this->input(inp)),
                this->debug_info()
            );
            builder.add_memlet(
                *new_block,
                *new_in_accesses.at(this->input(inp + 1)),
                "void",
                tasklet,
                int_conn1,
                in_indices.at(this->input(inp + 1)),
                in_types.at(this->input(inp + 1)),
                this->debug_info()
            );
            builder.add_memlet(*new_block, tasklet, "_out", tmp_access, "void", {}, tmp_type, this->debug_info());
            new_in_accesses[this->input(inp + 1)] = &tmp_access;
            in_indices[this->input(inp + 1)].clear();
            in_types.erase(this->input(inp + 1));
            in_types.insert({this->input(inp + 1), tmp_type});
        }
        std::string int_conn0 = conn_to_internal.at(this->input(inp));
        std::string int_conn1 = conn_to_internal.at(this->input(inp + 1));
        std::string int_conn2 = conn_to_internal.at(this->input(inp + 2));
        auto& tasklet = builder.add_tasklet(
            *new_block, data_flow::TaskletCode::fp_fma, {"_out"}, {int_conn0, int_conn1, int_conn2}, this->debug_info()
        );
        builder.add_memlet(
            *new_block,
            *new_in_accesses.at(this->input(inp)),
            "void",
            tasklet,
            int_conn0,
            in_indices.at(this->input(inp)),
            in_types.at(this->input(inp)),
            this->debug_info()
        );
        builder.add_memlet(
            *new_block,
            *new_in_accesses.at(this->input(inp + 1)),
            "void",
            tasklet,
            int_conn1,
            in_indices.at(this->input(inp + 1)),
            in_types.at(this->input(inp + 1)),
            this->debug_info()
        );
        builder.add_memlet(
            *new_block,
            *new_in_accesses.at(this->input(inp + 2)),
            "void",
            tasklet,
            int_conn2,
            in_indices.at(this->input(inp + 2)),
            in_types.at(this->input(inp + 2)),
            this->debug_info()
        );
        builder.add_memlet(
            *new_block, tasklet, "_out", *new_out_access, "void", this->out_indices(), *out_type, this->debug_info()
        );
    }

    // Remove EinsumNode and its access nodes and memlets
    std::unordered_set<data_flow::AccessNode*> old_accesses;
    while (dfg.in_edges(*this).begin() != dfg.in_edges(*this).end()) {
        auto& iedge = *dfg.in_edges(*this).begin();
        old_accesses.insert(dynamic_cast<data_flow::AccessNode*>(&iedge.src()));
        builder.remove_memlet(*block, iedge);
    }
    while (dfg.out_edges(*this).begin() != dfg.out_edges(*this).end()) {
        auto& oedge = *dfg.out_edges(*this).begin();
        old_accesses.insert(dynamic_cast<data_flow::AccessNode*>(&oedge.dst()));
        builder.remove_memlet(*block, oedge);
    }
    for (auto* old_access : old_accesses) {
        if (dfg.in_degree(*old_access) == 0 && dfg.out_degree(*old_access) == 0) {
            builder.remove_node(*block, *old_access);
        }
    }
    builder.remove_node(*block, *this);

    // Remove block before loops if empty
    size_t block_index = sequence->index(*block);
    if (dfg.nodes().size() == 0 && sequence->at(block_index).second.empty()) {
        builder.remove_child(*sequence, sequence->index(*block));
    }

    // Remove block after loops if empty
    if (block_after.dataflow().nodes().size() == 0) {
        builder.remove_child(*sequence, sequence->index(block_after));
    }

    return true;
}

symbolic::SymbolSet EinsumNode::symbols() const {
    symbolic::SymbolSet result;
    symbolic::SymbolSet internal = this->internal_symbols();

    // Collect only external symbols from bounds and init expressions
    for (auto& dim : this->dims()) {
        for (auto& symbol : symbolic::atoms(dim.init)) {
            if (!internal.count(symbol)) {
                result.insert(symbol);
            }
        }
        for (auto& symbol : symbolic::atoms(dim.bound)) {
            if (!internal.count(symbol)) {
                result.insert(symbol);
            }
        }
    }

    // Note: indices only contain internal indvars, so skip them

    return result;
}

void EinsumNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    // Skip if old_expression is an internal symbol (indvar)
    for (auto& dim : this->dims()) {
        if (symbolic::eq(dim.indvar, old_expression)) {
            return; // Internal symbol - do not replace
        }
    }

    // Only replace external symbols in bounds/init expressions
    for (auto& dim : this->dims_) {
        dim.init = symbolic::subs(dim.init, old_expression, new_expression);
        dim.bound = symbolic::subs(dim.bound, old_expression, new_expression);
    }

    // Note: indices only contain internal indvars, so no substitution needed
}

std::string EinsumNode::toStr() const {
    std::stringstream stream;

    stream << this->output(0);
    for (auto& index : this->out_indices()) {
        stream << "[" << index->__str__() << "]";
    }
    stream << " = ";
    size_t num_inputs = this->inputs().size();
    if (num_inputs > 1) {
        for (size_t i = 0; i < num_inputs - 1; i++) {
            if (i > 0) {
                stream << " * ";
            }
            stream << this->input(i);
            for (auto& index : this->in_indices(i)) {
                stream << "[" << index->__str__() << "]";
            }
        }
        stream << " + ";
    }
    stream << this->input(num_inputs - 1);
    for (auto& index : this->in_indices(num_inputs - 1)) {
        stream << "[" << index->__str__() << "]";
    }

    for (auto& dim : this->dims()) {
        stream << " for " << dim.indvar->__str__() << " = " << dim.init->__str__() << " : " << dim.bound->__str__();
    }

    return stream.str();
}

symbolic::Expression EinsumNode::flop() const {
    symbolic::SymbolMap dim_map;
    symbolic::Expression result = symbolic::one();

    for (size_t i = 0; i < this->dims().size(); i++) {
        symbolic::Expression dim_expr = symbolic::sub(this->bound(i), this->init(i));
        for (size_t j = 0; j < i; j++) {
            for (auto& symbol : symbolic::atoms(dim_expr)) {
                if (symbolic::eq(symbol, this->indvar(j))) {
                    dim_expr =
                        symbolic::subs(dim_expr, symbol, symbolic::div(dim_map.at(symbol), symbolic::integer(2)));
                }
            }
        }
        dim_map.insert({this->indvar(i), dim_expr});
        result = symbolic::mul(result, dim_expr);
    }

    return symbolic::mul(result, symbolic::integer(this->inputs().size() - 1));
}

std::unique_ptr<data_flow::DataFlowNode> EinsumNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::make_unique<EinsumNode>(
        element_id,
        this->debug_info(),
        vertex,
        parent,
        std::vector<std::string>(this->inputs().begin(), this->inputs().end() - 1),
        this->dims(),
        this->out_indices(),
        std::vector<data_flow::Subset>(this->in_indices().begin(), this->in_indices().end() - 1),
        false // skip renaming - already internal symbols
    );
}

void EinsumNode::validate(const Function& function) const {
    // Check inputs
    size_t inputs_size = this->inputs().size();
    if (inputs_size == 0) {
        throw InvalidSDFGException("EinsumNode: Inputs of EinsumNode must not be empty");
    }
    for (size_t i = 0; i < inputs_size - 1; i++) {
        if (this->input(i) == "__einsum_out") {
            throw InvalidSDFGException("EinsumNode: Input '__einsum_out' at wrong position");
        }
    }
    if (this->input(inputs_size - 1) != "__einsum_out") {
        throw InvalidSDFGException("EinsumNode: Last input of EinsumNode must be '__einsum_out'");
    }

    // Check last in indices
    if (this->out_indices().size() != this->in_indices(inputs_size - 1).size()) {
        throw InvalidSDFGException("EinsumNode: Out indices and last in indices have different sizes");
    }
    for (size_t i = 0; i < this->out_indices().size(); i++) {
        if (!symbolic::eq(this->out_index(i), this->in_index(inputs_size - 1, i))) {
            throw InvalidSDFGException("EinsumNode: Out indices and last in indices do not match");
        }
    }

    // Check input containers
    auto& dfg = this->get_parent();
    auto& oedge = *dfg.out_edges(*this).begin();
    std::string out_container = dynamic_cast<const data_flow::AccessNode&>(oedge.dst()).data();
    for (auto& iedge : dfg.in_edges(*this)) {
        auto& src = dynamic_cast<const data_flow::AccessNode&>(iedge.src());
        if (src.data() != out_container && iedge.dst_conn() == "__einsum_out") {
            throw InvalidSDFGException("EinsumNode: Out container must occur as a summation in the inputs");
        }
    }

    // Check if dimensions index variables occur at least once as in/out indices
    for (size_t i = 0; i < this->dims().size(); i++) {
        bool unused = true;
        for (auto& index : this->out_indices()) {
            for (auto& symbol : symbolic::atoms(index)) {
                if (symbolic::eq(this->indvar(i), symbol)) {
                    unused = false;
                    break;
                }
            }
            if (!unused) {
                break;
            }
        }
        if (!unused) {
            continue;
        }
        for (auto& indices : this->in_indices()) {
            for (auto& index : indices) {
                for (auto& symbol : symbolic::atoms(index)) {
                    if (symbolic::eq(this->indvar(i), symbol)) {
                        unused = false;
                        break;
                    }
                }
                if (!unused) {
                    break;
                }
            }
            if (!unused) {
                break;
            }
        }
        if (unused) {
            throw InvalidSDFGException(
                "EinsumNode: Dimension indvar does not occur in the in/out indices: " + this->indvar(i)->__str__()
            );
        }
    }
}

nlohmann::json EinsumSerializer::serialize(const data_flow::LibraryNode& libnode) {
    if (libnode.code() != LibraryNodeType_Einsum) {
        throw InvalidSDFGException("EinsumSerializer: Invalid library node type");
    }

    const auto& einsum_node = static_cast<const EinsumNode&>(libnode);
    serializer::JSONSymbolicPrinter printer;

    nlohmann::json j;
    j["type"] = "library_node";
    j["code"] = std::string(LibraryNodeType_Einsum.value());
    j["side_effect"] = einsum_node.side_effect();

    j["output"] = einsum_node.output(0);

    j["inputs"] = nlohmann::json::array();
    for (auto& input : einsum_node.inputs()) {
        j["inputs"].push_back(input);
    }

    j["dims"] = nlohmann::json::array();
    for (auto& dim : einsum_node.dims()) {
        nlohmann::json dimj;
        dimj["indvar"] = printer.apply(dim.indvar);
        dimj["init"] = printer.apply(dim.init);
        dimj["bound"] = printer.apply(dim.bound);
        j["dims"].push_back(dimj);
    }

    j["out_indices"] = nlohmann::json::array();
    for (auto& index : einsum_node.out_indices()) {
        j["out_indices"].push_back(printer.apply(index));
    }

    j["in_indices"] = nlohmann::json::array();
    for (auto& indices : einsum_node.in_indices()) {
        nlohmann::json indicesj = nlohmann::json::array();
        for (auto& index : indices) {
            indicesj.push_back(printer.apply(index));
        }
        j["in_indices"].push_back(indicesj);
    }

    return j;
}

data_flow::LibraryNode& EinsumSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    assert(j.contains("type"));
    assert(j["type"].is_string());
    assert(j.contains("code"));
    assert(j["code"].is_string());
    assert(j.contains("side_effect"));
    assert(j["side_effect"].is_boolean());
    assert(j.contains("output"));
    assert(j["output"].is_string());
    assert(j.contains("inputs"));
    assert(j["inputs"].is_array());
    assert(j.contains("dims"));
    assert(j["dims"].is_array());
    assert(j.contains("out_indices"));
    assert(j["out_indices"].is_array());
    assert(j.contains("in_indices"));
    assert(j["in_indices"].is_array());
    assert(j["inputs"].size() == j["in_indices"].size());

    auto type = j["type"].get<std::string>();
    if (type != "library_node") {
        throw InvalidSDFGException("EinsumSerializer: Invalid library node type");
    }

    auto code = j["code"].get<std::string>();
    if (code != LibraryNodeType_Einsum.value()) {
        throw InvalidSDFGException("EinsumSerializer: Invalid library node code");
    }

    auto side_effect = j["side_effect"].get<bool>();
    if (side_effect) {
        throw InvalidSDFGException("EinsumSerializer: EinsumNodes must be free of side effects");
    }

    auto output = j["output"].get<std::string>();
    if (output != "__einsum_out") {
        throw InvalidSDFGException("EinsumSerializer: Output of EinsumNode must be '__einsum_out'");
    }

    auto inputs = j["inputs"].get<std::vector<std::string>>();
    size_t inputs_size = inputs.size();
    if (inputs_size == 0) {
        throw InvalidSDFGException("EinsumSerializer: Inputs of EinsumNode must not be empty");
    }
    if (inputs[inputs_size - 1] != "__einsum_out") {
        throw InvalidSDFGException("EinsumSerializer: Last input of EinsumNode must be '__einsum_out'");
    }

    std::vector<EinsumDimension> dims;
    for (size_t i = 0; i < j["dims"].size(); i++) {
        auto& dimj = j["dims"][i];
        assert(dimj.is_object());
        assert(dimj.contains("indvar"));
        assert(dimj["indvar"].is_string());
        assert(dimj.contains("init"));
        assert(dimj["init"].is_string());
        assert(dimj.contains("bound"));
        assert(dimj["bound"].is_string());

        EinsumDimension dim;
        dim.indvar = symbolic::symbol(dimj["indvar"]);
        dim.init = symbolic::parse(dimj["init"]);
        dim.bound = symbolic::parse(dimj["bound"]);
        dims.push_back(dim);
    }

    data_flow::Subset out_indices;
    auto out_indices_str = j["out_indices"].get<std::vector<std::string>>();
    for (auto& index_str : out_indices_str) {
        out_indices.push_back(symbolic::parse(index_str));
    }

    std::vector<data_flow::Subset> in_indices;
    for (size_t i = 0; i < j["in_indices"].size(); i++) {
        assert(j["in_indices"][i].is_array());

        data_flow::Subset indices;
        auto indices_str = j["in_indices"][i].get<std::vector<std::string>>();
        for (auto& index_str : indices_str) {
            indices.push_back(symbolic::parse(index_str));
        }
        in_indices.push_back(indices);
    }
    if (out_indices.size() != in_indices[inputs_size - 1].size()) {
        throw InvalidSDFGException("EinsumSerializer: Out indices and last in indices have different sizes");
    }
    for (size_t i = 0; i < out_indices.size(); i++) {
        if (!symbolic::eq(out_indices[i], in_indices[inputs_size - 1][i])) {
            throw InvalidSDFGException("EinsumSerializer: Out indices and last in indices do not match");
        }
    }

    auto& einsum_node = builder.add_library_node<
        EinsumNode,
        const std::vector<std::string>&,
        const std::vector<EinsumDimension>&,
        const data_flow::Subset&,
        const std::vector<data_flow::Subset>&,
        bool>(
        parent,
        DebugInfo(),
        std::vector<std::string>(inputs.begin(), inputs.end() - 1),
        dims,
        out_indices,
        std::vector<data_flow::Subset>(in_indices.begin(), in_indices.end() - 1),
        false // skip renaming - already internal symbols from serialization
    );

    return einsum_node;
}

} // namespace einsum
} // namespace sdfg
