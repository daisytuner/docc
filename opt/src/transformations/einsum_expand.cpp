#include "sdfg/transformations/einsum_expand.h"

#include <cassert>
#include <cstddef>
#include <nlohmann/json_fwd.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/code_node.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/einsum/einsum.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/structured_control_flow/while.h"
#include "sdfg/symbolic/conjunctive_normal_form.h"
#include "sdfg/symbolic/symbolic.h"
#include "symengine/logic.h"
#include "symengine/symengine_rcp.h"

namespace sdfg {
namespace transformations {

symbolic::Expression EinsumExpand::cnf_to_upper_bound(const symbolic::CNF& cnf, const symbolic::Symbol indvar) {
    std::vector<symbolic::Expression> candidates;

    for (const auto& clause : cnf) {
        for (const auto& literal : clause) {
            // Comparison: indvar < expr
            if (SymEngine::is_a<SymEngine::StrictLessThan>(*literal)) {
                auto lt = SymEngine::rcp_static_cast<const SymEngine::StrictLessThan>(literal);
                if (symbolic::eq(lt->get_arg1(), indvar) && !symbolic::uses(lt->get_arg2(), indvar)) {
                    candidates.push_back(lt->get_arg2());
                }
            }
            // Comparison: indvar <= expr
            else if (SymEngine::is_a<SymEngine::LessThan>(*literal)) {
                auto le = SymEngine::rcp_static_cast<const SymEngine::LessThan>(literal);
                if (symbolic::eq(le->get_arg1(), indvar) && !symbolic::uses(le->get_arg2(), indvar)) {
                    candidates.push_back(symbolic::add(le->get_arg2(), symbolic::one()));
                }
            }
            // Comparison: indvar == expr
            else if (SymEngine::is_a<SymEngine::Equality>(*literal)) {
                auto eq = SymEngine::rcp_static_cast<const SymEngine::Equality>(literal);
                if (symbolic::eq(eq->get_arg1(), indvar) && !symbolic::uses(eq->get_arg2(), indvar)) {
                    candidates.push_back(symbolic::add(eq->get_arg2(), symbolic::one()));
                }
            }
        }
    }

    if (candidates.empty()) {
        return SymEngine::null;
    }

    // Return the smallest upper bound across all candidate constraints
    symbolic::Expression result = candidates[0];
    for (size_t i = 1; i < candidates.size(); ++i) {
        result = symbolic::min(result, candidates[i]);
    }

    return result;
}

bool EinsumExpand::subset_contains_symbol(const data_flow::Subset& subset, const symbolic::Symbol& symbol) {
    for (auto& expr : subset) {
        if (symbolic::uses(expr, symbol)) {
            return true;
        }
    }
    return false;
}

EinsumExpand::EinsumExpand(einsum::EinsumNode& einsum_node) : einsum_node_(einsum_node) {}

std::string EinsumExpand::name() const { return "EinsumExpand"; }

bool EinsumExpand::can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    // Get & check DFG
    auto& dfg = this->einsum_node_.get_parent();
    if (dfg.library_nodes().size() > 1 || dfg.tasklets().size() > 0) {
        return false;
    }

    // Get block
    auto* block = dynamic_cast<structured_control_flow::Block*>(dfg.get_parent());
    if (!block) {
        return false;
    }

    // Get & check sequence
    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto* sequence = dynamic_cast<structured_control_flow::Sequence*>(scope_analysis.parent_scope(block));
    if (!sequence) {
        return false;
    }
    if (sequence->size() > 1) {
        return false;
    }

    // Get loop
    auto* loop = dynamic_cast<structured_control_flow::StructuredLoop*>(scope_analysis.parent_scope(sequence));
    if (!loop) {
        return false;
    }

    // Check that loop is of sufficient form
    try {
        auto cnf = symbolic::conjunctive_normal_form(loop->condition());
        auto ub = this->cnf_to_upper_bound(cnf, loop->indvar());
        if (ub.is_null()) {
            return false;
        }
    } catch (const symbolic::CNFException& e) {
        return false;
    }
    if (!symbolic::eq(loop->update(), symbolic::add(loop->indvar(), symbolic::one()))) {
        return false;
    }

    // Prevent one of the input containers to be the index variable
    for (auto& iedge : dfg.in_edges(this->einsum_node_)) {
        auto& access_node = static_cast<data_flow::AccessNode&>(iedge.src());
        if (access_node.data() == loop->indvar()->get_name()) {
            return false;
        }
    }

    // Check that the index variable of the loop does not collide with an einsum dimension index
    for (auto& dim : this->einsum_node_.dims()) {
        if (symbolic::eq(loop->indvar(), dim.indvar)) {
            return false;
        }
    }

    // Check that the indices contain the index variable of the loop
    bool indvar_in_indices = false;
    for (auto& indices : this->einsum_node_.in_indices()) {
        for (auto& expr : indices) {
            if (symbolic::uses(expr, loop->indvar())) {
                indvar_in_indices = true;
                break;
            }
        }
        if (indvar_in_indices) {
            break;
        }
    }
    if (!indvar_in_indices) {
        return false;
    }

    // Create a map from each input connector to its input container of the einsum node
    std::unordered_map<std::string, std::string> in_containers;
    for (auto& iedge : dfg.in_edges(this->einsum_node_)) {
        if (dynamic_cast<data_flow::ConstantNode*>(&iedge.src())) {
            continue;
        }
        auto& access_node = static_cast<data_flow::AccessNode&>(iedge.src());
        in_containers.insert({iedge.dst_conn(), access_node.data()});
    }

    // Create a map from each output connector to its output container of the einsum node
    std::unordered_map<std::string, std::string> out_containers;
    for (auto& oedge : dfg.out_edges(this->einsum_node_)) {
        auto& access_node = static_cast<data_flow::AccessNode&>(oedge.dst());
        out_containers.insert({oedge.src_conn(), access_node.data()});
    }

    // Check if all occurrences of the output container in the inputs have the index variable of the loop in their
    // subset
    // E.g., disallow x[i] += ... * x[j] where i is the index variable of the loop
    for (size_t i = 0; i < this->einsum_node_.inputs().size() - 1; i++) {
        if (!in_containers.contains(this->einsum_node_.input(i)) ||
            in_containers.at(this->einsum_node_.input(i)) != out_containers.at(this->einsum_node_.output(0))) {
            continue;
        }
        if (!this->subset_contains_symbol(this->einsum_node_.in_indices(i), loop->indvar())) {
            return false;
        }
    }

    return true;
}

void EinsumExpand::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    // Get DFG and block
    auto& dfg = this->einsum_node_.get_parent();
    auto* block = dynamic_cast<structured_control_flow::Block*>(dfg.get_parent());
    assert(block);

    // Get sequence and loop
    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto* sequence = dynamic_cast<structured_control_flow::Sequence*>(scope_analysis.parent_scope(block));
    assert(sequence);
    auto* loop = dynamic_cast<structured_control_flow::StructuredLoop*>(scope_analysis.parent_scope(sequence));
    assert(loop);

    // Get new dimension data from loop
    symbolic::Symbol indvar = loop->indvar();
    symbolic::Expression init = loop->init();
    auto cnf = symbolic::conjunctive_normal_form(loop->condition());
    symbolic::Expression bound = this->cnf_to_upper_bound(cnf, indvar);
    assert(!bound.is_null());

    // Get the parent node of the loop
    auto* loop_parent = dynamic_cast<structured_control_flow::Sequence*>(scope_analysis.parent_scope(loop));
    assert(loop_parent);

    // Add a new block after the loop
    auto& new_block = builder.add_block_after(*loop_parent, *loop, {}, block->debug_info());

    // Copy the access to the einsum node from the old block to the new one
    std::unordered_map<std::string, data_flow::AccessNode&> out_access, in_access;
    for (auto& oedge : dfg.out_edges(this->einsum_node_)) {
        auto& access_node = static_cast<data_flow::AccessNode&>(oedge.dst());
        auto& new_access_node = builder.add_access(new_block, access_node.data(), access_node.debug_info());
        out_access.insert({oedge.src_conn(), new_access_node});
    }
    for (auto& iedge : dfg.in_edges(this->einsum_node_)) {
        auto& access_node = static_cast<data_flow::AccessNode&>(iedge.src());
        data_flow::AccessNode* new_access_node;
        if (auto* constant_node = dynamic_cast<data_flow::ConstantNode*>(&access_node)) {
            new_access_node =
                &builder
                     .add_constant(new_block, constant_node->data(), constant_node->type(), constant_node->debug_info());
        } else {
            new_access_node = &builder.add_access(new_block, access_node.data(), access_node.debug_info());
        }
        in_access.insert({iedge.dst_conn(), *new_access_node});
    }

    // Add the expanded einsum node to the new block after the loop
    std::vector<einsum::EinsumDimension> new_dims;
    new_dims.push_back({.indvar = indvar, .init = init, .bound = bound});
    for (size_t i = 0; i < this->einsum_node_.dims().size(); i++) {
        new_dims.push_back(this->einsum_node_.dim(i));
    }
    std::vector<std::string> new_inputs(this->einsum_node_.inputs().begin(), this->einsum_node_.inputs().end() - 1);
    std::vector<data_flow::Subset>
        new_in_indices(this->einsum_node_.in_indices().begin(), this->einsum_node_.in_indices().end() - 1);
    auto& new_libnode = builder.add_library_node<
        einsum::EinsumNode,
        const std::vector<std::string>&,
        const std::vector<einsum::EinsumDimension>&,
        const data_flow::Subset&,
        const std::vector<data_flow::Subset>&>(
        new_block, this->einsum_node_.debug_info(), new_inputs, new_dims, this->einsum_node_.out_indices(), new_in_indices
    );

    // Create the memlets in the new block after the loops
    for (auto& oedge : dfg.out_edges(this->einsum_node_)) {
        if (out_access.contains(oedge.src_conn())) {
            builder.add_memlet(
                new_block,
                new_libnode,
                oedge.src_conn(),
                out_access.at(oedge.src_conn()),
                oedge.dst_conn(),
                oedge.subset(),
                oedge.base_type(),
                oedge.debug_info()
            );
        }
    }
    for (auto& iedge : dfg.in_edges(this->einsum_node_)) {
        if (in_access.contains(iedge.dst_conn())) {
            builder.add_memlet(
                new_block,
                in_access.at(iedge.dst_conn()),
                iedge.src_conn(),
                new_libnode,
                iedge.dst_conn(),
                iedge.subset(),
                iedge.base_type(),
                iedge.debug_info()
            );
        }
    }

    // Remove the block with the original einsum node
    size_t block_index = sequence->index(*block);
    builder.remove_child(*sequence, block_index);

    // If possible, remove empty loop
    size_t loop_index = loop_parent->index(*loop);
    builder.remove_child(*loop_parent, loop_index);

    analysis_manager.invalidate_all();
}

void EinsumExpand::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["einsum_node_element_id"] = this->einsum_node_.element_id();
}

EinsumExpand EinsumExpand::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j) {
    assert(j.contains("einsum_node_element_id"));
    assert(j["einsum_node_element_id"].is_number_unsigned());
    size_t einsum_node_id = j["einsum_node_element_id"].get<size_t>();
    auto* einsum_node_element = builder.find_element_by_id(einsum_node_id);
    if (!einsum_node_element) {
        throw InvalidTransformationDescriptionException(
            "Element with ID " + std::to_string(einsum_node_id) + " not found"
        );
    }
    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(einsum_node_element);
    if (!einsum_node) {
        throw InvalidTransformationDescriptionException(
            "Element with ID " + std::to_string(einsum_node_id) + " is not an EinsumNode"
        );
    }

    return EinsumExpand(*einsum_node);
}

} // namespace transformations
} // namespace sdfg
