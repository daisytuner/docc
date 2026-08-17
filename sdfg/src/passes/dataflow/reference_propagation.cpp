#include "sdfg/passes/dataflow/reference_propagation.h"
#include <unordered_set>

#include "sdfg/analysis/dominance_analysis.h"
#include "sdfg/analysis/reference_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/element.h"
#include "sdfg/exceptions.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/reduce.h"
#include "sdfg/types/utils.h"

namespace sdfg {
namespace passes {

namespace {

// A Reduce node stores its accumulator container name denormalized in the
// ReductionInfo. Reference propagation rewrites accumulator access nodes
// directly, so walk up from a rewritten node and retarget any enclosing
// reduction to keep the ReductionInfo consistent with the dataflow graph.
void retarget_enclosing_reductions(
    data_flow::AccessNode& node, const std::string& old_container, const std::string& new_container
) {
    if (old_container == new_container) {
        return;
    }
    auto* current = dynamic_cast<structured_control_flow::ControlFlowNode*>(node.get_parent().get_parent());
    while (current != nullptr) {
        if (auto* reduce = dynamic_cast<structured_control_flow::Reduce*>(current)) {
            reduce->replace_reduction_container(old_container, new_container);
        }
        current = current->get_parent();
    }
}

} // namespace

bool ReferencePropagation::
    compatible_type(const Function& function, const data_flow::Memlet& reference, const data_flow::Memlet& target) {
    auto& ref_type = reference.base_type();
    if (ref_type.type_id() != types::TypeID::Pointer) {
        return false;
    }
    auto& ref_pointer_type = static_cast<const types::Pointer&>(ref_type);
    if (ref_pointer_type.pointee_type().type_id() != types::TypeID::Array &&
        ref_pointer_type.pointee_type().type_id() != types::TypeID::Structure) {
        return false;
    }
    auto& ref_subset = reference.subset();

    auto& tar_type = target.base_type();
    if (tar_type.type_id() != types::TypeID::Pointer) {
        return false;
    }
    auto& tar_pointer_type = static_cast<const types::Pointer&>(tar_type);
    if (tar_pointer_type.pointee_type().type_id() != types::TypeID::Scalar) {
        return false;
    }
    auto& tar_subset = target.subset();

    // Check if trailing zeros yield compatible type
    for (auto dim : tar_subset) {
        if (!symbolic::eq(dim, symbolic::zero())) {
            return false;
        }
    }
    auto expanded_subset = ref_subset;
    for (auto& dim : tar_subset) {
        expanded_subset.push_back(dim);
    }
    try {
        auto new_res_type = types::infer_type(function, ref_type, expanded_subset);
        auto tar_res_type = target.result_type(function);
        return *new_res_type == *tar_res_type;
    } catch (const InvalidSDFGException&) {
        return false;
    }
}

ReferencePropagation::ReferencePropagation()
    : Pass() {

      };

std::string ReferencePropagation::name() { return "ReferencePropagation"; };

bool ReferencePropagation::run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    bool applied = false;

    auto& sdfg = builder.subject();

    // Replaces all views
    auto& users_analysis = analysis_manager.get<analysis::Users>();
    auto& dominance_analysis = analysis_manager.get<analysis::DominanceAnalysis>();
    auto& reference_analysis = analysis_manager.get<analysis::ReferenceAnalysis>();

    std::unordered_set<data_flow::AccessNode*> replaced_nodes;
    std::unordered_set<std::string> invalidated;
    for (auto& container : sdfg.containers()) {
        if (invalidated.find(container) != invalidated.end()) {
            continue;
        }

        // Criterion: Must be a transient pointer
        if (!sdfg.is_transient(container)) {
            continue;
        }
        auto& type = sdfg.type(container);
        if (type.type_id() != types::TypeID::Pointer) {
            continue;
        }

        auto move_groups = reference_analysis.defined_by(container);
        for (auto& entry : move_groups) {
            // If not exclusive write, skip
            if (entry.second.size() != 1) {
                continue;
            }
            auto move = *entry.second.begin();
            auto user = entry.first;

            // Criterion: Must be moved by reference memlet
            auto& access_node = static_cast<data_flow::AccessNode&>(*move->element());
            auto& dataflow = access_node.get_parent();
            auto& move_edge = *dataflow.in_edges(access_node).begin();
            if (move_edge.type() != data_flow::MemletType::Reference) {
                continue;
            }
            // Criterion: Cannot be address of (&<scalar_type>)
            auto& move_subset = move_edge.subset();
            if (move_subset.empty()) {
                continue;
            }

            // Criterion: Must be viewing another container
            auto& viewed_node = static_cast<const data_flow::AccessNode&>(move_edge.src());
            if (dynamic_cast<const data_flow::ConstantNode*>(&viewed_node) != nullptr) {
                continue;
            }
            auto& viewed_container = viewed_node.data();

            // Criterion: Must be an access node
            if (!dynamic_cast<data_flow::AccessNode*>(user->element())) {
                continue;
            }

            // Criterion: Must be dominated by the move
            if (!dominance_analysis.dominates(*move, *user)) {
                continue;
            }

            // Criterion: No reassignment of pointer or view in between
            if (users_analysis.moves(viewed_container).size() > 0) {
                auto uses_between = users_analysis.all_uses_between(*move, *user);
                bool unsafe = false;
                for (auto& use : uses_between) {
                    if (use->use() != analysis::Use::MOVE) {
                        continue;
                    }
                    // Pointer is not constant
                    if (use->container() == viewed_container) {
                        unsafe = true;
                        break;
                    }
                }
                if (unsafe) {
                    continue;
                }
            }

            auto& user_node = static_cast<data_flow::AccessNode&>(*user->element());

            // Simple case: No arithmetic on pointer, just replace container
            if (move_subset.size() == 1 && symbolic::eq(move_subset[0], symbolic::zero())) {
                user_node.data(viewed_container);
                retarget_enclosing_reductions(user_node, container, viewed_container);
                applied = true;
                invalidated.insert(viewed_container);
                replaced_nodes.insert(&user_node);
                continue;
            }

            // General case: Arithmetic on pointer, need to update memlet subsets

            // Criterion: Must be computational memlets
            // Criterion: No type casting

            auto deref_type = move_edge.result_type(builder.subject());
            sdfg::types::Pointer ref_type(static_cast<const types::IType&>(*deref_type));

            bool safe = true;
            auto& user_graph = user_node.get_parent();
            for (auto& oedge : user_graph.out_edges(user_node)) {
                if (oedge.type() != data_flow::MemletType::Computational &&
                    oedge.type() != data_flow::MemletType::Reference) {
                    safe = false;
                    break;
                }
                auto& old_subset = oedge.subset();
                if (old_subset.empty()) {
                    safe = false;
                    break;
                }
                if (oedge.base_type() != ref_type) {
                    // Special case: compatible pointer types
                    if (!compatible_type(builder.subject(), move_edge, oedge)) {
                        safe = false;
                        break;
                    }
                }
            }
            if (!safe) {
                continue;
            }
            for (auto& iedge : user_graph.in_edges(user_node)) {
                if (iedge.type() != data_flow::MemletType::Computational &&
                    iedge.type() != data_flow::MemletType::Reference) {
                    safe = false;
                    break;
                }
                auto& old_subset = iedge.subset();
                if (old_subset.empty()) {
                    safe = false;
                    break;
                }
                if (iedge.base_type() != ref_type) {
                    if (!compatible_type(builder.subject(), move_edge, iedge)) {
                        safe = false;
                        break;
                    }
                }
            }
            if (!safe) {
                continue;
            }

            // Propagate pointer type

            // Step 1: Replace container
            user_node.data(viewed_container);
            retarget_enclosing_reductions(user_node, container, viewed_container);

            // Step 2: Update edges
            for (auto& oedge : user_graph.out_edges(user_node)) {
                // Compute new subset
                data_flow::Subset new_subset;
                for (auto dim : move_subset) {
                    new_subset.push_back(dim);
                }

                auto old_subset = oedge.subset();

                if (oedge.base_type() != ref_type) {
                    for (auto& dim : old_subset) {
                        new_subset.push_back(dim);
                    }
                } else {
                    // Handle first trailing dimensions
                    auto& trail_dim = old_subset.front();
                    auto& current_dim = new_subset.back();
                    auto new_dim = symbolic::add(current_dim, trail_dim);
                    new_subset.back() = new_dim;
                    old_subset.erase(old_subset.begin());

                    // Add remaining trailing dimensions
                    for (auto dim : old_subset) {
                        new_subset.push_back(dim);
                    }
                }

                // Build new type
                oedge.set_subset(new_subset);
                oedge.set_base_type(move_edge.base_type());
            }

            for (auto& iedge : user_graph.in_edges(user_node)) {
                // Compute new subset
                data_flow::Subset new_subset;
                for (auto dim : move_subset) {
                    new_subset.push_back(dim);
                }

                auto old_subset = iedge.subset();
                if (iedge.base_type() != ref_type) {
                    for (auto& dim : old_subset) {
                        new_subset.push_back(dim);
                    }
                } else {
                    // Handle first trailing dimensions
                    auto& trail_dim = old_subset.front();
                    auto& current_dim = new_subset.back();
                    auto new_dim = symbolic::add(current_dim, trail_dim);
                    new_subset.back() = new_dim;
                    old_subset.erase(old_subset.begin());

                    // Add remaining trailing dimensions
                    for (auto dim : old_subset) {
                        new_subset.push_back(dim);
                    }
                }

                iedge.set_subset(new_subset);
                iedge.set_base_type(move_edge.base_type());
            }

            applied = true;
            invalidated.insert(viewed_container);
            replaced_nodes.insert(&user_node);
        }
    }

    // Post-processing: Merge access nodes and remove dangling nodes
    // Avoid removing elements while iterating above
    for (auto* node : replaced_nodes) {
        builder.merge_siblings(*node);
    }
    for (auto* node : replaced_nodes) {
        auto& graph = node->get_parent();
        auto* block = static_cast<structured_control_flow::Block*>(graph.get_parent());
        for (auto& dnode : graph.data_nodes()) {
            if (graph.in_degree(*dnode) == 0 && graph.out_degree(*dnode) == 0) {
                builder.remove_node(*block, *dnode);
            }
        }
    }

    return applied;
};

} // namespace passes
} // namespace sdfg
