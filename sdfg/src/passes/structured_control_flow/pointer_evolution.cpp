#include "sdfg/passes/structured_control_flow/pointer_evolution.h"

#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/dominance_analysis.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/symbolic/conjunctive_normal_form.h"
#include "sdfg/symbolic/polynomials.h"

namespace sdfg {
namespace passes {

IteratorToIndvar::IteratorToIndvar(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
    : NonStoppingStructuredSDFGVisitor(builder, analysis_manager) {

      };

bool IteratorToIndvar::accept(structured_control_flow::For& loop) {
    // Pattern: recursive pointer update
    // a = &a[1]

    // Criterion: Find iterator
    auto& users_analysis = analysis_manager_.get<analysis::Users>();
    analysis::UsersView body_users(users_analysis, loop.root());
    if (body_users.moves().size() != 1) {
        return false;
    }
    auto move = body_users.moves().at(0);
    auto& iterator = move->container();

    // Check whether update is recursive
    auto& move_dst = static_cast<data_flow::AccessNode&>(*move->element());
    auto& dfg = move_dst.get_parent();
    auto& move_edge = *dfg.in_edges(move_dst).begin();
    if (move_edge.type() != data_flow::MemletType::Reference) {
        return false;
    }
    auto& move_src = static_cast<data_flow::AccessNode&>(move_edge.src());
    if (move_src.data() != iterator) {
        return false;
    }

    // Criterion: Offset must be constant and in bytes
    auto& move_subset = move_edge.subset();
    if (move_subset.size() != 1) {
        return false;
    }
    auto offset = move_subset.at(0);
    if (!SymEngine::is_a<SymEngine::Integer>(*offset)) {
        return false;
    }
    int offset_int = SymEngine::rcp_static_cast<const SymEngine::Integer>(offset)->as_int();
    if (offset_int <= 0) {
        return false;
    }
    if (move_edge.base_type().type_id() != types::TypeID::Pointer) {
        return false;
    }
    auto& base_type = static_cast<const types::Pointer&>(move_edge.base_type());
    if (!base_type.has_pointee_type() || base_type.pointee_type().type_id() != types::TypeID::Scalar) {
        return false;
    }
    auto& pointee_type = static_cast<const types::Scalar&>(base_type.pointee_type());
    if (pointee_type.primitive_type() != types::PrimitiveType::UInt8 &&
        pointee_type.primitive_type() != types::PrimitiveType::Int8) {
        return false;
    }

    // All uses of iterator happen before update
    std::unordered_set<data_flow::Memlet*> edges;
    auto& dominance_analysis = analysis_manager_.get<analysis::DominanceAnalysis>();
    for (auto& use : body_users.uses(iterator)) {
        // Ignore move
        if (use == move) {
            continue;
        }
        // Ignore view of move
        if (use->element() == &move_src && use->use() == analysis::Use::VIEW) {
            continue;
        }
        if (!dynamic_cast<data_flow::AccessNode*>(use->element())) {
            return false;
        }
        auto& use_node = static_cast<data_flow::AccessNode&>(*use->element());
        auto& use_dfg = use_node.get_parent();

        // No second view
        if (use->use() == analysis::Use::VIEW) {
            return false;
        }
        // Happens before
        if (!dominance_analysis.post_dominates(*move, *use)) {
            return false;
        }

        if (use->use() == analysis::Use::READ) {
            for (auto& edge : use_dfg.out_edges(use_node)) {
                auto& subset = edge.subset();
                if (subset.size() != 1) {
                    return false;
                }
                if (!symbolic::eq(subset.at(0), symbolic::zero())) {
                    return false;
                }
                // Criterion: offseted bytes should equal interpreted type
                auto& use_edge_type = edge.base_type();
                if (use_edge_type.type_id() != types::TypeID::Pointer) {
                    return false;
                }
                auto& use_pointee_type = static_cast<const types::Pointer&>(use_edge_type).pointee_type();
                if (use_pointee_type.type_id() != types::TypeID::Scalar) {
                    return false;
                }
                auto use_pointee_bitwidth = types::bit_width(use_pointee_type.primitive_type());
                if (use_pointee_bitwidth != offset_int * 8) {
                    return false;
                }
                edges.insert(&edge);
            }
        } else if (use->use() == analysis::Use::WRITE) {
            for (auto& edge : use_dfg.in_edges(use_node)) {
                auto& subset = edge.subset();
                if (subset.size() != 1) {
                    return false;
                }
                if (!symbolic::eq(subset.at(0), symbolic::zero())) {
                    return false;
                }
                // Criterion: offseted bytes should equal interpreted type
                auto& use_edge_type = edge.base_type();
                if (use_edge_type.type_id() != types::TypeID::Pointer) {
                    return false;
                }
                auto& use_pointee_type = static_cast<const types::Pointer&>(use_edge_type).pointee_type();
                if (use_pointee_type.type_id() != types::TypeID::Scalar) {
                    return false;
                }
                auto use_pointee_bitwidth = types::bit_width(use_pointee_type.primitive_type());
                if (use_pointee_bitwidth != offset_int * 8) {
                    return false;
                }
                edges.insert(&edge);
            }
        } else {
            // Other uses are not allowed
            return false;
        }
    }

    // Step 1: Replace all subsets with iterator - init
    data_flow::Subset new_subset = {symbolic::sub(loop.indvar(), loop.init())};
    for (auto& edge : edges) {
        edge->set_subset(new_subset);
    }

    // Step 2: Remove iterator
    auto& block = static_cast<structured_control_flow::Block&>(*dfg.get_parent());
    builder_.remove_memlet(block, move_edge);
    builder_.remove_node(block, move_dst);
    builder_.remove_node(block, move_src);

    return true;
};

} // namespace passes
} // namespace sdfg
