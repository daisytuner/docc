#include "sdfg/passes/dataflow/byte_reference_elimination.h"

#include "sdfg/analysis/dominance_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/types/utils.h"

namespace sdfg {
namespace passes {

ByteReferenceElimination::ByteReferenceElimination()
    : Pass() {

      };

std::string ByteReferenceElimination::name() {
    return "ByteReferenceElimination";
};

bool ByteReferenceElimination::
    run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    bool applied = false;

    auto& sdfg = builder.subject();
    auto& users_analysis = analysis_manager.get<analysis::Users>();
    auto& dominance_analysis = analysis_manager.get<analysis::DominanceAnalysis>();

    std::unordered_set<data_flow::AccessNode*> replaced_nodes;
    for (auto& name : sdfg.containers()) {
        if (!sdfg.is_transient(name)) {
            continue;
        }
        if (sdfg.type(name).type_id() != types::TypeID::Pointer) {
            continue;
        }
        auto moves = users_analysis.moves(name);
        if (moves.size() != 1) {
            continue;
        }

        // Find Move
        auto move = *moves.begin();
        auto move_node = dynamic_cast<data_flow::AccessNode*>(move->element());
        auto& move_graph = move_node->get_parent();
        auto& move_edge = *move_graph.in_edges(*move_node).begin();
        auto& move_src = static_cast<data_flow::AccessNode&>(move_edge.src());

        // Criterion: Move must be a constant offset in bytes
        auto& move_type = move_edge.base_type();
        if (move_type.type_id() != types::TypeID::Pointer) {
            continue;
        }
        auto& move_pointer_type = static_cast<const types::Pointer&>(move_type);
        if (!move_pointer_type.has_pointee_type()) {
            continue;
        }
        auto& move_pointee_type = move_pointer_type.pointee_type();
        if (move_pointee_type.type_id() != types::TypeID::Scalar) {
            continue;
        }
        auto& move_pointee_type_bytes = static_cast<const types::Scalar&>(move_pointee_type);
        if (move_pointee_type_bytes.primitive_type() != types::PrimitiveType::Int8 &&
            move_pointee_type_bytes.primitive_type() != types::PrimitiveType::UInt8) {
            continue;
        }

        auto& move_subset = move_edge.subset();
        if (move_subset.size() != 1) {
            continue;
        }
        auto move_offset = move_subset.at(0);
        if (!SymEngine::is_a<SymEngine::Integer>(*move_offset)) {
            continue;
        }
        auto move_offset_int = SymEngine::rcp_static_cast<const SymEngine::Integer>(move_offset);
        int move_offset_bytes = move_offset_int->as_int();

        // Replace uses of byte-offseted pointer
        auto uses = users_analysis.uses(name);
        for (auto& use : uses) {
            if (use->use() != analysis::Use::VIEW && use->use() != analysis::Use::READ &&
                use->use() != analysis::Use::WRITE) {
                continue;
            }

            auto access_node = dynamic_cast<data_flow::AccessNode*>(use->element());
            if (!access_node) {
                continue;
            }
            auto& use_graph = access_node->get_parent();
            if (use_graph.out_degree(*access_node) + use_graph.in_degree(*access_node) != 1) {
                continue;
            }
            data_flow::Memlet* use_edge = nullptr;
            if (use_graph.in_degree(*access_node) == 1) {
                use_edge = &*use_graph.in_edges(*access_node).begin();
            } else {
                use_edge = &*use_graph.out_edges(*access_node).begin();
            }
            if (use_edge->type() == data_flow::MemletType::Dereference_Dst ||
                use_edge->type() == data_flow::MemletType::Dereference_Src) {
                continue;
            }
            if (use_edge->subset().empty()) {
                continue;
            }

            if (!dominance_analysis.dominates(*move, *use)) {
                continue;
            }

            // Criterion: No reassignment of pointer or view in between
            if (users_analysis.moves(move_src.data()).size() > 0) {
                auto uses_between = users_analysis.all_uses_between(*move, *use);
                bool unsafe = false;
                for (auto& use : uses_between) {
                    if (use->use() != analysis::Use::MOVE) {
                        continue;
                    }
                    // Pointer is not constant
                    if (use->container() == move_src.data()) {
                        unsafe = true;
                        break;
                    }
                }
                if (unsafe) {
                    continue;
                }
            }

            // Criterion: View must be a pointer
            auto& base_type = use_edge->base_type();
            if (base_type.type_id() != types::TypeID::Pointer) {
                continue;
            }
            auto& base_pointer_type = static_cast<const types::Pointer&>(base_type);
            if (!base_pointer_type.has_pointee_type()) {
                continue;
            }
            auto& pointee_type = base_pointer_type.pointee_type();
            auto pointee_type_size = types::get_type_size(pointee_type, false);
            if (pointee_type_size.is_null()) {
                continue;
            }
            if (!SymEngine::is_a<SymEngine::Integer>(*pointee_type_size)) {
                continue;
            }
            int pointee_type_int = SymEngine::rcp_static_cast<const SymEngine::Integer>(pointee_type_size)->as_int();
            if (move_offset_bytes % pointee_type_int != 0) {
                continue;
            }
            int elements = move_offset_bytes / pointee_type_int;

            data_flow::Subset new_subset = use_edge->subset();
            auto offset = use_edge->subset().at(0);
            new_subset[0] = symbolic::add(offset, symbolic::integer(elements));
            use_edge->set_subset(new_subset);
            access_node->data(move_src.data());
            replaced_nodes.insert(access_node);

            applied = true;
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
