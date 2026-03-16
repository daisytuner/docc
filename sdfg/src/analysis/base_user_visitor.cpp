#include "sdfg/analysis/base_user_visitor.h"

namespace sdfg::analysis {

bool BaseUserVisitor::visit(sdfg::structured_control_flow::Block& node) {
    auto& dflow = node.dataflow();
    for (auto dnode : dflow.topological_sort()) {
        data_flow::AccessNode* access_src = nullptr;
        std::string maybe_container;

        if (auto libnode = dynamic_cast<data_flow::LibraryNode*>(dnode)) {
            for (auto atom : libnode->symbols()) {
                if (!symbolic::is_nullptr(atom)) {
                    use_as_symbol_read(atom->get_name(), libnode, SymbolReadLocation::LibraryNode, -1, SymEngine::null);
                }
            }
        } else if (auto access_node = dynamic_cast<data_flow::AccessNode*>(dnode)) {
            if (dynamic_cast<data_flow::ConstantNode*>(dnode) == nullptr) {
                maybe_container = access_node->data();
                if (maybe_container != symbolic::__nullptr__()->get_name()) {
                    access_src = access_node;
                }
            }
        }

        for (auto& edge : dflow.out_edges(*dnode)) {
            if (access_src) {
                use_as_src_node(maybe_container, *access_src, edge, node);
            }

            for (int i = 0; i < edge.subset().size(); ++i) {
                auto& subset_part = edge.subset().at(i);
                for (auto& atom : symbolic::atoms(subset_part)) {
                    if (!symbolic::is_nullptr(atom)) {
                        use_as_symbol_read(atom->get_name(), &edge, SymbolReadLocation::MemletSubset, i, subset_part);
                    }
                }
            }

            if (auto dst_access = dynamic_cast<data_flow::AccessNode*>(&edge.dst())) {
                use_as_dst_node(dst_access->data(), *dst_access, edge, node); // all dsts that are access_nodes must be
                                                                              // some kind of write
            }
        }
    }

    return true;
}

bool BaseUserVisitor::visit(sdfg::structured_control_flow::Return& node) {
    if (node.is_data()) {
        auto& container = node.data();
        if (!container.empty()) {
            use_as_return_src(node.data(), node);
        }
    }

    return true;
}

bool BaseUserVisitor::visit(sdfg::structured_control_flow::Sequence& node) {
    for (int i = 0; i < node.size(); ++i) {
        auto [child, transition] = node.at(i);

        dispatch(child);

        for (auto& entry : transition.assignments()) {
            for (auto& atom : symbolic::atoms(entry.second)) {
                if (!symbolic::is_nullptr(atom)) {
                    use_as_symbol_read(atom->get_name(), &transition, SymbolReadLocation::Assignment, i, entry.second);
                }
            }
        }
    }
    return true;
}

bool BaseUserVisitor::visit(sdfg::structured_control_flow::IfElse& node) {
    for (int i = 0; i < node.size(); ++i) {
        auto [seq, condition] = node.at(i);
        for (auto& atom : symbolic::atoms(condition)) {
            if (!symbolic::is_nullptr(atom)) {
                use_as_symbol_read(atom->get_name(), &node, SymbolReadLocation::IfHeader, i, condition);
            }

            dispatch(seq);
        }
    }
    return true;
}

bool BaseUserVisitor::handleStructuredLoop(StructuredLoop& loop) {
    use_as_symbol_write(loop.indvar(), &loop, SymbolWriteLocation::LoopHeader); // for both init and update

    for (auto& atom : symbolic::atoms(loop.init())) {
        if (!symbolic::is_nullptr(atom)) {
            use_as_symbol_read(atom->get_name(), &loop, SymbolReadLocation::LoopHeader, LOC_LOOP_READ_INIT, loop.init());
        }
    }
    for (auto& atom : symbolic::atoms(loop.condition())) {
        if (!symbolic::is_nullptr(atom)) {
            use_as_symbol_read(
                atom->get_name(), &loop, SymbolReadLocation::LoopHeader, LOC_LOOP_READ_CONDITION, loop.init()
            );
        }
    }

    ActualStructuredSDFGVisitor::handleStructuredLoop(loop); // descend further

    for (auto& atom : symbolic::atoms(loop.update())) {
        if (!symbolic::is_nullptr(atom)) {
            use_as_symbol_read(atom->get_name(), &loop, SymbolReadLocation::LoopHeader, LOC_LOOP_READ_UPDATE, loop.init());
        }
    }

    return true;
}

} // namespace sdfg::analysis
