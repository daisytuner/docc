#include "sdfg/users/legacy_user_visitor.h"

#include <unordered_set>

#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/structured_control_flow/reduce.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg::users {

void LegacyUserVisitor::traverse(structured_control_flow::ControlFlowNode& node) {
    this->on_enter(node);
    node.accept(*this);
    this->on_leave(node);
}

bool LegacyUserVisitor::visit(structured_control_flow::Block& node) {
    auto& dataflow = node.dataflow();
    for (auto dnode : dataflow.topological_sort()) {
        if (dynamic_cast<data_flow::ConstantNode*>(dnode) != nullptr) {
            continue;
        }

        if (auto access_node = dynamic_cast<data_flow::AccessNode*>(dnode)) {
            if (!symbolic::is_pointer(symbolic::symbol(access_node->data()))) {
                if (dataflow.in_degree(*dnode) > 0) {
                    Use use = Use::WRITE;

                    // Check if the pointer itself is moved (overwritten)
                    for (auto& iedge : dataflow.in_edges(*access_node)) {
                        if (iedge.type() == data_flow::MemletType::Reference ||
                            iedge.type() == data_flow::MemletType::Dereference_Src) {
                            use = Use::MOVE;
                            break;
                        }
                    }

                    this->on_use(access_node->data(), access_node, use, LoopPart::None);
                }
                if (dataflow.out_degree(*access_node) > 0) {
                    Use use = Use::READ;

                    // Check if the pointer itself is viewed (aliased)
                    for (auto& oedge : dataflow.out_edges(*access_node)) {
                        if (oedge.type() == data_flow::MemletType::Reference ||
                            oedge.type() == data_flow::MemletType::Dereference_Dst) {
                            use = Use::VIEW;
                            break;
                        }
                    }

                    // A container feeding a library node's write-only pointer input is
                    // written *through* that pointer (the node addresses it internally),
                    // so the out-edge is a write, not a read.
                    if (use == Use::READ) {
                        for (auto& oedge : dataflow.out_edges(*access_node)) {
                            auto* lib = dynamic_cast<data_flow::LibraryNode*>(&oedge.dst());
                            if (lib == nullptr) {
                                continue;
                            }
                            auto meta = lib->pointer_access_type(oedge);
                            if (meta && meta->may_contain_writes() && !meta->may_contain_reads()) {
                                use = Use::WRITE;
                                break;
                            }
                        }
                    }

                    this->on_use(access_node->data(), access_node, use, LoopPart::None);
                }
            }
        } else if (auto library_node = dynamic_cast<data_flow::LibraryNode*>(dnode)) {
            for (auto& symbol : library_node->symbols()) {
                this->on_use(symbol->get_name(), library_node, Use::READ, LoopPart::None);
            }
        }

        for (auto& oedge : dataflow.out_edges(*dnode)) {
            std::unordered_set<std::string> used;
            for (auto dim : oedge.subset()) {
                for (auto atom : symbolic::atoms(dim)) {
                    if (used.insert(atom->get_name()).second) {
                        this->on_use(atom->get_name(), &oedge, Use::READ, LoopPart::None);
                    }
                }
            }
        }
    }
    return true;
}

bool LegacyUserVisitor::visit(structured_control_flow::AssignmentBlock& node) {
    std::unordered_set<std::string> used;
    for (auto& assignment : node.assignments()) {
        for (auto atom : symbolic::atoms(assignment.second)) {
            if (symbolic::is_pointer(atom)) {
                continue;
            }
            if (used.insert(atom->get_name()).second) {
                this->on_use(atom->get_name(), &node, Use::READ, LoopPart::None);
            }
        }
    }
    for (auto& assignment : node.assignments()) {
        this->on_use(assignment.first->get_name(), &node, Use::WRITE, LoopPart::None);
    }
    return true;
}

bool LegacyUserVisitor::visit(structured_control_flow::Sequence& node) {
    for (size_t i = 0; i < node.size(); i++) {
        auto& child = node.at(i);
        this->traverse(child);
        // Code after a return is unreachable
        if (dynamic_cast<structured_control_flow::Return*>(&child)) {
            break;
        }
    }
    return true;
}

bool LegacyUserVisitor::visit(structured_control_flow::Return& node) {
    if (node.is_data() && !node.data().empty()) {
        this->on_use(node.data(), &node, Use::READ, LoopPart::None);
    }
    return true;
}

bool LegacyUserVisitor::visit(structured_control_flow::IfElse& node) {
    std::unordered_set<std::string> used;
    for (size_t i = 0; i < node.size(); i++) {
        for (auto atom : symbolic::atoms(node.at(i).second)) {
            if (used.insert(atom->get_name()).second) {
                this->on_use(atom->get_name(), &node, Use::READ, LoopPart::None);
            }
        }
    }
    for (size_t i = 0; i < node.size(); i++) {
        this->traverse(node.at(i).first);
    }
    return true;
}

bool LegacyUserVisitor::handleStructuredLoop(structured_control_flow::StructuredLoop& loop) {
    if (auto* reduction = dynamic_cast<structured_control_flow::Reduce*>(&loop)) {
        for (const auto& entry : reduction->reductions()) {
            if (entry.original_index.is_null()) {
                continue;
            }
            for (auto use : {Use::READ, Use::WRITE}) {
                this->on_use(entry.container, reduction, use, LoopPart::None);
            }
            for (auto atom : symbolic::atoms(entry.original_index)) {
                this->on_use(atom->get_name(), reduction, Use::READ, LoopPart::None);
            }
        }
    }

    for (auto atom : symbolic::atoms(loop.init())) {
        this->on_use(atom->get_name(), &loop, Use::READ, LoopPart::Init);
    }
    this->on_use(loop.indvar()->get_name(), &loop, Use::WRITE, LoopPart::Init);

    for (auto atom : symbolic::atoms(loop.condition())) {
        this->on_use(atom->get_name(), &loop, Use::READ, LoopPart::Condition);
    }

    this->traverse(loop.root());

    for (auto atom : symbolic::atoms(loop.update())) {
        this->on_use(atom->get_name(), &loop, Use::READ, LoopPart::Update);
    }
    this->on_use(loop.indvar()->get_name(), &loop, Use::WRITE, LoopPart::Update);
    return true;
}

bool LegacyUserVisitor::visit(structured_control_flow::While& node) {
    this->traverse(node.root());
    return true;
}

// Break and Continue have no users; their control flow is approximated by the enclosing loop.
bool LegacyUserVisitor::visit(structured_control_flow::Continue& node) {
    return true;
}

bool LegacyUserVisitor::visit(structured_control_flow::Break& node) {
    return true;
}

} // namespace sdfg::users
