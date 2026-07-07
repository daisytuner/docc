#include "sdfg/visitor/immutable_structured_sdfg_visitor.h"

namespace sdfg {
namespace visitor {

ImmutableStructuredSDFGVisitor::
    ImmutableStructuredSDFGVisitor(StructuredSDFG& sdfg, analysis::AnalysisManager& analysis_manager)
    : sdfg_(sdfg), analysis_manager_(analysis_manager) {}

bool ImmutableStructuredSDFGVisitor::visit() { return this->visit_internal(sdfg_.root()); }

bool ImmutableStructuredSDFGVisitor::visit_internal(structured_control_flow::Sequence& parent) {
    if (this->accept(parent)) {
        return true;
    }

    for (size_t i = 0; i < parent.size(); i++) {
        auto& current = parent.at(i).first;

        if (auto block_stmt = dyn_cast<structured_control_flow::Block*>(&current)) {
            if (this->accept(*block_stmt)) {
                return true;
            }
        } else if (auto sequence_stmt = dyn_cast<structured_control_flow::Sequence*>(&current)) {
            if (this->visit_internal(*sequence_stmt)) {
                return true;
            }
        } else if (auto if_else_stmt = dyn_cast<structured_control_flow::IfElse*>(&current)) {
            if (this->accept(*if_else_stmt)) {
                return true;
            }

            for (size_t i = 0; i < if_else_stmt->size(); i++) {
                if (this->visit_internal(if_else_stmt->at(i).first)) {
                    return true;
                }
            }
        } else if (auto for_stmt = dyn_cast<structured_control_flow::For*>(&current)) {
            if (this->accept(*for_stmt)) {
                return true;
            }

            if (this->visit_internal(for_stmt->root())) {
                return true;
            }
        } else if (auto map_stmt = dyn_cast<structured_control_flow::Map*>(&current)) {
            if (this->accept(*map_stmt)) {
                return true;
            }

            if (this->visit_internal(map_stmt->root())) {
                return true;
            }
        } else if (auto reduce_stmt = dyn_cast<structured_control_flow::Reduce*>(&current)) {
            if (this->accept(*reduce_stmt)) {
                return true;
            }

            if (this->visit_internal(reduce_stmt->root())) {
                return true;
            }
        } else if (auto while_stmt = dyn_cast<structured_control_flow::While*>(&current)) {
            if (this->accept(*while_stmt)) {
                return true;
            }

            if (this->visit_internal(while_stmt->root())) {
                return true;
            }
        } else if (auto continue_stmt = dyn_cast<structured_control_flow::Continue*>(&current)) {
            if (this->accept(*continue_stmt)) {
                return true;
            }
        } else if (auto break_stmt = dyn_cast<structured_control_flow::Break*>(&current)) {
            if (this->accept(*break_stmt)) {
                return true;
            }
        } else if (auto return_stmt = dyn_cast<structured_control_flow::Return*>(&current)) {
            if (this->accept(*return_stmt)) {
                return true;
            }
        }
    }

    return false;
};

bool ImmutableStructuredSDFGVisitor::accept(structured_control_flow::Block& node) { return false; };

bool ImmutableStructuredSDFGVisitor::accept(structured_control_flow::Sequence& node) { return false; };

bool ImmutableStructuredSDFGVisitor::accept(structured_control_flow::Return& node) { return false; };

bool ImmutableStructuredSDFGVisitor::accept(structured_control_flow::IfElse& node) { return false; };

bool ImmutableStructuredSDFGVisitor::accept(structured_control_flow::While& node) { return false; };

bool ImmutableStructuredSDFGVisitor::accept(structured_control_flow::Continue& node) { return false; };

bool ImmutableStructuredSDFGVisitor::accept(structured_control_flow::Break& node) { return false; };

bool ImmutableStructuredSDFGVisitor::accept(structured_control_flow::For& node) { return false; };

bool ImmutableStructuredSDFGVisitor::accept(structured_control_flow::Map& node) { return false; };

bool ImmutableStructuredSDFGVisitor::accept(structured_control_flow::Reduce& node) { return false; };

} // namespace visitor
} // namespace sdfg
