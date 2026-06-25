#include "sdfg/structured_control_flow/block.h"

#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace structured_control_flow {

Block::Block(size_t element_id, const DebugInfo& debug_info, ControlFlowNode* parent)
    : ControlFlowNode(element_id, debug_info, parent) {
    this->dataflow_ = std::make_unique<data_flow::DataFlowGraph>(this);
};

void Block::validate(const Function& function) const {
    this->dataflow_->validate(function);
    if (this->dataflow().get_parent() != this) {
        throw InvalidSDFGException("Block::validate: Dataflow parent does not point to self");
    }
};

const data_flow::DataFlowGraph& Block::dataflow() const { return *this->dataflow_; };

data_flow::DataFlowGraph& Block::dataflow() { return *this->dataflow_; };

void Block::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    this->dataflow_->replace(old_expression, new_expression);
}

void Block::replace(const symbolic::ExpressionMapping& replacements) { this->dataflow_->replace(replacements); }

} // namespace structured_control_flow
} // namespace sdfg
