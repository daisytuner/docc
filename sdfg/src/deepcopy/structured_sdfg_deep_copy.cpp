#include "sdfg/deepcopy/structured_sdfg_deep_copy.h"

#include "sdfg/element.h"

namespace sdfg {
namespace deepcopy {

void StructuredSDFGDeepCopy::append(structured_control_flow::Sequence& root, structured_control_flow::Sequence& source) {
    for (size_t i = 0; i < source.size(); i++) {
        auto child = source.at(i);
        auto& node = child.first;
        auto& trans = child.second;

        if (auto block_stmt = dynamic_cast<structured_control_flow::Block*>(&node)) {
            auto& block =
                this->builder_.add_block(root, block_stmt->dataflow(), trans.assignments(), block_stmt->debug_info());
            this->node_mapping[block_stmt] = &block;
        } else if (auto sequence_stmt = dynamic_cast<structured_control_flow::Sequence*>(&node)) {
            auto& new_seq = this->builder_.add_sequence(root, trans.assignments(), sequence_stmt->debug_info());
            this->node_mapping[sequence_stmt] = &new_seq;
            this->append(new_seq, *sequence_stmt);
        } else if (auto if_else_stmt = dynamic_cast<structured_control_flow::IfElse*>(&node)) {
            auto& new_scope = this->builder_.add_if_else(root, trans.assignments(), if_else_stmt->debug_info());
            this->node_mapping[if_else_stmt] = &new_scope;
            for (size_t i = 0; i < if_else_stmt->size(); i++) {
                auto branch = if_else_stmt->at(i);
                auto& new_branch = this->builder_.add_case(new_scope, branch.second, branch.first.debug_info());
                this->node_mapping[&branch.first] = &new_branch;
                this->append(new_branch, branch.first);
            }
        } else if (auto loop_stmt = dynamic_cast<structured_control_flow::While*>(&node)) {
            auto& new_scope = this->builder_.add_while(root, trans.assignments(), loop_stmt->debug_info());
            this->node_mapping[loop_stmt] = &new_scope;
            this->append(new_scope.root(), loop_stmt->root());
        } else if (auto cont_stmt = dynamic_cast<structured_control_flow::Continue*>(&node)) {
            auto& new_cont = this->builder_.add_continue(root, trans.assignments(), cont_stmt->debug_info());
            this->node_mapping[cont_stmt] = &new_cont;
        } else if (auto br_stmt = dynamic_cast<structured_control_flow::Break*>(&node)) {
            auto& new_br = this->builder_.add_break(root, trans.assignments(), br_stmt->debug_info());
            this->node_mapping[br_stmt] = &new_br;
        } else if (auto ret_stmt = dynamic_cast<structured_control_flow::Return*>(&node)) {
            if (ret_stmt->is_data()) {
                auto& new_ret =
                    this->builder_.add_return(root, ret_stmt->data(), trans.assignments(), ret_stmt->debug_info());
                this->node_mapping[ret_stmt] = &new_ret;
            } else if (ret_stmt->is_constant()) {
                auto& new_ret = this->builder_.add_constant_return(
                    root, ret_stmt->data(), ret_stmt->type(), trans.assignments(), ret_stmt->debug_info()
                );
                this->node_mapping[ret_stmt] = &new_ret;
            }
        } else if (auto for_stmt = dynamic_cast<structured_control_flow::For*>(&node)) {
            auto& new_scope = this->builder_.add_for(
                root,
                for_stmt->indvar(),
                for_stmt->condition(),
                for_stmt->init(),
                for_stmt->update(),
                trans.assignments(),
                for_stmt->debug_info()
            );
            this->node_mapping[for_stmt] = &new_scope;
            this->append(new_scope.root(), for_stmt->root());
        } else if (auto map_stmt = dynamic_cast<structured_control_flow::Map*>(&node)) {
            auto& new_scope = this->builder_.add_map(
                root,
                map_stmt->indvar(),
                map_stmt->condition(),
                map_stmt->init(),
                map_stmt->update(),
                map_stmt->schedule_type(),
                trans.assignments(),
                map_stmt->debug_info()
            );
            this->node_mapping[map_stmt] = &new_scope;
            this->append(new_scope.root(), map_stmt->root());
        } else if (auto reduce_stmt = dynamic_cast<structured_control_flow::Reduce*>(&node)) {
            auto& new_scope = this->builder_.add_reduce(
                root,
                reduce_stmt->indvar(),
                reduce_stmt->condition(),
                reduce_stmt->init(),
                reduce_stmt->update(),
                reduce_stmt->reductions(),
                reduce_stmt->schedule_type(),
                trans.assignments(),
                reduce_stmt->debug_info()
            );
            this->node_mapping[reduce_stmt] = &new_scope;
            this->append(new_scope.root(), reduce_stmt->root());
        } else {
            throw std::runtime_error("Deep copy not implemented");
        }
    }
};

void StructuredSDFGDeepCopy::
    insert(structured_control_flow::Sequence& root, structured_control_flow::ControlFlowNode& source) {
    if (auto block_stmt = dynamic_cast<structured_control_flow::Block*>(&source)) {
        auto& block = this->builder_.add_block(root, block_stmt->dataflow(), {}, block_stmt->debug_info());
        this->node_mapping[block_stmt] = &block;
    } else if (auto sequence_stmt = dynamic_cast<structured_control_flow::Sequence*>(&source)) {
        auto& new_seq = this->builder_.add_sequence(root, {}, sequence_stmt->debug_info());
        this->node_mapping[sequence_stmt] = &new_seq;
        this->append(new_seq, *sequence_stmt);
    } else if (auto if_else_stmt = dynamic_cast<structured_control_flow::IfElse*>(&source)) {
        auto& new_scope = this->builder_.add_if_else(root);
        this->node_mapping[if_else_stmt] = &new_scope;
        for (size_t i = 0; i < if_else_stmt->size(); i++) {
            auto branch = if_else_stmt->at(i);
            auto& new_branch = this->builder_.add_case(new_scope, branch.second, branch.first.debug_info());
            this->node_mapping[&branch.first] = &new_branch;
            this->append(new_branch, branch.first);
        }
    } else if (auto loop_stmt = dynamic_cast<structured_control_flow::While*>(&source)) {
        auto& new_scope = this->builder_.add_while(root, {}, loop_stmt->debug_info());
        this->node_mapping[loop_stmt] = &new_scope;
        this->append(new_scope.root(), loop_stmt->root());
    } else if (auto for_stmt = dynamic_cast<structured_control_flow::For*>(&source)) {
        auto& new_scope = this->builder_.add_for(
            root,
            for_stmt->indvar(),
            for_stmt->condition(),
            for_stmt->init(),
            for_stmt->update(),
            {},
            for_stmt->debug_info()
        );
        this->node_mapping[for_stmt] = &new_scope;
        this->append(new_scope.root(), for_stmt->root());
    } else if (auto cont_stmt = dynamic_cast<structured_control_flow::Continue*>(&source)) {
        auto& new_cont = this->builder_.add_continue(root, {}, cont_stmt->debug_info());
        this->node_mapping[cont_stmt] = &new_cont;
    } else if (auto br_stmt = dynamic_cast<structured_control_flow::Break*>(&source)) {
        auto& new_br = this->builder_.add_break(root, {}, br_stmt->debug_info());
        this->node_mapping[br_stmt] = &new_br;
    } else if (auto ret_stmt = dynamic_cast<structured_control_flow::Return*>(&source)) {
        if (ret_stmt->is_data()) {
            auto& new_ret = this->builder_.add_return(root, ret_stmt->data(), {}, ret_stmt->debug_info());
            this->node_mapping[ret_stmt] = &new_ret;
        } else if (ret_stmt->is_constant()) {
            auto& new_ret =
                this->builder_.add_constant_return(root, ret_stmt->data(), ret_stmt->type(), {}, ret_stmt->debug_info());
            this->node_mapping[ret_stmt] = &new_ret;
        }
    } else if (auto map_stmt = dynamic_cast<structured_control_flow::Map*>(&source)) {
        auto& new_scope = this->builder_.add_map(
            root,
            map_stmt->indvar(),
            map_stmt->condition(),
            map_stmt->init(),
            map_stmt->update(),
            map_stmt->schedule_type(),
            {},
            map_stmt->debug_info()
        );
        this->node_mapping[map_stmt] = &new_scope;
        this->append(new_scope.root(), map_stmt->root());
    } else if (auto reduce_stmt = dynamic_cast<structured_control_flow::Reduce*>(&source)) {
        auto& new_scope = this->builder_.add_reduce(
            root,
            reduce_stmt->indvar(),
            reduce_stmt->condition(),
            reduce_stmt->init(),
            reduce_stmt->update(),
            reduce_stmt->reductions(),
            reduce_stmt->schedule_type(),
            {},
            reduce_stmt->debug_info()
        );
        this->node_mapping[reduce_stmt] = &new_scope;
        this->append(new_scope.root(), reduce_stmt->root());
    } else {
        throw std::runtime_error("Deep copy not implemented");
    }
};

StructuredSDFGDeepCopy::StructuredSDFGDeepCopy(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& root,
    structured_control_flow::ControlFlowNode& source
)
    : builder_(builder), root_(root), source_(source) {};

std::unordered_map<const structured_control_flow::ControlFlowNode*, const structured_control_flow::ControlFlowNode*>
StructuredSDFGDeepCopy::copy() {
    this->node_mapping.clear();
    this->insert(this->root_, this->source_);
    return this->node_mapping;
};

std::unordered_map<const structured_control_flow::ControlFlowNode*, const structured_control_flow::ControlFlowNode*>
StructuredSDFGDeepCopy::insert() {
    if (auto seq_source = dynamic_cast<structured_control_flow::Sequence*>(&this->source_)) {
        this->node_mapping.clear();
        this->append(this->root_, *seq_source);
        return this->node_mapping;
    } else {
        throw std::runtime_error("Source node must be a sequence");
    }
};

} // namespace deepcopy
} // namespace sdfg
