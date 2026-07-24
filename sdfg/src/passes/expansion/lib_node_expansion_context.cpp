#include "lib_node_expansion_context.h"

namespace sdfg::passes::expansion {

class AccessNodeExpansion : public LibNodeExpander::AccessNodeExpand {
    LibNodeExpansionContext* context_;
    std::vector<const data_flow::AccessNode*> base_ins_;
    std::vector<const data_flow::AccessNode*> base_outs_;
    bool expansion_will_empty_block_;

public:
    AccessNodeExpansion(
        LibNodeExpansionContext* context,
        std::vector<const data_flow::AccessNode*> base_ins,
        std::vector<const data_flow::AccessNode*> base_outs,
        bool expansion_will_empty_block
    )
        : context_(context), base_ins_(base_ins), base_outs_(base_outs),
          expansion_will_empty_block_(expansion_will_empty_block) {
        context->expanded_ = true;
    }

    builder::StructuredSDFGBuilder& builder() override { return context_->builder_; }

    structured_control_flow::Sequence& replace_with_sequence() override {
        auto child_idx = context_->child_idx_;
        auto insertion_idx = child_idx + 1;
        auto& parent = context_->parent_;
        auto& block = context_->block_;

        return context_->builder_.add_sequence_at(parent, insertion_idx, block.debug_info());
    }

    structured_control_flow::StructuredLoop& replace_with_structured_loop(
        LoopType type,
        const symbolic::Symbol indvar,
        const symbolic::Condition condition,
        const symbolic::Expression init,
        const symbolic::Expression update,
        const ScheduleType& schedule_type
    ) override {
        auto child_idx = context_->child_idx_;
        auto insertion_idx = child_idx + 1;
        auto& parent = context_->parent_;
        auto& block = context_->block_;

        switch (type) {
            case LoopType::For:
                return context_->builder_
                    .add_for_at(parent, insertion_idx, indvar, condition, init, update, schedule_type, block.debug_info());
            case LoopType::Map:
                return context_->builder_
                    .add_map_at(parent, insertion_idx, indvar, condition, init, update, schedule_type, block.debug_info());
            default:
                throw std::runtime_error("Unsupported LoopType: " + std::to_string(static_cast<int>(type)));
        }
    }

    data_flow::AccessNode& add_scalar_input_access(structured_control_flow::Block& block, size_t input_idx) override {
        auto& org = base_ins_.at(input_idx);
        if (auto const_node = dynamic_cast<const data_flow::ConstantNode*>(org)) {
            return context_->builder_.add_constant(block, const_node->data(), const_node->type(), org->debug_info());
        } else {
            return context_->builder_.add_access(block, org->data(), org->debug_info());
        }
    }

    data_flow::AccessNode& add_indirect_read_access(structured_control_flow::Block& block, size_t input_idx) override {
        // for now, this is the same, we just want callers to use separate methods for future-proofing, debugging and
        // validation
        return add_scalar_input_access(block, input_idx);
    }

    data_flow::AccessNode& add_indirect_write_access(structured_control_flow::Block& block, size_t input_idx) override {
        // if it could be found as standalone access node, then we can just also create write accesses
        // it will only get complicated if we start supporting this on non-standalone inputs
        // (then this changes where we need to cut the dflow apart into [prev], replacement, [succ])
        // for now, this is the same, we just want callers to use separate methods for future-proofing, debugging and
        // validation
        return add_scalar_input_access(block, input_idx);
    }

    data_flow::AccessNode& add_output_access(structured_control_flow::Block& block, size_t output_idx) override {
        auto& org = base_outs_.at(output_idx);
        return context_->builder_.add_access(block, org->data(), org->debug_info());
    }

    LibNodeExpander::ExpandOutcome successfully_expanded() override { return LibNodeExpander::ExpandOutcome(true); }
};

void LibNodeExpansionContext::cleanup() {
    // for now, with a standalone requirement, this will be enough
    builder_.clear_code_node_legacy(block_, node_);
    if (block_.dataflow().nodes().empty()) {
        builder_.remove_child(parent_, child_idx_);
        this->dropped_block_ = true;
    }
}

std::unique_ptr<LibNodeExpander::AccessNodeExpand> LibNodeExpansionContext::
    replacement_requires_access_nodes(const std::vector<LibNodeExpander::InputUse>& access_dirs) {
    auto& dflow = block_.dataflow();

    auto& conns = node_.inputs();

    std::vector<const data_flow::AccessNode*> base_in_access_nodes(access_dirs.size());
    std::vector<const data_flow::AccessNode*> base_out_access_nodes;

    for (auto i = 0; i < access_dirs.size(); ++i) {
        auto& dir = access_dirs[i];

        if (dir != LibNodeExpander::InputUse::Skip) {
            auto& conn = conns[i];

            auto* edge = dflow.in_edge_for_connector(node_, conn);
            if (!edge) {
                return {}; // missing edge
            }
            auto* standalone = dflow.find_standalone_entry(edge);
            if (!standalone) {
                return {}; // for now, cutting is unsupported
            }

            base_in_access_nodes[i] = standalone;
        }
    }

    for (auto& conn : node_.outputs()) {
        auto edges = dflow.out_edges_for_connector(node_, conn);
        if (edges.size() != 1) {
            return {};
        }
        auto* edge = edges.at(0);
        auto* standalone = dflow.find_standalone_exit(edge);
        if (!standalone) {
            return {}; // for now, cutting is unsupported
        }
        base_out_access_nodes.push_back(standalone);
    }

    bool nothing_else_in_block = base_in_access_nodes.size() + base_out_access_nodes.size() + 1 == dflow.nodes().size();

    return std::make_unique<
        AccessNodeExpansion>(this, base_in_access_nodes, base_out_access_nodes, nothing_else_in_block);
}

LibNodeExpander::ExpandOutcome LibNodeExpansionContext::unable() { return LibNodeExpander::ExpandOutcome(false); }

LibNodeExpander::ExpandOutcome LibNodeExpansionContext::unapplicable() { return LibNodeExpander::ExpandOutcome(false); }

} // namespace sdfg::passes::expansion
