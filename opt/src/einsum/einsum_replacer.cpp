#include "sdfg/einsum/einsum_replacer.h"

#include <cstddef>
#include <stdexcept>

#include "sdfg/data_flow/access_node.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/sequence.h"

namespace sdfg::einsum {

class EinsumAccessNodeExpand;

class EinsumReplacementContextImpl : public EinsumReplacementContext {
    friend class EinsumAccessNodeExpand;

    builder::StructuredSDFGBuilder& builder_;
    const EinsumCluster& cluster_;

    structured_control_flow::ControlFlowNode* site_;
    structured_control_flow::Sequence* parent_;
    size_t child_idx_;

    bool replaced_ = false;
    bool dropped_block_ = false;

public:
    EinsumReplacementContextImpl(builder::StructuredSDFGBuilder& builder, const EinsumCluster& cluster);

    std::unique_ptr<passes::LibNodeExpander::AccessNodeExpand> replacement_requires_access_nodes(
        const std::vector<passes::LibNodeExpander::InputUse>& access_dirs, bool leave_unconsumed_indices
    ) override;

    ReplaceOutcome successfully_modified_node_only() override;
    ReplaceOutcome unable() override;
    ReplaceOutcome unapplicable() override;

    bool replaced() const { return replaced_; }
    bool dropped_block() const { return dropped_block_; }

    /// Remove the original einsum computation (the consumed loops or the reduction core).
    void cleanup();
};

/// AccessNodeExpand implementation sourcing its base access nodes from an EinsumCluster.
class EinsumAccessNodeExpand : public passes::LibNodeExpander::AccessNodeExpand {
    EinsumReplacementContextImpl* context_;
    std::vector<const data_flow::AccessNode*> base_ins_;
    std::vector<const data_flow::AccessNode*> base_outs_;
    bool impl_handles_unconsumed_;

public:
    EinsumAccessNodeExpand(
        EinsumReplacementContextImpl* context,
        std::vector<const data_flow::AccessNode*> base_ins,
        std::vector<const data_flow::AccessNode*> base_outs,
        bool impl_handles_unconsumed
    )
        : context_(context), base_ins_(std::move(base_ins)), base_outs_(std::move(base_outs)),
          impl_handles_unconsumed_(impl_handles_unconsumed) {
        context_->replaced_ = true;

        if (!impl_handles_unconsumed) {
            throw std::invalid_argument("EinsumReplace: impl_handles_unconsumed=false is not supported yet");
        }
    }

    builder::StructuredSDFGBuilder& builder() override { return context_->builder_; }

    structured_control_flow::Sequence& replace_with_sequence() override {
        return context_->builder_
            .add_sequence_at(*context_->parent_, context_->child_idx_ + 1, context_->site_->debug_info());
    }

    structured_control_flow::StructuredLoop& replace_with_structured_loop(
        LoopType type,
        const symbolic::Symbol indvar,
        const symbolic::Condition condition,
        const symbolic::Expression init,
        const symbolic::Expression update,
        const ScheduleType& schedule_type
    ) override {
        auto insertion_idx = context_->child_idx_ + 1;
        auto& parent = *context_->parent_;
        auto debug_info = context_->site_->debug_info();
        switch (type) {
            case LoopType::For:
                return context_->builder_
                    .add_for_at(parent, insertion_idx, indvar, condition, init, update, schedule_type, debug_info);
            case LoopType::Map:
                return context_->builder_
                    .add_map_at(parent, insertion_idx, indvar, condition, init, update, schedule_type, debug_info);
            default:
                throw std::runtime_error("Unsupported LoopType: " + std::to_string(static_cast<int>(type)));
        }
    }

    data_flow::AccessNode& add_scalar_input_access(structured_control_flow::Block& block, size_t input_idx) override {
        auto* org = base_ins_.at(input_idx);
        if (auto* const_node = dynamic_cast<const data_flow::ConstantNode*>(org)) {
            return context_->builder_.add_constant(block, const_node->data(), const_node->type(), org->debug_info());
        }
        return context_->builder_.add_access(block, org->data(), org->debug_info());
    }

    data_flow::AccessNode& add_indirect_read_access(structured_control_flow::Block& block, size_t input_idx) override {
        return add_scalar_input_access(block, input_idx);
    }

    data_flow::AccessNode& add_indirect_write_access(structured_control_flow::Block& block, size_t input_idx) override {
        return add_scalar_input_access(block, input_idx);
    }

    data_flow::AccessNode& add_output_access(structured_control_flow::Block& block, size_t output_idx) override {
        auto* org = base_outs_.at(output_idx);
        return context_->builder_.add_access(block, org->data(), org->debug_info());
    }

    passes::LibNodeExpander::ExpandOutcome successfully_expanded() override {
        return passes::LibNodeExpander::ExpandOutcome(true);
    }
};

EinsumReplacementContextImpl::
    EinsumReplacementContextImpl(builder::StructuredSDFGBuilder& builder, const EinsumCluster& cluster)
    : builder_(builder), cluster_(cluster) {
    if (cluster.consumed_loops.empty()) {
        site_ = cluster.block;
    } else {
        site_ = cluster.consumed_loops.front();
    }
    parent_ = dynamic_cast<structured_control_flow::Sequence*>(site_->get_parent());
    child_idx_ = parent_ ? static_cast<size_t>(parent_->index(*site_)) : 0;
}

std::unique_ptr<passes::LibNodeExpander::AccessNodeExpand> EinsumReplacementContextImpl::replacement_requires_access_nodes(
    const std::vector<passes::LibNodeExpander::InputUse>& access_dirs, bool leave_unconsumed_indices
) {
    if (!parent_) {
        return {};
    }

    std::vector<const data_flow::AccessNode*> base_ins(access_dirs.size(), nullptr);
    for (size_t i = 0; i < access_dirs.size(); ++i) {
        if (access_dirs[i] == passes::LibNodeExpander::InputUse::Skip) {
            continue;
        }
        if (i >= cluster_.inputs.size()) {
            return {};
        }
        auto it = cluster_.input_nodes.find(cluster_.inputs[i]);
        if (it == cluster_.input_nodes.end()) {
            return {};
        }
        auto* access = dynamic_cast<const data_flow::AccessNode*>(it->second);
        if (!access) {
            return {};
        }
        base_ins[i] = access;
    }

    std::vector<const data_flow::AccessNode*> base_outs;
    if (cluster_.output_node) {
        base_outs.push_back(cluster_.output_node);
    }

    return std::make_unique<
        EinsumAccessNodeExpand>(this, std::move(base_ins), std::move(base_outs), leave_unconsumed_indices);
}

ReplaceOutcome EinsumReplacementContextImpl::successfully_modified_node_only() { return ReplaceOutcome(true); }

ReplaceOutcome EinsumReplacementContextImpl::unable() { return ReplaceOutcome(false); }

ReplaceOutcome EinsumReplacementContextImpl::unapplicable() { return ReplaceOutcome(false); }

void EinsumReplacementContextImpl::cleanup() {
    if (!parent_) {
        return;
    }

    // With consumed loops, the outermost loop subtree is exactly the einsum computation.
    if (!cluster_.consumed_loops.empty()) {
        builder_.remove_child(*parent_, static_cast<size_t>(parent_->index(*site_)));
        dropped_block_ = true;
        return;
    }

    // Otherwise remove the reduction core nodes from the block, dropping the block if it empties.
    auto& block = *cluster_.block;
    auto& dfg = block.dataflow();
    for (auto* node : cluster_.consumed_nodes) {
        while (dfg.in_edges(*node).begin() != dfg.in_edges(*node).end()) {
            builder_.remove_memlet(block, *dfg.in_edges(*node).begin());
        }
        while (dfg.out_edges(*node).begin() != dfg.out_edges(*node).end()) {
            builder_.remove_memlet(block, *dfg.out_edges(*node).begin());
        }
        builder_.remove_node(block, *node);
    }
    if (dfg.nodes().empty()) {
        builder_.remove_child(*parent_, static_cast<size_t>(parent_->index(block)));
        dropped_block_ = true;
    }
}

ReplaceOutcome replace_einsum_cluster(
    builder::StructuredSDFGBuilder& builder, const EinsumCluster& cluster, const EinsumReplacer& replacer
) {
    EinsumReplacementContextImpl context(builder, cluster);
    auto outcome = replacer.replace(context, cluster);
    if (context.replaced()) {
        context.cleanup();
    }
    return outcome;
}

} // namespace sdfg::einsum
