#pragma once

#include "sdfg/passes/expansion/lib_node_expander.h"

namespace sdfg::passes::expansion {

class AccessNodeExpansion;

class LibNodeExpansionContext : public LibNodeExpander::ExpandContext {
    friend AccessNodeExpansion;

    builder::StructuredSDFGBuilder& builder_;
    Sequence& parent_;
    size_t child_idx_;
    Block& block_;
    data_flow::LibraryNode& node_;
    const Options& options_;
    bool expanded_ = false;
    bool dropped_block_ = false;

public:
    LibNodeExpansionContext(
        builder::StructuredSDFGBuilder& builder,
        Sequence& parent,
        size_t child_idx,
        Block& block,
        data_flow::LibraryNode& node,
        const Options& options = Options::empty()
    )
        : builder_(builder), parent_(parent), child_idx_(child_idx), block_(block), node_(node), options_(options) {}

    std::unique_ptr<LibNodeExpander::AccessNodeExpand> replacement_requires_access_nodes(const std::vector<
                                                                                         LibNodeExpander::InputUse>&
                                                                                             access_dirs) override;

    LibNodeExpander::ExpandOutcome successfully_modified_node_only() override;

    LibNodeExpander::ExpandOutcome unable() override;
    LibNodeExpander::ExpandOutcome unapplicable() override;

    const Options& options() const override { return options_; }

    bool expanded() const { return expanded_; }

    bool dropped_block() const { return dropped_block_; }

    void cleanup();
};

} // namespace sdfg::passes::expansion
