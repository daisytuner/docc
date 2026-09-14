/**
 * @file expansion_pass.h
 * @brief Library node expansion pass
 *
 * This file defines a peephole optimizer that will find all LibraryNodes with ImplementationType_NONE in the graph
 * and attempt to expand all of them into more baser operations.
 * It internally uses a list of LibNodeExpanders, so there could be multiple alternative variants to expand a given
 * node, but for now it simply redirects to MathNode.expand() to stay more compatible with existing code.
 *
 * The pass tries to isolate the contents of the expansion from the surrounding SDFG as best as possible,
 * handling finding access nodes or moving edges in a generic way. This makes expansions simpler and allows us to
 * upgrade the passes' handling in the future, for example to cut Dataflow into multiple parts to allow for expansion.
 *
 * To achieve this, every Expander is given a context and the block to do preliminary checks, whether it can /
 * wants to expand this node with its properties. Access to src and dest edges may be neccessary,
 * as some LibraryNodes currently infer their data types from other nodes. Access to src and dest nodes however should
 * be avoided, as the pass may change or remove them.
 *
 * In case, an Expander wants to proceed with the expansion, it needs to call one of the available methods on the
 * ExpansionContext:
 *  * replacement_requires_access_nodes()
 *    * The classic way, where every input and output edge is going to be replaced with AccessNodes.
 *    This way, the inputs might be accessed inside newly created loops etc.
 *  * replace_dataflow()
 *    * A simpler way, where the replacement will happen inside the existing DataFlowGraph. In this case,
 *    no access nodes will be touched. The Expander will create its new nodes and define where which original input and
 * output edge will go The generic expansion logic will then check if this is possible or the graph can be changed to
 * make this possible (future improvements here) It might return nullptr, if it does not know how to isolate the
 * LibraryNode Otherwise, if the replacement was started, it must be finished successfully to not leave the SDFG in a
 * possibly broken state.
 */

#pragma once

#include "sdfg/data_flow/library_nodes/math/math_node.h"
#include "sdfg/passes/expansion/lib_node_expander.h"
#include "sdfg/passes/expansion/lib_node_expansion_context.h"
#include "sdfg/passes/pass.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {

class MathNodeExpander : public TypedLibNodeExpander<math::MathNode> {
public:
    LibNodeExpander::ExpandOutcome handle_expand(ExpandContext& context, Block& block, math::MathNode& node)
        const override;
};

namespace expansion {
struct BlockOutcome {
    bool block_removed;
    int skip_count;

    BlockOutcome(bool block_removed = false, int skip_count = 0)
        : block_removed(block_removed), skip_count(skip_count) {}
};

struct NodeOutcome : public BlockOutcome {
    bool expanded;

    NodeOutcome(bool expanded = false, bool block_removed = false, int skip_count = 0)
        : BlockOutcome(block_removed, skip_count), expanded(expanded) {}
};
} // namespace expansion

struct SingleExpanderHolder {
    using StorageType = std::shared_ptr<LibNodeExpander>;
    const LibNodeExpander& expander_;

    SingleExpanderHolder(const LibNodeExpander& expander) : expander_(expander) {}
    SingleExpanderHolder(StorageType expander) : expander_(*expander) {}

    const LibNodeExpander* get_expander_for(const data_flow::LibraryNode& node) const {
        return expander_.for_lib_node(node);
    }
};

template<typename Holder>
concept HolderPolicy = requires(Holder& h, const data_flow::LibraryNode& node) { h.get_expander_for(node); };

template<HolderPolicy Holder = SingleExpanderHolder>
class LibraryNodeExpansionPass;

template<HolderPolicy Holder = SingleExpanderHolder>
class LibNodeExpansionVisitor : public visitor::ActualStructuredSDFGVisitor {
    friend LibraryNodeExpansionPass<Holder>;

public:

private:
    builder::StructuredSDFGBuilder& builder_;
    analysis::AnalysisManager& analysis_manager_;
    bool expanded_any_ = false;
    Holder holder_;
    bool force_expand_;


    expansion::NodeOutcome try_expand(Sequence& parent, size_t child_idx, Block& block, sdfg::data_flow::LibraryNode& node) {
        auto expander = holder_.get_expander_for(node);

        if (expander) {
            expansion::LibNodeExpansionContext
                ctx(this->builder_, parent, child_idx, block, node, analysis_manager_.options());

            auto outcome = expander->handle_expand(ctx, block, node);

            if (ctx.expanded()) {
                ctx.cleanup();
                return {true, ctx.dropped_block()};
            }
        }

        return {};
    }

public:
    /**
     * @brief Construct the expansion visitor
     * @param builder SDFG builder for creating new nodes
     * @param analysis_manager Analysis manager for querying properties
     */
    LibNodeExpansionVisitor(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        Holder holder,
        bool force_expand
    )
        : visitor::ActualStructuredSDFGVisitor(), builder_(builder), analysis_manager_(analysis_manager),
          holder_(holder), force_expand_(force_expand) {}

    bool visit(sdfg::structured_control_flow::Sequence& seq) override {
        bool may_contain_libnodes = true;
        size_t i = 0;

        // expansion can remove and replace entire blocks, so need smarter sequence visiting that can keep iterating
        // even when the children change expansion can only remove the current node and add 1 or more children in its
        // stead, so idx will never decrease but we do need the info what was changed to avoid invalidated parts

        do {
            size_t total_children = seq.size();
            if (total_children == 0) {
                may_contain_libnodes = false;
            } else if (i >= total_children) {
                return true;
            }

            for (; i < total_children; ++i) {
                auto& child = seq.at(i);
                if (auto* block = dyn_cast<structured_control_flow::Block*>(&child)) {
                    auto outcome = handle_block(seq, i, *block);

                    if (outcome.block_removed) { // recheck with same i, its now a new block
                        if (outcome.skip_count) {
                            i += outcome.skip_count;
                        }
                        break;
                    } else if (outcome.skip_count) {
                        i += outcome.skip_count;
                        // +1 from for loop
                    }
                } else {
                    dispatch(child);
                }
            }
        } while (may_contain_libnodes);

        return true;
    }

    bool handleStructuredLoop(sdfg::structured_control_flow::StructuredLoop& loop) override {
        auto& schedType = loop.schedule_type();
        int scope_id = -1;
        if constexpr (requires { holder_.scope_enter(schedType); }) {
            scope_id = holder_.scope_enter(schedType);
        }

        auto res = ActualStructuredSDFGVisitor::handleStructuredLoop(loop);

        if constexpr (requires { holder_.scope_exit(schedType); }) {
            holder_.scope_exit(scope_id);
        }

        return res;
    }

    expansion::BlockOutcome handle_block(
        structured_control_flow::Sequence& parent, size_t child_idx, sdfg::structured_control_flow::Block& block
    ) {
        auto& dataflow = block.dataflow();

        ElementId last_element_id = 0;

        bool may_contain_lib_nodes = true;
        bool handled_any = false;

        const bool force_expand = force_expand_;

        do {
            // expansion may change the contents of this block or even remove it.
            // to ensure stable order, order by element_id
            // track the last handled element_id, because changes are not allowed  to affect other libnodes.
            // So if the current block gets invalidated (but not removed) we can restart iterating above the last
            // processed element_id
            auto libnodes =
                dataflow.nodes() |
                std::views::transform([](auto& n) { return dynamic_cast<const data_flow::LibraryNode*>(&n); }) |
                std::views::filter([](auto* n) { return n != nullptr; }) |
                std::views::filter([last_element_id, force_expand](auto* n) {
                    return (force_expand || n->implementation_type() == data_flow::ImplementationType_NONE) &&
                           n->element_id() > last_element_id;
                });
            std::vector<const data_flow::LibraryNode*> sorted_nodes(libnodes.begin(), libnodes.end());
            std::ranges::sort(sorted_nodes, std::less<>{}, [](const auto* n) { return n->element_id(); });

            may_contain_lib_nodes = !libnodes.empty();

            bool block_changed = false;
            for (auto* library_node : sorted_nodes) {
                last_element_id = library_node->element_id();
                auto outcome = try_expand(parent, child_idx, block, *const_cast<data_flow::LibraryNode*>(library_node));
                handled_any |= outcome.expanded;

                if (outcome.block_removed) {
                    this->expanded_any_ = true;
                    return static_cast<expansion::BlockOutcome>(outcome);
                } else if (outcome.expanded) {
                    // libnodes inside this block have become invalid
                    block_changed = true;
                    break;
                }
            }
            if (block_changed) { // here as a reminder, that we must not do anything else in this iteration, as the
                                 // underlying nodes might be invalid now
                continue;
            }
        } while (may_contain_lib_nodes);

        if (handled_any) {
            this->expanded_any_ = true;
        }

        return {};
    }
};

/**
 * @class LibraryNodeExpansionPass
 * @brief Looks for and expands library nodes in a single pass, potentially recursively
 */
template<HolderPolicy Holder>
class LibraryNodeExpansionPass : public Pass {
    Holder::StorageType holder_options_;

public:
    static constexpr std::string_view NAME{"library_node_expansion"};
    static constexpr OptionKey<bool> FORCE_EXPAND{"library_node_expansion.force_expand"};
    static constexpr OptionKey<bool> ONLINE_SOFTMAX{"library_node_expansion.online_softmax"};

    LibraryNodeExpansionPass() : holder_options_(std::make_shared<MathNodeExpander>()) {}
    explicit LibraryNodeExpansionPass(const Options& options)
        : Pass(options), holder_options_(std::make_shared<MathNodeExpander>()) {}
    LibraryNodeExpansionPass(Holder::StorageType options) : holder_options_(std::move(options)) {}

    std::string name() override { return std::string(NAME); }

    std::vector<OptionSpec> options() override {
        return {
            FORCE_EXPAND.spec(false, "Also lower all library nodes that already have an implementation type"),
            ONLINE_SOFTMAX.spec(
                false,
                "Expand softmax as the single-pass online (log-sum-exp monoid) form instead of the default "
                "three-pass (max; exp+sum; normalize) form"
            )
        };
    }

    bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override {
        LibNodeExpansionVisitor<Holder>
            v(builder, analysis_manager, Holder(holder_options_), this->option(FORCE_EXPAND));

        v.dispatch(builder.subject().root());

        return v.expanded_any_;
    }
};

// Deduction guide
template<typename T>
LibraryNodeExpansionPass(T) -> LibraryNodeExpansionPass<T>;

namespace expansion {

expansion::NodeOutcome expand_single_node(
    builder::StructuredSDFGBuilder& builder, Block& block, data_flow::LibraryNode& node, const LibNodeExpander& expanders
);

expansion::NodeOutcome
expand_single_math_node(builder::StructuredSDFGBuilder& builder, Block& block, data_flow::LibraryNode& node);

} // namespace expansion

} // namespace passes
} // namespace sdfg
