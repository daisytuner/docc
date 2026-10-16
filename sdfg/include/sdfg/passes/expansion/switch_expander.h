#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sdfg/data_flow/library_node.h"
#include "sdfg/passes/expansion/lib_node_expander.h"

namespace sdfg::passes {

/**
 * @brief An expander that bundles several sub-expanders and switches to the correct one per node.
 *
 * The @ref LibNodeExpander interface already supports "tested" expanders: every expander exposes
 * @ref LibNodeExpander::for_lib_node which returns the concrete expander to use for a given node, or
 * @c nullptr if the expander does not apply.
 *
 * Most expanders dispatch on a node's @ref data_flow::LibraryNodeCode (a string match). This class
 * exploits that: code-based children advertise their code via @ref LibNodeExpander::matched_code, so
 * the switch builds a @c code -> @c expander hashmap once and then jumps straight to the right
 * expander in O(1) per node, instead of testing every child linearly.
 *
 *  * @ref for_lib_node looks the node's code up in the map and returns the matching child (giving the
 *    child a final say via its own @ref LibNodeExpander::for_lib_node so it can refine further). Any
 *    child that does not dispatch purely by code is kept in a small linear fallback list.
 *  * If nothing applies, @c nullptr is returned and the node is left untouched by the pass.
 *
 * Expanders that handle an entire subclass of nodes (and therefore several codes) can be wired into
 * the fast path explicitly via @ref register_for_code, registering the same expander under each code
 * it is responsible for.
 *
 * Because it is itself a @ref LibNodeExpander, a @ref SwitchExpander plugs straight into the existing
 * @ref SingleExpanderHolder / @ref LibraryNodeExpansionPass machinery without any changes:
 *
 * @code
 * auto expander = std::make_shared<SwitchExpander>(
 *     std::make_shared<omp::OMPConvExpander>(),
 *     std::make_shared<math::tensor::MatMulBlasExpander>()
 * );
 * // optionally wire an expander that covers a whole node subclass under specific codes:
 * expander->register_for_code(SomeLibraryNodeCode, std::make_shared<SomeSubclassExpander>());
 *
 * LibraryNodeExpansionPass pass(expander);
 * pass.run(builder, analysis_manager);
 * @endcode
 */
class SwitchExpander : public LibNodeExpander {
    /// Owns the registered expanders.
    std::vector<std::shared_ptr<const LibNodeExpander>> expanders_;
    /// Fast lookup by library node code (string) for code-dispatching expanders.
    std::unordered_map<std::string, const LibNodeExpander*> by_code_;
    /// Expanders that do not dispatch purely by code, tested linearly as a fallback.
    std::vector<const LibNodeExpander*> fallback_;

public:
    /**
     * @brief Construct from a variadic list of expanders.
     * @param expanders one or more @c shared_ptr to @ref LibNodeExpander. Code-dispatching expanders
     *        (e.g. @ref CodeLibNodeExpander subclasses) are indexed by their code for O(1) lookup; any
     *        others are consulted linearly as a fallback.
     */
    template<typename... Expanders>
    explicit SwitchExpander(Expanders... expanders) {
        expanders_.reserve(sizeof...(Expanders));
        (register_expander(std::move(expanders)), ...);
    }

    /// Construct from an already assembled list of expanders.
    explicit SwitchExpander(std::vector<std::shared_ptr<const LibNodeExpander>> expanders) {
        expanders_.reserve(expanders.size());
        for (auto& expander : expanders) {
            register_expander(std::move(expander));
        }
    }

    /**
     * @brief Register @p expander.
     * If it dispatches on a single code, it is added to the fast lookup map via @ref SwitchExpander::register_for_code
     * otherwise it will be matched less efficiently
     */
    void register_expander(std::shared_ptr<const LibNodeExpander> expander) {
        if (!expander) {
            return;
        }
        if (const auto* code = expander->matched_code()) {
            register_for_code(*code, std::move(expander));
        } else {
            fallback_.push_back(expander.get());
            expanders_.push_back(std::move(expander));
        }
    }

    /**
     * @brief Register @p expander as the handler for library nodes with the given @p code.
     *
     * @param code the library node code to route to @p expander.
     * @param expander the expander to invoke for nodes with that code.
     * @throws std::invalid_argument if another expander is already registered for @p code.
     */
    void register_for_code(const data_flow::LibraryNodeCode& code, std::shared_ptr<const LibNodeExpander> expander) {
        if (!expander) {
            return;
        }
        auto [it, inserted] = by_code_.emplace(code.value(), expander.get());
        if (!inserted) {
            throw std::invalid_argument(
                "SwitchExpander: more than one expander registered for library node code '" + code.value() + "'"
            );
        }
        expanders_.push_back(std::move(expander));
    }

    /**
     * @brief Find the expander that applies to @p node via an O(1) code lookup, with a linear fallback.
     * @return the matching concrete expander, or @c nullptr if none applies.
     */
    const LibNodeExpander* for_lib_node(const data_flow::LibraryNode& node) const override {
        if (auto it = by_code_.find(node.code().value()); it != by_code_.end()) {
            if (const auto* match = it->second->for_lib_node(node)) {
                return match;
            }
        }
        for (const auto* expander : fallback_) {
            if (const auto* match = expander->for_lib_node(node)) {
                return match;
            }
        }
        return nullptr;
    }

    /**
     * @brief Delegate expansion to the matching sub-expander.
     *
     * In the normal pass flow this is not reached, because @ref for_lib_node already hands back the
     * concrete child expander and the pass calls @c handle_expand on that child directly. It is
     * implemented for completeness and for callers that drive this expander manually.
     */
    ExpandOutcome handle_expand(
        ExpandContext& context, structured_control_flow::Block& block, data_flow::LibraryNode& node
    ) const override {
        if (const auto* expander = for_lib_node(node); expander && expander != this) {
            return expander->handle_expand(context, block, node);
        }
        return context.unapplicable();
    }
};

} // namespace sdfg::passes
