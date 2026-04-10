#pragma once

#include <unordered_set>

#include "sdfg/data_flow/library_nodes/math/blas/blas_node.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/return.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/structured_control_flow/while.h"

namespace sdfg {
namespace analysis {

enum class MergeMode {
    CUMULATIVE,
    BRANCHES,
};

template<typename T>
struct DataFlowState {
    using ExposedType = T;

    virtual ~DataFlowState() = default;

    virtual bool ran_at_least_once() const = 0;
    virtual bool update(const T& incoming) = 0;
    virtual const T& forward_exposed() const = 0;
    virtual bool update_incoming(const T& incoming) = 0;
    virtual bool update_forward_exposed(const T& forward_exposed) = 0;
};

typedef size_t ElementId;

template<typename T, typename I>
struct ElementIdMapDataFlowState : public DataFlowState<std::unordered_map<ElementId, T>> {
    using ExposedType = DataFlowState<std::unordered_map<ElementId, T>>::ExposedType;
    using InternalType = std::unordered_map<ElementId, std::unique_ptr<I>>;

protected:
    ExposedType incoming_;
    InternalType generated_;
    ExposedType forward_exposed_;
    bool ran_ = false;

public:
    bool ran_at_least_once() const override { return ran_; }

    bool update(const ExposedType& incoming) override {
        bool needs_update = update_incoming(incoming);

        if (!needs_update) {
            return false;
        }

        ExposedType exposed = incoming_;
        apply_kills_and_changes(exposed);
        for (auto& [id, gen] : generated_) {
            exposed.insert({id, expose(*gen)});
        }

        return update_forward_exposed(exposed);
    }

    virtual T expose(I& internal) = 0;

    bool update_incoming(const ExposedType& incoming) override {
        bool any_changes = false;
        for (auto it = incoming.begin(); it != incoming.end(); ++it) {
            auto [placed, changed] = incoming_.insert(*it);
            any_changes |= changed;
        }
        return any_changes || !ran_;
    }

    bool update_forward_exposed(const ExposedType& forward_exposed) override {
        bool any_changes = false;
        for (auto it = forward_exposed.begin(); it != forward_exposed.end(); ++it) {
            auto [placed, changed] = forward_exposed_.insert(*it);
            any_changes |= changed;
        }
        if (!ran_) {
            ran_ = true;
            return true;
        } else {
            return any_changes;
        }
    }

    const ExposedType& forward_exposed() const override { return forward_exposed_; }

    static ExposedType empty_in() { return ExposedType(); }

    static void merge(ExposedType& merge_into, const ExposedType& other, MergeMode mode = MergeMode::CUMULATIVE) {
        if (mode == MergeMode::BRANCHES) {
            for (auto it = merge_into.begin(); it != merge_into.end();) {
                auto key = it->first;
                if (!other.contains(key)) {
                    it = merge_into.erase(it);
                    continue;
                }
                ++it;
            }
        } else {
            merge_into.insert(other.begin(), other.end());
        }
    }

    virtual void apply_kills_and_changes(ExposedType& exposed) const = 0;
};

/**
 * @brief Base class for forward data-flow analysis over structured control flow.
 *
 * The analysis walks the structured CFG tree (Sequence, Block, IfElse,
 * StructuredLoop, While, Return, Break, Continue) in forward order and
 * propagates sets of instances of type `T` using the classic gen/kill
 * framework:
 *
 *     out(n) = gen(n) ∪ (in(n) − kill(n))
 *
 * Merge at join points (after IfElse branches, loop back-edges) is
 * performed by the virtual `merge` function which defaults to set union.
 *
 * **Usage:**
 *
 * 1. Derive from `ForwardDataFlowAnalysis<YourElementType>`.
 * 2. Override `transfer` to provide the gen/kill sets for leaf nodes.
 *    At a minimum you should handle `Block`; the other overloads have
 *    sensible defaults (empty gen/kill).
 * 3. Optionally override `merge` if you need intersection semantics
 *    (must-analysis) instead of the default union (may-analysis).
 * 4. Optionally override `boundary` to provide the initial set that
 *    flows into the root of the tree.
 * 5. Call `run(root_sequence)`.  Afterwards query `state(node)` for any
 *    visited node.
 *
 * @tparam T Element type.  Must support `operator<` (used in std::set).
 */
template<typename State>
class ForwardStructuredDataFlowAnalysis {
public:
    virtual ~ForwardStructuredDataFlowAnalysis() = default;

    using ExposedType = typename State::ExposedType;

    // -----------------------------------------------------------------
    //  Main entry point
    // -----------------------------------------------------------------

    struct SolveProgress {
        std::variant<
            structured_control_flow::ControlFlowNode*,
            structured_control_flow::StructuredLoop*,
            structured_control_flow::While*,
            structured_control_flow::IfElse*,
            structured_control_flow::Sequence*>
            node;
        int index = 0;
    };

    /**
     * @brief Run the analysis starting from the given root sequence.
     * @param root Typically `sdfg.root()`.
     */
    void run_forward(structured_control_flow::Sequence& root) { solve(root, boundary()); }

    // -----------------------------------------------------------------
    //  Query results
    // -----------------------------------------------------------------

    /**
     * @brief Retrieve the computed data-flow state for a node.
     * @param node A node that was visited during `run`.
     * @return Reference to its DataFlowState (in/out sets).
     */
    const State& state(const structured_control_flow::ControlFlowNode& node) const { return *states_.at(&node); }

    bool has_state(const structured_control_flow::ControlFlowNode& node) const { return states_.contains(&node); }

protected:
    State& get_or_create_state(const structured_control_flow::ControlFlowNode& node) {
        auto& ptr = states_[&node];
        if (!ptr) {
            ptr = create_initial_state(node);
        }
        return *ptr;
    }

    virtual std::unique_ptr<State> create_initial_state(const structured_control_flow::ControlFlowNode& node) = 0;
    // -----------------------------------------------------------------
    //  Merge at join points
    // -----------------------------------------------------------------

    /**
     * @brief Provide the initial set flowing into the root.
     *
     * Override to seed the analysis with an initial set (e.g. function
     * arguments that are live on entry).  Default: empty set.
     */
    virtual ExposedType boundary() { return State::empty_in(); }

private:
    std::unordered_map<const structured_control_flow::ControlFlowNode*, std::unique_ptr<State>> states_;

    // -----------------------------------------------------------------
    //  Recursive solvers for each node type
    // -----------------------------------------------------------------

    /**
     * @brief Solve a generic ControlFlowNode by dispatching on its
     *        dynamic type.
     * @param node  The node to solve.
     * @param in    The set flowing into this node.
     * @return      The set flowing out of this node.
     */
    std::pair<bool, const ExposedType&> solve(structured_control_flow::ControlFlowNode& node, const ExposedType& in) {
        // Dispatch to the concrete type
        if (auto* seq = dynamic_cast<structured_control_flow::Sequence*>(&node)) {
            return solve(*seq, in);
        }
        if (auto* if_else = dynamic_cast<structured_control_flow::IfElse*>(&node)) {
            return solve(*if_else, in);
        }
        if (auto* for_loop = dynamic_cast<structured_control_flow::For*>(&node)) {
            return solve_loop(*for_loop, in);
        }
        if (auto* map_loop = dynamic_cast<structured_control_flow::Map*>(&node)) {
            return solve_loop(*map_loop, in);
        }
        if (auto* while_loop = dynamic_cast<structured_control_flow::While*>(&node)) {
            return solve(*while_loop, in);
        }

        // Unknown node – treat as transparent
        auto& s = get_or_create_state(node);
        bool changed = s.update(in);
        return {changed, s.forward_exposed()};
    }

    // -- Sequence: propagate through children left-to-right ---------------

    std::pair<bool, const ExposedType&> solve(structured_control_flow::Sequence& seq, const ExposedType& in) {
        auto& seq_state = get_or_create_state(seq);
        seq_state.update_incoming(in);

        const ExposedType* current = &in;
        bool updating = true;
        for (size_t i = 0; i < seq.size() && updating; ++i) {
            auto [child, transition] = seq.at(i);
            auto [changed, output] = solve(child, *current);
            updating = changed;
            current = &output;
        }

        if (updating) {
            auto changed = seq_state.update_forward_exposed(*current);
            return {changed, *current};
        } else {
            return {false, seq_state.forward_exposed()};
        }
    }

    // -- IfElse: propagate into each branch, merge results ----------------

    std::pair<bool, const ExposedType&> solve(structured_control_flow::IfElse& if_else, const ExposedType& in) {
        auto& state = get_or_create_state(if_else);
        state.update_incoming(in);

        ExposedType merged = State::empty_in();
        bool first = true;
        bool changed = true;
        for (size_t i = 0; i < if_else.size(); ++i) {
            auto [branch_seq, cond] = if_else.at(i);
            auto [branch_changed, branch_out] = solve(branch_seq, in);
            changed |= branch_changed;
            State::merge(merged, branch_out, first ? MergeMode::CUMULATIVE : MergeMode::BRANCHES);
            first = false;
        }

        // If the IfElse is not complete (no else-branch covering all cases)
        // the input can also flow through unchanged.
        if (!if_else.is_complete()) {
            State::merge(merged, in, MergeMode::BRANCHES);
        }

        if (changed) {
            changed = state.update_forward_exposed(merged);
        }
        return {changed, state.forward_exposed()};
    }

    // -- StructuredLoop (For / Map): fixed-point iteration ----------------

    std::pair<bool, const ExposedType&> solve_loop(structured_control_flow::StructuredLoop& loop, const ExposedType& in) {
        // todo factor in loop header exposes

        return solve_loop_body(get_or_create_state(loop), loop.root(), in, [&](ExposedType& after_body) {
            // todo factor in loop update exposes
        });
    }

    std::pair<bool, const ExposedType&> solve_loop_body(
        State& state,
        structured_control_flow::Sequence& body,
        const ExposedType& in,
        std::function<void(ExposedType&)> apply_after_body
    ) {
        state.update_incoming(in);

        ExposedType current = in;
        bool changing = false;
        bool changed = false;
        do {
            auto [body_changed, body_out] = solve(body, current);
            changing = body_changed;
            changed |= body_changed;

            if (body_changed) {
                State::merge(current, body_out, MergeMode::BRANCHES);
            }

            if (apply_after_body) {
                apply_after_body(current);
            }
        } while (changing);

        if (changed) {
            changed = state.update_forward_exposed(current);
        }

        return {changed, state.forward_exposed()};
    }

    // -- While: same fixed-point strategy ---------------------------------

    std::pair<bool, const ExposedType&> solve(structured_control_flow::While& loop, const ExposedType& in) {
        return solve_loop_body(get_or_create_state(loop), loop.root(), in, nullptr);
    }
};

} // namespace analysis
} // namespace sdfg
