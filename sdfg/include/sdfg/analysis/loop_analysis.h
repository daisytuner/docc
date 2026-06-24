#pragma once

#include <unordered_set>
#include "sdfg/analysis/analysis.h"
#include "sdfg/structured_control_flow/structured_loop.h"

namespace sdfg {
namespace analysis {

class AssumptionsAnalysis;

struct DFSLoopComparator {
    const std::vector<structured_control_flow::ControlFlowNode*>* loops_order_;

    DFSLoopComparator(const std::vector<structured_control_flow::ControlFlowNode*>* loops_order)
        : loops_order_(loops_order) {}

    bool operator()(const structured_control_flow::ControlFlowNode* lhs, const structured_control_flow::ControlFlowNode* rhs)
        const {
        return std::find(loops_order_->begin(), loops_order_->end(), lhs) <
               std::find(loops_order_->begin(), loops_order_->end(), rhs);
    }
};

#define LOOP_INFO_PROPERTIES              \
    X(int, loopnest_index, -1)            \
    X(sdfg::ElementId, element_id, 0)     \
    X(size_t, num_loops, 0)               \
    X(size_t, num_maps, 0)                \
    X(size_t, num_fors, 0)                \
    X(size_t, num_whiles, 0)              \
    X(size_t, max_depth, 0)               \
    X(bool, is_perfectly_nested, false)   \
    X(bool, is_perfectly_parallel, false) \
    X(bool, is_elementwise, false)        \
    X(bool, has_side_effects, false)      \
    X(uint32_t, loop_level, 0)            \
    X(uint32_t, map_stack_depth, 0)

/// is_perfectly_nested: is the entire loop hierarchy with loop at the top perfectly nested on every level
/// is_perfectly_parallel: is the entire loop hierarchy with loop at the top perfectly parallel on every level

/// loop_level: nesting level of loop: 0 == outermost loop

/// map_stack_depth: how many layers deep from here (including the loop itself), do we have perfectly parallel & nested
///   i.e. a value of 3 here, means this loop and the next 2 levels of loops are all maps, all perfectly nested, but no
///   info on what's below that Mostly relevant in case there are children that are not perfectly parallel or nested,
///   but some of the parents are

struct LoopInfo {
#define X(type, name, val) type name = val;
    LOOP_INFO_PROPERTIES
#undef X
};

/**
 * Meant to contain only per-loop data (does not reflect what loops nested inside do / have)
 * This is a pre-processing step to collect the loop-nest-wide info.
 * It is a separate struct to not change the previous interface too much or pollute it with confusingly similar data
 * And to allow for separate allocation in the future
 */
struct LocalLoopInfo {
    /**
     * Unique ID for each loop, not just loop nests. Numbered in  pre-order DFS (children will always have higher
     * numbers than their parents)
     */
    uint32_t loop_id;
    /**
     * ID of the last child. Allows for efficient, transitive is_parent / is_child checks
     *   `parent.loop_id > child.loop_id && parent.last_child_id >= child.loop_id` => is (transitive) child
     */
    uint32_t last_child_id;
    uint32_t loop_level;
    bool is_map;
    bool is_elementwise;
    /**
     * The loop contains anything other than a direct child loop. Be it a block, an additional sequence or multiple
     * loops
     */
    bool contains_non_perfectly_nested;
    /**
     * The loop contains side-effects directly in its body (does not include anything that happens in child loops)
     */
    bool contains_side_effects;
};

typedef std::pair<LocalLoopInfo, LoopInfo> LoopState;

inline nlohmann::json loop_info_to_json(LoopInfo info) {
    nlohmann::json j = nlohmann::json{
        {"loopnest_index", info.loopnest_index},
        {"element_id", info.element_id},
        {"num_loops", info.num_loops},
        {"num_maps", info.num_maps},
        {"num_fors", info.num_fors},
        {"num_whiles", info.num_whiles},
        {"max_depth", info.max_depth},
        {"is_perfectly_nested", info.is_perfectly_nested},
        {"is_perfectly_parallel", info.is_perfectly_parallel},
        {"is_elementwise", info.is_elementwise},
        {"has_side_effects", info.has_side_effects},
        {"loop_level", info.loop_level},
        {"map_stack_depth", info.map_stack_depth}
    };
    return j;
}

/**
 * A perfectly nested, perfectly parallel loop-nest (only maps, 1 map per level, no instructions other then inside the
 * innermost map) Because you likely want to look at multiple dimensions at once to be more efficient. The stack ends at
 * the first loop that does not fit the criteria. So there might be further loops inside, but they would then require
 * more complex logic to handle as one
 */
struct MapStack {
    std::vector<structured_control_flow::ControlFlowNode*> outer_to_inner;
    /**
     * There are no more loops nested inside the innermost of the stack
     */
    bool innermost_is_leaf;
};

class LoopAnalysis : public Analysis {
private:
    uint32_t next_loop_id_ = 0;

    std::vector<structured_control_flow::ControlFlowNode*> loops_;
    std::map<structured_control_flow::ControlFlowNode*, structured_control_flow::ControlFlowNode*, DFSLoopComparator>
        loop_tree_;
    std::unordered_map<structured_control_flow::ControlFlowNode*, std::vector<structured_control_flow::ControlFlowNode*>>
        loop_children_;

    std::unordered_map<structured_control_flow::ControlFlowNode*, LoopState> loop_infos_;

    /**
     *
     * @param info reference to fill
     * @param loop_level current loop level
     * @param loop the loop for which this is the info
     * @param map if the loop is a map, also supply this one
     * @param while_loop
     */
    void init_new_loop_info(
        LoopState& info,
        uint32_t loop_level,
        structured_control_flow::ControlFlowNode* loop,
        structured_control_flow::Map* map,
        structured_control_flow::ControlFlowNode* while_loop
    );

    void
    run(structured_control_flow::ControlFlowNode& scope,
        structured_control_flow::ControlFlowNode* parent_loop,
        uint32_t loop_level);

    structured_control_flow::Sequence* get_loop_content_root(structured_control_flow::ControlFlowNode* loop);

    std::list<std::vector<sdfg::structured_control_flow::ControlFlowNode*>> loop_tree_paths(
        sdfg::structured_control_flow::ControlFlowNode* loop,
        const std::map<
            sdfg::structured_control_flow::ControlFlowNode*,
            sdfg::structured_control_flow::ControlFlowNode*,
            DFSLoopComparator>& tree
    ) const;

    LoopState& compute_loop_infos(structured_control_flow::ControlFlowNode* loop);

public:
    LoopAnalysis(StructuredSDFG& sdfg);

    std::string name() const override { return "LoopAnalysis"; }

    void run(analysis::AnalysisManager& analysis_manager) override;

    const std::vector<structured_control_flow::ControlFlowNode*> loops() const;

    const std::map<structured_control_flow::ControlFlowNode*, structured_control_flow::ControlFlowNode*, DFSLoopComparator>&
    loop_tree() const;

    LoopInfo loop_info(structured_control_flow::ControlFlowNode* loop) const;

    const LocalLoopInfo& loop_info_local(structured_control_flow::ControlFlowNode* loop) const;

    structured_control_flow::ControlFlowNode* find_loop_by_indvar(const std::string& indvar);

    structured_control_flow::ControlFlowNode* parent_loop(structured_control_flow::ControlFlowNode* loop) const;

    const std::vector<structured_control_flow::ControlFlowNode*>& outermost_loops() const;

    bool is_outermost_loop(structured_control_flow::ControlFlowNode* loop) const;

    const std::vector<structured_control_flow::ControlFlowNode*> outermost_maps() const;

    const std::vector<sdfg::structured_control_flow::ControlFlowNode*>&
    children(sdfg::structured_control_flow::ControlFlowNode* node) const;

    std::list<std::vector<sdfg::structured_control_flow::ControlFlowNode*>>
    loop_tree_paths(sdfg::structured_control_flow::ControlFlowNode* loop) const;

    std::unordered_set<sdfg::structured_control_flow::ControlFlowNode*>
    descendants(sdfg::structured_control_flow::ControlFlowNode* loop) const;

    void dump_to_file(std::filesystem::path file) const;
};

} // namespace analysis
} // namespace sdfg
