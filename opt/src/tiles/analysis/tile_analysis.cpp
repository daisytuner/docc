#include "sdfg/tiles/analysis/tile_analysis.h"

#include <functional>
#include <optional>
#include <unordered_set>
#include <utility>

#include "sdfg/analysis/base_user_visitor.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/analysis/memory_layout_analysis.h"
#include "sdfg/analysis/pointer_analyzers.h"
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace tiles {

using structured_control_flow::ControlFlowNode;
using structured_control_flow::StructuredLoop;

TileAnalysis::TileAnalysis(StructuredSDFG& sdfg) : analysis::Analysis(sdfg) {}

namespace {

/// Escape/overwrite/read/write policy for a single container, fed by the shared
/// pointer analyzers.
struct ContainerAccessPolicy {
    std::string container;
    bool reads = false;
    bool writes = false;
    bool aliased = false; ///< escaped, overwritten, or captured

    void on_escape(const std::string& c, const structured_control_flow::ControlFlowNode*, const Element*) {
        if (c == container) aliased = true;
    }
    void on_overwrite(const std::string& c, const structured_control_flow::ControlFlowNode*, const Element*) {
        if (c == container) aliased = true;
    }
    void on_read_via(const std::string& c, const structured_control_flow::ControlFlowNode*, const data_flow::Memlet*) {
        if (c == container) reads = true;
    }
    void on_write_via(const std::string& c, const structured_control_flow::ControlFlowNode*, const data_flow::Memlet*) {
        if (c == container) writes = true;
    }
};

/// Composes the shared PointerEscape/Overwrite/Used analyzers over a subtree,
/// mirroring MemoryOwnershipAnalysis. Adds one refinement DataDependencyAnalysis
/// carries but the analyzers do not: a library node consuming the pointer with a
/// missing or non-`no_capture` `pointer_access_type` is treated as aliasing.
class ContainerAccessVisitor : public analysis::BaseUserVisitor,
                               public analysis::PointerEscapeAnalyzer<ContainerAccessPolicy>,
                               public analysis::PointerOverwriteAnalyzer<ContainerAccessPolicy>,
                               public analysis::PointerUsedAnalyzer<ContainerAccessPolicy> {
    ContainerAccessPolicy& policy_;

    void capture_check(const data_flow::Memlet& edge, const data_flow::DataFlowNode& other) {
        if (auto* lib = dynamic_cast<const data_flow::LibraryNode*>(&other)) {
            auto access = lib->pointer_access_type(edge);
            if (!access || !access->no_capture()) {
                policy_.aliased = true;
            }
        }
    }

public:
    ContainerAccessVisitor(const StructuredSDFG& sdfg, ContainerAccessPolicy& policy)
        : analysis::PointerEscapeAnalyzer<ContainerAccessPolicy>(sdfg, policy),
          analysis::PointerOverwriteAnalyzer<ContainerAccessPolicy>(sdfg, policy),
          analysis::PointerUsedAnalyzer<ContainerAccessPolicy>(sdfg, policy), policy_(policy) {}

    void use_as_src_node(
        const std::string& c,
        const data_flow::AccessNode& n,
        const data_flow::Memlet& e,
        const structured_control_flow::Block& b
    ) override {
        analysis::PointerEscapeAnalyzer<ContainerAccessPolicy>::use_as_src_node(c, n, e, b);
        analysis::PointerUsedAnalyzer<ContainerAccessPolicy>::use_as_src_node(c, n, e, b);
        if (c == policy_.container) capture_check(e, e.dst());
    }
    void use_as_dst_node(
        const std::string& c,
        const data_flow::AccessNode& n,
        const data_flow::Memlet& e,
        const structured_control_flow::Block& b
    ) override {
        analysis::PointerOverwriteAnalyzer<ContainerAccessPolicy>::use_as_dst_node(c, n, e, b);
        analysis::PointerUsedAnalyzer<ContainerAccessPolicy>::use_as_dst_node(c, n, e, b);
        if (c == policy_.container) capture_check(e, e.src());
    }
    void use_as_return_src(const std::string& c, const structured_control_flow::Return& r) override {
        analysis::PointerEscapeAnalyzer<ContainerAccessPolicy>::use_as_return_src(c, r);
    }
    void use_as_symbol_read(
        const std::string& c,
        const structured_control_flow::ControlFlowNode* n,
        const Element* u,
        SymbolReadLocation loc,
        int loc_index,
        symbolic::Expression expr
    ) override {
        analysis::PointerEscapeAnalyzer<ContainerAccessPolicy>::use_as_symbol_read(c, n, u, loc, loc_index, expr);
    }
    void use_as_symbol_write(
        const symbolic::Symbol& c,
        const structured_control_flow::ControlFlowNode* n,
        const Element* u,
        SymbolWriteLocation loc
    ) override {
        analysis::PointerOverwriteAnalyzer<ContainerAccessPolicy>::use_as_symbol_write(c, n, u, loc);
    }
};

std::optional<tiles::Tile> build_tile(
    const StructuredSDFG& sdfg, StructuredLoop& loop, const std::string& container, const analysis::MemoryTile& mt
) {
    // Source layout: tile-local coordinate -> global element. The over-approximated
    // extents must be constant-bounded (non-null) to form a Layout.
    auto extents = mt.extents_approx();
    if (extents.empty()) {
        return std::nullopt;
    }
    for (const auto& e : extents) {
        if (e.is_null()) {
            return std::nullopt;
        }
    }
    const analysis::MemoryLayout& ml = mt.layout;
    if (ml.strides().size() != extents.size() || mt.min_subset.size() != extents.size()) {
        return std::nullopt;
    }
    // Fold the tile base into the offset: g(local) = off + sum_d stride_d * local_d,
    // off = ml.offset + sum_d stride_d * min_d.
    symbolic::Expression off = ml.offset();
    for (size_t d = 0; d < extents.size(); ++d) {
        off = symbolic::add(off, symbolic::mul(ml.strides()[d], mt.min_subset[d]));
    }
    math::tensor::TensorLayout tl(extents, ml.strides(), off);

    auto axes = tiles::TileAxis::enclosing(loop, mt.min_subset);
    const auto summary = TileAnalysis::summarize(sdfg, loop, container);
    return tiles::Tile(container, tiles::Layout::from_tensor(tl), std::move(axes), summary.reads, summary.writes);
}

} // namespace

TileAnalysis::AccessSummary TileAnalysis::
    summarize(const StructuredSDFG& sdfg, StructuredLoop& loop, const std::string& container) {
    ContainerAccessPolicy policy;
    policy.container = container;
    ContainerAccessVisitor visitor(sdfg, policy);
    visitor.visit(loop.root()); // walks the loop body only
    return AccessSummary{policy.reads, policy.writes, policy.aliased};
}

void TileAnalysis::run(analysis::AnalysisManager& analysis_manager) {
    tiles_.clear();
    auto& mla = analysis_manager.get<analysis::MemoryLayoutAnalysis>();
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();

    for (auto* node : loop_analysis.loops()) {
        auto* loop = dynamic_cast<StructuredLoop*>(node);
        if (loop == nullptr) {
            continue;
        }
        for (const auto& container : sdfg_.containers()) {
            const analysis::MemoryTile* mt = mla.tile(*loop, container);
            if (mt == nullptr) {
                continue;
            }
            auto tile = build_tile(sdfg_, *loop, container, *mt);
            if (tile) {
                tiles_.emplace(std::make_pair(node, container), std::move(*tile));
            }
        }
    }
}

const tiles::Tile* TileAnalysis::tile(const ControlFlowNode& scope, const std::string& container) const {
    auto it = tiles_.find(std::make_pair(&scope, container));
    return it == tiles_.end() ? nullptr : &it->second;
}

bool is_constant_bounded(const analysis::MemoryTileGroup* group) {
    if (!group) {
        return false;
    }
    auto extents = group->tile.extents_approx();
    if (extents.empty()) {
        return false;
    }
    for (auto& extent : extents) {
        if (extent.is_null() || !SymEngine::is_a<SymEngine::Integer>(*extent)) {
            return false;
        }
    }
    return true;
}

const std::vector<analysis::MemoryTileGroup>* tile_groups(
    structured_control_flow::StructuredLoop& loop,
    const std::string& container,
    analysis::AnalysisManager& analysis_manager
) {
    return analysis_manager.get<analysis::MemoryLayoutAnalysis>().tile_groups(loop, container);
}

symbolic::Expression tile_element_count(const analysis::MemoryTileGroup* group) {
    if (!group) {
        return symbolic::integer(0);
    }
    symbolic::Expression count = symbolic::integer(1);
    for (auto& extent : group->tile.extents_approx()) {
        if (extent.is_null()) {
            return SymEngine::null;
        }
        count = symbolic::mul(count, extent);
    }
    return symbolic::simplify(count);
}

const analysis::MemoryTileGroup* localizable_tile(
    structured_control_flow::StructuredLoop& loop,
    const std::string& container,
    analysis::AnalysisManager& analysis_manager
) {
    auto* groups = analysis_manager.get<analysis::MemoryLayoutAnalysis>().tile_groups(loop, container);
    if (!groups || groups->size() != 1) {
        return nullptr;
    }
    const auto& group = groups->front();
    std::unordered_set<const data_flow::Memlet*> members(group.memlets.begin(), group.memlets.end());

    // Every memlet of the container in the loop body must belong to the group;
    // an unanalyzable (ungrouped) or split memlet makes wholesale rewriting unsafe.
    bool covered = true;
    std::function<void(structured_control_flow::ControlFlowNode&)> walk;
    walk = [&](structured_control_flow::ControlFlowNode& node) {
        if (!covered) {
            return;
        }
        if (auto* block = dyn_cast<structured_control_flow::Block*>(&node)) {
            auto& dfg = block->dataflow();
            for (auto* access : dfg.data_nodes()) {
                if (access->data() != container) {
                    continue;
                }
                for (auto& memlet : dfg.out_edges(*access)) {
                    if (members.count(&memlet) == 0) {
                        covered = false;
                        return;
                    }
                }
                for (auto& memlet : dfg.in_edges(*access)) {
                    if (members.count(&memlet) == 0) {
                        covered = false;
                        return;
                    }
                }
            }
        } else if (auto* seq = dyn_cast<structured_control_flow::Sequence*>(&node)) {
            for (size_t i = 0; i < seq->size(); i++) {
                walk(seq->at(i));
            }
        } else if (auto* inner = dyn_cast<structured_control_flow::StructuredLoop*>(&node)) {
            walk(inner->root());
        } else if (auto* if_else = dyn_cast<structured_control_flow::IfElse*>(&node)) {
            for (size_t i = 0; i < if_else->size(); i++) {
                walk(if_else->at(i).first);
            }
        }
    };
    walk(loop.root());

    return covered ? &group : nullptr;
}

} // namespace tiles
} // namespace sdfg
