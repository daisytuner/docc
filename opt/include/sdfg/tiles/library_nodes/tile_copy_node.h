#pragma once

#include "sdfg/data_flow/library_node.h"

#include <functional>
#include <string>

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/tiles/tiled_copy.h"

namespace sdfg {
namespace tiles {

inline data_flow::LibraryNodeCode LibraryNodeType_TileCopy{"tile_copy"};

/**
 * @brief A tiled copy carried as a value @ref TiledCopy plan.
 *
 * Instead of expanding a copy into a map-nest + scalar tasklet + reference-memlet
 * subgraph, the whole copy is one node. The `{_dst, _src}` inputs are the bare
 * base pointers (empty-subset computational memlets — no dereference); the element
 * offsets and the widening legality both derive from @ref plan(): `src`/`dst` are
 * the global/buffer geometries and `(thread, value)` partition the tile. This keeps
 * `symbols()`/`replace()` self-contained (they rewrite the plan's layouts directly),
 * so a pass can retarget the copy without touching surrounding control flow.
 */
class TileCopyNode : public data_flow::LibraryNode {
    TiledCopy plan_;
    CopyDirection direction_;
    size_t bytes_; ///< transfer width per lane in bytes (the atom's width)
    /// Per-element ragged-tile boundary predicate; trivial ⇒ fully covering.
    TileGuard guard_;
    /// Which block spatial axes (0=x,1=y,2=z) cooperate on the sweep. Empty means the
    /// whole block (all threads) fills a slot-free tile. A non-empty subset is the
    /// per-thread-slot mode: each thread stages its *own* slot (the slot index is baked
    /// into `plan.dst`/`plan.src` offsets) while these axes split the tile — compatible
    /// with a ragged per-thread guard, unlike a whole-block fill.
    std::vector<int> coop_axes_;

public:
    TileCopyNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const data_flow::ImplementationType& implementation_type,
        TiledCopy plan,
        CopyDirection direction,
        size_t bytes,
        TileGuard guard = {},
        std::vector<int> coop_axes = {}
    );

    const TiledCopy& plan() const { return plan_; }
    void set_plan(TiledCopy plan) { plan_ = std::move(plan); }

    CopyDirection direction() const { return direction_; }
    CopyAtom atom() const { return plan_.atom; }
    void set_atom(CopyAtom atom) { plan_.atom = atom; }

    size_t bytes() const { return bytes_; }
    void set_bytes(size_t bytes) { bytes_ = bytes; }

    const TileGuard& guard() const { return guard_; }

    /// Re-discharge the boundary guard against @p assumptions (dims proven fully
    /// covering are dropped), so a fullness proof from a later pass — loop peeling,
    /// condition propagation — cleans the guard as it would a map-nest copy. Returns
    /// true if the guard changed.
    bool normalize_guard(const symbolic::SymbolSet& parameters, const symbolic::Assumptions& assumptions) {
        return guard_.discharge(parameters, assumptions);
    }

    const std::vector<int>& coop_axes() const { return coop_axes_; }


    void validate(const Function& function) const override;

    symbolic::SymbolSet symbols() const override;

    std::unique_ptr<DataFlowNode> clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent)
        const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    void replace(const symbolic::ExpressionMapping& replacements) override;

    /// {"_dst", "_src"}: writes _dst, reads _src, captures neither. Lets escape
    /// analysis treat a container staged through this copy as an ordinary copy so a
    /// later transformation (e.g. a register-accumulator LocalStorage) still applies.
    data_flow::PointerAccessType pointer_access_type(int input_idx) const override;
};

class TileCopyNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const sdfg::data_flow::LibraryNode& library_node) override;
    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j,
        sdfg::builder::StructuredSDFGBuilder& builder,
        sdfg::structured_control_flow::Block& parent
    ) override;
};

/// The single-element copy statement `dst[dst_geo(index)] = src[src_geo(index)];`
/// for the tile coordinate @p index, addressing both bare-pointer sides as flat
/// @p elem pointers via the plan's layouts. Shared by every @ref TileCopyNode
/// dispatcher so the loop shape (sequential / cooperative / async) is the only
/// difference between targets. @p dst_expr / @p src_expr are the `{_dst,_src}`
/// input expressions.
std::string emit_tile_copy_stmt(
    codegen::LanguageExtension& language_extension,
    const TileCopyNode& node,
    const std::string& dst_expr,
    const std::string& src_expr,
    types::PrimitiveType elem,
    const symbolic::Expression& index
);

/// Emit the block-cooperative staging loop for @p node into @p stream: every thread
/// of the enclosing kernel block strides the flat tile from its flat index by the
/// block thread count. `ScalarSync` moves one element per step; `VectorSync` moves a
/// contiguous @ref TileCopyNode::bytes -wide vector per step (the loop strides by the
/// vector's element count). For `CpAsync`, @p async_stmt renders the per-step async
/// transfer given the destination/source element addresses and the byte width (the
/// target supplies its own primitive; e.g. CUDA `__pipeline_memcpy_async`). Shared by
/// the CUDA and ROCm dispatchers; the visibility barrier is the producer's.
void emit_cooperative_copy_loop(
    codegen::LanguageExtension& language_extension,
    codegen::PrettyPrinter& stream,
    const TileCopyNode& node,
    const std::string& dst_expr,
    const std::string& src_expr,
    types::PrimitiveType elem,
    const std::function<std::string(const std::string& dst_addr, const std::string& src_addr, size_t bytes)>&
        async_stmt = {}
);

/**
 * @brief Reference (sequential, scalar) dispatcher for @ref TileCopyNode.
 *
 * Emits the whole copy as one sequential loop over the flat tile:
 * `for c in [0, size): dst[dst_geo(c)] = src[src_geo(c)]`, addressing each side by
 * the plan's @ref Layout::resolve_element. The `{_dst,_src}` inputs are the base pointers, so
 * both sides are reinterpreted to a flat element pointer and indexed linearly. Used
 * on the host (CPU private copy) and as the fallback where no cooperative/async atom
 * applies; GPU targets register their own dispatcher for the offload atoms.
 */
class TileCopyNodeDispatcher : public codegen::LibraryNodeDispatcher {
public:
    TileCopyNodeDispatcher(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const TileCopyNode& node
    );

    void dispatch_code_with_edges(
        codegen::CodegenOutput& out,
        std::vector<codegen::DispatchInput>& inputs,
        std::vector<codegen::DispatchOutput>& outputs
    ) override;
};

} // namespace tiles
} // namespace sdfg
