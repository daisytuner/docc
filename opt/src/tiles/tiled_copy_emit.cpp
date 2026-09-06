#include "sdfg/tiles/tiled_copy_emit.h"

#include "sdfg/data_flow/library_nodes/barrier_local_node.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/types/scalar.h"

namespace sdfg {
namespace tiles {

namespace {

// Emit one element copy `global[global_addr] <-> buffer[buffer_addr]` into @p body,
// oriented by @p direction.
void emit_element(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& body,
    const CopyContainers& c,
    CopyDirection direction,
    const data_flow::Subset& global_addr,
    const data_flow::Subset& buffer_addr
) {
    auto& block = builder.add_block(body);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    if (direction == CopyDirection::In) {
        auto& g = builder.add_access(block, c.global);
        auto& b = builder.add_access(block, c.buffer);
        builder.add_computational_memlet(block, g, tasklet, "_in", global_addr, *c.global_type);
        builder.add_computational_memlet(block, tasklet, "_out", b, buffer_addr, *c.buffer_type);
    } else {
        auto& b = builder.add_access(block, c.buffer);
        auto& g = builder.add_access(block, c.global);
        builder.add_computational_memlet(block, b, tasklet, "_in", buffer_addr, *c.buffer_type);
        builder.add_computational_memlet(block, tasklet, "_out", g, global_addr, *c.global_type);
    }
}

// Row-major decomposition of a flat index into per-mode coordinates (matches the
// buffer's tile linearization, so the flat map addresses src/dst consistently).
std::vector<symbolic::Expression>
delinearize_rowmajor(const symbolic::Expression& flat, const symbolic::MultiExpression& sizes) {
    std::vector<symbolic::Expression> decomp;
    symbolic::Expression remainder = flat;
    for (size_t i = 0; i < sizes.size(); ++i) {
        if (i + 1 < sizes.size()) {
            symbolic::Expression divisor = symbolic::integer(1);
            for (size_t j = i + 1; j < sizes.size(); ++j) divisor = symbolic::mul(divisor, sizes[j]);
            decomp.push_back(symbolic::div(remainder, divisor));
            remainder = symbolic::mod(remainder, divisor);
        } else {
            decomp.push_back(remainder);
        }
    }
    return decomp;
}

} // namespace

void emit_into(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& scope,
    const TiledCopy& plan,
    const CopyContainers& containers,
    CopyDirection direction,
    const PackedBuffer& dst_buffer,
    const structured_control_flow::ScheduleType* schedule,
    const std::vector<symbolic::Expression>& slot_indices,
    const BoundaryGuard& guard,
    Coverage coverage
) {
    auto sched = schedule ? *schedule : structured_control_flow::ScheduleType_Sequential::create();

    structured_control_flow::Sequence* current = &scope;
    std::vector<symbolic::Expression> coords;
    if (coverage == Coverage::Flat) {
        // A single map over the linearized tile; the offload schedule splits it
        // across the cooperative lanes. Row-major delinearize back to tile coords.
        auto name = builder.find_new_name("__tc_c");
        builder.add_container(name, types::Scalar(types::PrimitiveType::Int32));
        auto c = symbolic::symbol(name);
        auto& map = builder.add_map(
            scope,
            c,
            symbolic::Lt(c, plan.src.size()),
            symbolic::integer(0),
            symbolic::add(c, symbolic::integer(1)),
            sched,
            DebugInfo()
        );
        current = &map.root();
        coords = delinearize_rowmajor(c, plan.src.shape());
    } else {
        // One nested map per src mode (a map, not a for: the element copies are
        // independent, so later passes are free to reschedule/parallelize them). The
        // per-mode indices form the tile coordinate.
        for (size_t m = 0; m < plan.src.rank(); ++m) {
            auto name = builder.find_new_name("__tc_i");
            builder.add_container(name, types::Scalar(types::PrimitiveType::Int32));
            auto iv = symbolic::symbol(name);
            coords.push_back(iv);
            auto& map = builder.add_map(
                *current,
                iv,
                symbolic::Lt(iv, plan.src.shape()[m]),
                symbolic::integer(0),
                symbolic::add(iv, symbolic::integer(1)),
                sched,
                DebugInfo()
            );
            current = &map.root();
        }
    }

    // Global side addressed linearly (flat pointer); buffer side multi-dimensionally
    // via the packed-buffer model (per-thread slot prefix + per-axis tile indices).
    data_flow::Subset global_addr = {plan.src.apply_coords(coords)};
    data_flow::Subset buffer_addr = dst_buffer.subset(slot_indices, coords);

    // Element-predicate the copy: the over-approximated tile may address
    // out-of-bounds global memory on ragged blocks, so skip those elements.
    structured_control_flow::Sequence* body = current;
    if (guard) {
        auto g = guard(coords);
        if (!symbolic::is_true(g)) {
            auto& if_else = builder.add_if_else(*current, DebugInfo());
            body = &builder.add_case(if_else, g, DebugInfo());
        }
    }

    emit_element(builder, *body, containers, direction, global_addr, buffer_addr);
}

void emit(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& parent,
    structured_control_flow::ControlFlowNode& before,
    const TiledCopy& plan,
    const CopyContainers& containers,
    CopyDirection direction,
    SyncPolicy sync,
    const BoundaryGuard& guard
) {
    // Everything the copy emits lives in one scope inserted before the consumer, so
    // the map nest, the boundary guard, and the barriers compose uniformly (and the
    // degenerate rank-0 tile needs no special casing).
    auto& scope = builder.add_sequence_before(parent, before, DebugInfo());

    // Leading barrier (guards the buffer from overwrite before prior reads finish).
    if (sync == SyncPolicy::SingleStage) {
        auto& pre = builder.add_block(scope);
        builder.add_library_node<data_flow::BarrierLocalNode>(pre, DebugInfo());
    }

    // A dense multidimensional buffer whose axes are the tile modes (no slots).
    PackedBuffer dst_buffer{{}, plan.dst.shape(), BufferKind::MultiDim};
    emit_into(builder, scope, plan, containers, direction, dst_buffer, nullptr, {}, guard);

    // Trailing barrier (publishes the staged tile before consumers read it).
    if (sync == SyncPolicy::SingleStage) {
        auto& post = builder.add_block(scope);
        builder.add_library_node<data_flow::BarrierLocalNode>(post, DebugInfo());
    }
}

} // namespace tiles
} // namespace sdfg
