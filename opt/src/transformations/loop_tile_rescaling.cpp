#include "sdfg/transformations/loop_tile_rescaling.h"

#include <symengine/integer.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace transformations {

LoopTileRescaling::LoopTileRescaling(
    structured_control_flow::StructuredLoop& outer_loop,
    size_t new_tile_size
)
    : outer_loop_(outer_loop), new_tile_size_(new_tile_size) {}

std::string LoopTileRescaling::name() const { return "LoopTileRescaling"; }

bool LoopTileRescaling::can_be_applied(
    builder::StructuredSDFGBuilder& /*builder*/, analysis::AnalysisManager& /*analysis_manager*/
) {
    // Outer update must be indvar + k, k > 1
    auto step = symbolic::sub(outer_loop_.update(), outer_loop_.indvar());
    if (!SymEngine::is_a<SymEngine::Integer>(*step)) return false;
    old_tile_size_ =
        static_cast<size_t>(SymEngine::down_cast<const SymEngine::Integer&>(*step).as_int());
    if (old_tile_size_ <= 1) return false;

    // Find exactly one inner loop in the outer body whose init equals the outer indvar
    auto& body = outer_loop_.root();
    structured_control_flow::StructuredLoop* found_inner = nullptr;
    for (size_t i = 0; i < body.size(); ++i) {
        auto* candidate = dynamic_cast<structured_control_flow::StructuredLoop*>(&body.at(i).first);
        if (!candidate) continue;
        if (found_inner != nullptr) return false; // more than one inner loop
        found_inner = candidate;
    }
    if (!found_inner) return false;
    if (!symbolic::eq(found_inner->init(), outer_loop_.indvar())) return false;

    inner_loop_ = found_inner;
    return true;
}

void LoopTileRescaling::apply(
    builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager
) {
    if (new_tile_size_ == old_tile_size_) return;

    auto outer_sym = outer_loop_.indvar();

    // Update outer loop: change stride from old_tile_size_ to new_tile_size_
    auto new_outer_update = symbolic::add(outer_sym, symbolic::integer(static_cast<int64_t>(new_tile_size_)));
    builder.update_loop(outer_loop_, outer_sym, outer_loop_.condition(), outer_loop_.init(), new_outer_update);

    // Update inner loop condition: replace (outer + old_tile) with (outer + new_tile)
    auto old_bound = symbolic::add(outer_sym, symbolic::integer(static_cast<int64_t>(old_tile_size_)));
    auto new_bound = symbolic::add(outer_sym, symbolic::integer(static_cast<int64_t>(new_tile_size_)));
    auto new_inner_cond = symbolic::subs(inner_loop_->condition(), old_bound, new_bound);
    builder.update_loop(*inner_loop_, inner_loop_->indvar(), new_inner_cond, inner_loop_->init(), inner_loop_->update());

    analysis_manager.invalidate_all();
}

void LoopTileRescaling::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["subgraph"] = {
        {"0", {{"element_id", outer_loop_.element_id()}}}
    };
    j["parameters"] = {{"new_tile_size", new_tile_size_}};
}

LoopTileRescaling LoopTileRescaling::from_json(
    builder::StructuredSDFGBuilder& builder, const nlohmann::json& j
) {
    auto outer_id = j["subgraph"]["0"]["element_id"].get<size_t>();
    size_t new_tile_size = j["parameters"]["new_tile_size"].get<size_t>();

    auto* outer_elem = builder.find_element_by_id(outer_id);
    if (!outer_elem) {
        throw InvalidTransformationDescriptionException(
            "LoopTileRescaling: outer loop element not found (id=" + std::to_string(outer_id) + ")"
        );
    }

    auto* outer = dynamic_cast<structured_control_flow::StructuredLoop*>(outer_elem);
    if (!outer) {
        throw InvalidTransformationDescriptionException("LoopTileRescaling: element is not a StructuredLoop");
    }

    return LoopTileRescaling(*outer, new_tile_size);
}

} // namespace transformations
} // namespace sdfg
