#include "sdfg/transformations/multi_level_tiling.h"

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace transformations {

MultiLevelTiling::MultiLevelTiling(
    structured_control_flow::StructuredLoop& loop, size_t tile_size, size_t tile_size_2, bool simplify_bounds
)
    : LoopTiling(loop, tile_size, simplify_bounds), tile_size_2_(tile_size_2) {};

std::string MultiLevelTiling::name() const { return "MultiLevelTiling"; };

bool MultiLevelTiling::can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    if (!LoopTiling::can_be_applied(builder, analysis_manager)) {
        return false;
    }
    if (this->tile_size_2_ <= 1) {
        return false;
    }
    if (this->tile_size_2_ >= this->tile_size_) {
        return false;
    }
    if (this->tile_size_ % this->tile_size_2_ != 0) {
        return false;
    }
    return true;
};

void MultiLevelTiling::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    // First apply single-level tiling
    LoopTiling::apply(builder, analysis_manager);

    // Now tile the inner (point) loop again with tile_size_2_. Reuse the shared
    // tiling logic so that behaviour (schedule handling, Map sequentialization,
    // etc.) stays consistent with single-level tiling.
    auto& inner = *inner_loop_;
    middle_loop_ = &tile_loop(builder, inner, this->tile_size_2_, this->simplify_bounds_);

    analysis_manager.invalidate_all();
    inner_loop_ = &inner;
};

void MultiLevelTiling::to_json(nlohmann::json& j) const {
    LoopTiling::to_json(j);
    j["transformation_type"] = this->name();
    j["parameters"]["tile_size_2"] = tile_size_2_;
};

MultiLevelTiling MultiLevelTiling::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc) {
    auto loop_id = desc["subgraph"]["0"]["element_id"].get<size_t>();
    size_t tile_size = desc["parameters"]["tile_size"].get<size_t>();
    size_t tile_size_2 = desc["parameters"]["tile_size_2"].get<size_t>();
    bool simplify_bounds = desc["parameters"].value("simplify_bounds", false);
    auto element = builder.find_element_by_id(loop_id);
    if (!element) {
        throw InvalidTransformationDescriptionException("Element with ID " + std::to_string(loop_id) + " not found.");
    }
    auto loop = dyn_cast<structured_control_flow::StructuredLoop*>(element);

    return MultiLevelTiling(*loop, tile_size, tile_size_2, simplify_bounds);
};

structured_control_flow::StructuredLoop* MultiLevelTiling::middle_loop() {
    if (!applied_) {
        throw InvalidSDFGException("Accessing tiled loop before their creation.");
    }
    return middle_loop_;
}

} // namespace transformations
} // namespace sdfg
