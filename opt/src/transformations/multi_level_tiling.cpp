#include "sdfg/transformations/multi_level_tiling.h"

#include <cstddef>
#include <string>
#include <vector>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/exceptions.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/transformations/loop_tiling.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

MultiLevelTiling::MultiLevelTiling(
    structured_control_flow::StructuredLoop& loop, size_t tile_size, size_t tile_size_2, bool simplify_bounds
)
    : LoopTiling(loop, tile_size, simplify_bounds), additional_tile_sizes_({tile_size_2}), middle_loops_({nullptr}) {};

static size_t extract_first_tile_size(const std::vector<size_t>& tile_sizes) {
    if (tile_sizes.size() < 2) {
        throw InvalidTransformationException(
            "MultiLevelTiling: Expected at least two tile sizes but got: " + std::to_string(tile_sizes.size())
        );
    }
    return tile_sizes[0];
}

MultiLevelTiling::MultiLevelTiling(
    structured_control_flow::StructuredLoop& loop, const std::vector<size_t>& tile_sizes, bool simplify_bounds
)
    : LoopTiling(loop, extract_first_tile_size(tile_sizes), simplify_bounds),
      additional_tile_sizes_(tile_sizes.begin() + 1, tile_sizes.end()), middle_loops_(tile_sizes.size() - 1, nullptr) {}

std::string MultiLevelTiling::name() const { return "MultiLevelTiling"; };

bool MultiLevelTiling::can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    if (!LoopTiling::can_be_applied(builder, analysis_manager)) {
        return false;
    }

    size_t num_additional_tile_sizes = this->additional_tile_sizes_.size();
    for (size_t i = 0; i < num_additional_tile_sizes; i++) {
        if (this->additional_tile_sizes_[i] <= 1) {
            return false;
        }
        size_t previous_tile_size = (i == 0) ? this->tile_size_ : this->additional_tile_sizes_[i - 1];
        if (this->additional_tile_sizes_[i] >= previous_tile_size) {
            return false;
        }
        if (previous_tile_size % this->additional_tile_sizes_[i] != 0) {
            return false;
        }
    }

    std::vector<size_t> tile_sizes = {tile_size_};
    tile_sizes.insert(tile_sizes.end(), additional_tile_sizes_.begin(), additional_tile_sizes_.end());
    return reduction_buffers_supported(builder, analysis_manager, tile_sizes);
};

void MultiLevelTiling::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    // First apply single-level tiling
    LoopTiling::apply(builder, analysis_manager);

    // Now tile the inner loops again with the additional tile sizes. Reuse the shared tiling logic so that behaviour
    // (schedule handling, Map sequentialization, etc.) stays consistent with single-level tiling.
    size_t num_additional_tile_sizes = this->additional_tile_sizes_.size();
    for (size_t i = 0; i < num_additional_tile_sizes; i++) {
        this->middle_loops_[i] =
            &tile_loop(builder, *this->inner_loop_, this->additional_tile_sizes_[i], this->simplify_bounds_);
    }

    analysis_manager.invalidate_all();
};

void MultiLevelTiling::to_json(nlohmann::json& j) const {
    LoopTiling::to_json(j);
    j["transformation_type"] = this->name();
    j["parameters"]["additional_tile_sizes"] = nlohmann::json::array();
    for (size_t additional_tile_size : this->additional_tile_sizes_) {
        j["parameters"]["additional_tile_sizes"].push_back(additional_tile_size);
    }
};

MultiLevelTiling MultiLevelTiling::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc) {
    auto loop_id = desc["subgraph"]["0"]["element_id"].get<size_t>();
    size_t tile_size = desc["parameters"]["tile_size"].get<size_t>();
    std::vector<size_t> tile_sizes = {tile_size};
    if (desc["parameters"].contains("tile_size_2")) {
        // Old format
        size_t tile_size_2 = desc["parameters"]["tile_size_2"].get<size_t>();
        tile_sizes.push_back(tile_size_2);
    } else {
        // New format
        auto additional_tile_sizes = desc["parameters"]["additional_tile_sizes"].get<std::vector<size_t>>();
        tile_sizes.insert(tile_sizes.end(), additional_tile_sizes.begin(), additional_tile_sizes.end());
    }
    bool simplify_bounds = desc["parameters"].value("simplify_bounds", false);
    auto element = builder.find_element_by_id(loop_id);
    if (!element) {
        throw InvalidTransformationDescriptionException("Element with ID " + std::to_string(loop_id) + " not found.");
    }
    auto loop = dyn_cast<structured_control_flow::StructuredLoop*>(element);

    return MultiLevelTiling(*loop, tile_sizes, simplify_bounds);
};

const std::vector<structured_control_flow::StructuredLoop*>& MultiLevelTiling::middle_loops() {
    if (!applied_) {
        throw InvalidSDFGException("MultiLevelTiling: Accessing tiled loops before their creation.");
    }
    return this->middle_loops_;
}

structured_control_flow::StructuredLoop* MultiLevelTiling::middle_loop(size_t idx) {
    if (!applied_) {
        throw InvalidSDFGException("MultiLevelTiling: Accessing tiled loop before their creation.");
    }
    if (idx >= this->middle_loops_.size()) {
        throw InvalidSDFGException(
            "MultiLevelTiling: Access to tiled loop is out of bounds: " + std::to_string(idx) + " not in [0, " +
            std::to_string(this->middle_loops_.size() - 1) + "]"
        );
    }
    return this->middle_loops_[idx];
}

structured_control_flow::StructuredLoop* MultiLevelTiling::middle_loop() {
    if (this->middle_loops_.size() != 1) {
        throw InvalidSDFGException(
            "MultiLevelTiling: Expected exactly one middle loop but got: " + std::to_string(this->middle_loops_.size())
        );
    }
    return this->middle_loop(0);
}

} // namespace transformations
} // namespace sdfg
