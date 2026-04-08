#include "sdfg/transformations/collapse_to_depth.h"
#include <cstddef>

#include "sdfg/structured_control_flow/map.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/transformations/map_collapse.h"

namespace sdfg {
namespace transformations {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static size_t perfectly_nested_map_depth(structured_control_flow::Map& map) {
    size_t depth = 1;
    auto* current = &map;
    symbolic::SymbolSet indvars;
    indvars.insert(map.indvar());
    while (true) {
        auto& body = current->root();
        if (body.size() != 1) {
            break;
        }
        auto* next = dynamic_cast<structured_control_flow::Map*>(&body.at(0).first);
        if (!next) {
            break;
        }
        for (const auto& atom : symbolic::atoms(next->init())) {
            if (indvars.contains(atom)) {
                return depth;
            }
        }
        for (const auto& atom : symbolic::atoms(next->condition())) {
            if (indvars.contains(atom)) {
                return depth;
            }
        }
        ++depth;
        current = next;
        indvars.insert(current->indvar());
    }
    return depth;
}

// ---------------------------------------------------------------------------
// CollapseToDepth
// ---------------------------------------------------------------------------

CollapseToDepth::CollapseToDepth(structured_control_flow::Map& loop, size_t target_loops)
    : loop_(loop), target_loops_(target_loops) {}

std::string CollapseToDepth::name() const { return "CollapseToDepth"; }

bool CollapseToDepth::can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    if (target_loops_ < 1 || target_loops_ > 2) {
        return false;
    }

    size_t depth = perfectly_nested_map_depth(loop_);
    if (depth <= target_loops_) {
        return false;
    }

    if (target_loops_ == 1) {
        MapCollapse t(loop_, depth);
        return t.can_be_applied(builder, analysis_manager);
    }

    // target_loops_ == 2
    size_t outer_count = (depth + 1) / 2;
    size_t inner_count = depth - outer_count;

    // Check inner half first
    if (inner_count >= 2) {
        auto* inner_start = &loop_;
        for (size_t i = 0; i < outer_count; ++i) {
            inner_start = dynamic_cast<structured_control_flow::Map*>(&inner_start->root().at(0).first);
        }
        MapCollapse t_inner(*inner_start, inner_count);
        if (!t_inner.can_be_applied(builder, analysis_manager)) {
            return false;
        }
    }

    // Check outer half
    if (outer_count >= 2) {
        MapCollapse t_outer(loop_, outer_count);
        if (!t_outer.can_be_applied(builder, analysis_manager)) {
            return false;
        }
    }

    return true;
}

void CollapseToDepth::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    size_t depth = perfectly_nested_map_depth(loop_);

    if (target_loops_ == 1) {
        MapCollapse t(loop_, depth);
        t.apply(builder, analysis_manager);
        outer_loop_ = t.collapsed_loop();
        inner_loop_ = nullptr;
    } else {
        // target_loops_ == 2
        size_t outer_count = (depth + 1) / 2;
        size_t inner_count = depth - outer_count;

        // Collapse inner half first (so that the outer map chain stays valid)
        if (inner_count >= 2) {
            auto* inner_start = &loop_;
            for (size_t i = 0; i < outer_count; ++i) {
                inner_start = dynamic_cast<structured_control_flow::Map*>(&inner_start->root().at(0).first);
            }
            MapCollapse t_inner(*inner_start, inner_count);
            t_inner.apply(builder, analysis_manager);
            inner_loop_ = t_inner.collapsed_loop();
        } else {
            // inner half is a single map, no collapse needed — find it
            auto* m = &loop_;
            for (size_t i = 0; i < outer_count; ++i) {
                m = dynamic_cast<structured_control_flow::Map*>(&m->root().at(0).first);
            }
            inner_loop_ = m;
        }

        // Collapse outer half
        if (outer_count >= 2) {
            MapCollapse t_outer(loop_, outer_count);
            t_outer.apply(builder, analysis_manager);
            outer_loop_ = t_outer.collapsed_loop();
        } else {
            outer_loop_ = &loop_;
        }
    }

    applied_ = true;
}

void CollapseToDepth::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["subgraph"] = {{"0", {{"element_id", loop_.element_id()}, {"type", "map"}}}};
    j["parameters"] = {{"target_loops", target_loops_}};
}

CollapseToDepth CollapseToDepth::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc) {
    auto loop_id = desc["subgraph"]["0"]["element_id"].get<size_t>();
    size_t target_loops = desc["parameters"]["target_loops"].get<size_t>();
    auto element = builder.find_element_by_id(loop_id);
    if (!element) {
        throw InvalidTransformationDescriptionException("Element with ID " + std::to_string(loop_id) + " not found.");
    }
    auto loop = dynamic_cast<structured_control_flow::Map*>(element);
    return CollapseToDepth(*loop, target_loops);
}

structured_control_flow::Map* CollapseToDepth::outer_loop() {
    if (!applied_) {
        return &loop_;
    }
    return outer_loop_;
}

structured_control_flow::Map* CollapseToDepth::inner_loop() {
    if (!applied_) {
        throw InvalidSDFGException("Accessing collapsed loop before transformation has been applied.");
    }
    return inner_loop_;
}

} // namespace transformations
} // namespace sdfg
