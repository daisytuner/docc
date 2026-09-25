#include "sdfg/transformations/loop_tiling.h"

#include <algorithm>
#include <limits>

#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/gpu/gpu_map_utils.h"
#include "sdfg/tiles/analysis/reduction_buffer_analysis.h"

#include <symengine/integer.h>

namespace sdfg {
namespace transformations {

namespace {

/// A unit-stride loop whose exact trip is a positive integer multiple of `tile_size` fills every
/// tile, so the original bound is redundant after tiling and can be dropped. Cheap constant check
/// (no assumptions analysis); works progressively -- a first cut that drops its bound makes the
/// inner extent a constant, so a second (MultiLevelTiling) cut divides cleanly too.
bool tile_evenly_divides(const symbolic::Integer& stride, const symbolic::Expression& trip, size_t tile_size) {
    if (tile_size == 0 || tile_size > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
        return false;
    }
    if (stride.is_null() || !SymEngine::is_a<SymEngine::Integer>(*stride) ||
        SymEngine::rcp_static_cast<const SymEngine::Integer>(stride)->as_int() != 1) {
        return false;
    }
    if (trip.is_null() || !SymEngine::is_a<SymEngine::Integer>(*trip)) {
        return false;
    }
    long long trip_int = SymEngine::rcp_static_cast<const SymEngine::Integer>(trip)->as_int();
    return trip_int > 0 && (trip_int % static_cast<long long>(tile_size)) == 0;
}

struct TileGeometry {
    tiles::ReductionLoopHeader outer;
    tiles::ReductionLoopHeader inner;
    symbolic::Expression outer_count;
    symbolic::Expression inner_count;
};

TileGeometry tile_geometry(
    const symbolic::Symbol& indvar,
    const tiles::ReductionLoopHeader& header,
    const symbolic::Integer& stride,
    const symbolic::Expression& count,
    const symbolic::Symbol& tile_indvar,
    size_t tile_size,
    bool simplify_bounds
) {
    const bool full_tiles = tile_evenly_divides(stride, count, tile_size);
    auto tile_extent = symbolic::integer(tile_size);
    auto inner_condition = symbolic::Lt(indvar, symbolic::add(tile_indvar, tile_extent));
    if (!simplify_bounds || !full_tiles) {
        inner_condition = symbolic::And(inner_condition, header.condition);
    }
    return {
        {header.init, symbolic::subs(header.condition, indvar, tile_indvar), symbolic::add(tile_indvar, tile_extent)},
        {tile_indvar, inner_condition, symbolic::add(indvar, symbolic::one())},
        full_tiles ? symbolic::div(count, tile_extent) : SymEngine::null,
        full_tiles ? tile_extent : SymEngine::null
    };
}

std::vector<tiles::ReductionLoopDomain> projected_tile_domains(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::StructuredLoop& loop,
    const std::vector<size_t>& tile_sizes,
    bool simplify_bounds
) {
    tiles::ReductionLoopHeader header{loop.init(), loop.condition(), loop.update()};
    auto count = loop.num_iterations();
    auto stride = loop.stride();
    std::vector<tiles::ReductionLoopDomain> result;
    for (size_t level = 0; level < tile_sizes.size(); ++level) {
        auto tile_indvar =
            symbolic::symbol(builder
                                 .find_new_name(loop.indvar()->get_name() + "_tile_preview_" + std::to_string(level) + "_")
            );
        auto geometry =
            tile_geometry(loop.indvar(), header, stride, count, tile_indvar, tile_sizes[level], simplify_bounds);
        auto outer_domain = tiles::ReductionLoopDomain::from_header(tile_indvar, geometry.outer);
        if (!geometry.outer_count.is_null()) {
            outer_domain.count = geometry.outer_count;
        }
        result.push_back(std::move(outer_domain));
        header = std::move(geometry.inner);
        count = geometry.inner_count.is_null() ? tiles::ReductionLoopDomain::from_header(loop.indvar(), header).count
                                               : geometry.inner_count;
        stride = symbolic::one();
    }
    result.push_back({loop.indvar(), header.init, count, stride});
    std::reverse(result.begin(), result.end());
    return result;
}

} // namespace

LoopTiling::LoopTiling(structured_control_flow::StructuredLoop& loop, size_t tile_size, bool simplify_bounds)
    : loop_(loop), tile_size_(tile_size), simplify_bounds_(simplify_bounds) {};

std::string LoopTiling::name() const {
    return "LoopTiling";
};

bool LoopTiling::can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    if (this->tile_size_ <= 1) {
        return false;
    }
    return loop_.is_contiguous() && reduction_buffers_supported(builder, analysis_manager, {tile_size_});
};

bool LoopTiling::reduction_buffers_supported(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    const std::vector<size_t>& tile_sizes
) const {
    auto& buffers = analysis_manager.get<tiles::ReductionBufferAnalysis>();
    const auto affected = buffers.affected_reductions(loop_);
    if (affected.empty()) {
        return true;
    }
    for (auto* reduction : affected) {
        for (const auto& entry : reduction->reductions()) {
            if (!entry.original_index.is_null()) {
                return false;
            }
        }
    }
    {
        const auto tiles = projected_tile_domains(builder, loop_, tile_sizes, simplify_bounds_);
        auto& loops = analysis_manager.get<analysis::LoopAnalysis>();
        structured_control_flow::ControlFlowNode* root = &loop_;
        while (auto* parent = loops.parent_loop(root)) {
            root = parent;
        }
        auto candidates = loops.descendants(root);
        candidates.insert(root);
        bool exact = true;
        for (auto* node : candidates) {
            auto* reduction = dyn_cast<structured_control_flow::Reduce*>(node);
            if (!reduction || !gpu::is_gpu_schedule(reduction->schedule_type()) ||
                !reduction->schedule_type().properties().contains("target_level")) {
                continue;
            }
            std::vector<structured_control_flow::StructuredLoop*> inner;
            for (auto* descendant : loops.descendants(reduction)) {
                if (auto* loop = dyn_cast<structured_control_flow::StructuredLoop*>(descendant)) {
                    inner.push_back(loop);
                }
            }
            std::sort(inner.begin(), inner.end(), [&](auto* left, auto* right) {
                return loops.ancestors(left).size() > loops.ancestors(right).size();
            });
            std::vector<tiles::ReductionLoopDomain> domains;
            for (auto* loop : inner) {
                if (loop == &loop_) {
                    domains.insert(domains.end(), tiles.begin(), tiles.end());
                } else {
                    domains.push_back({loop->indvar(), loop->init(), loop->num_iterations(), loop->stride()});
                }
            }
            for (const auto& entry : reduction->reductions()) {
                auto footprint = buffers.buffer(*reduction, entry.container);
                if (!loops.descendants(reduction).contains(&loop_)) {
                    exact &= footprint.status == tiles::ReductionBufferStatus::Exact;
                    continue;
                }
                auto estimate = buffers.estimate_geometry(*reduction, entry.container, std::move(footprint), domains);
                exact &= estimate.status == tiles::ReductionBufferStatus::Exact;
            }
        }
        return exact;
    }
}

void LoopTiling::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    if (!can_be_applied(builder, analysis_manager)) {
        throw InvalidSDFGException("LoopTiling: unsupported loop or proposed reduction footprint");
    }
    outer_loop_ = &tile_loop(builder, loop_, this->tile_size_, this->simplify_bounds_);
    inner_loop_ = &loop_;

    analysis_manager.invalidate_all();
    applied_ = true;
};

structured_control_flow::StructuredLoop& LoopTiling::tile_loop(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::StructuredLoop& loop,
    size_t tile_size,
    bool simplify_bounds
) {
    auto& sdfg = builder.subject();

    auto parent = static_cast<structured_control_flow::Sequence*>(loop.get_parent());
    size_t index = parent->index(loop);

    auto indvar = loop.indvar();

    // Step 1: Define new outer loop
    auto outer_indvar_str = builder.find_new_name(indvar->get_name() + "_tile");
    builder.add_container(outer_indvar_str, sdfg.type(loop.indvar()->get_name()));
    auto outer_indvar = symbolic::symbol(outer_indvar_str);
    const auto geometry = tile_geometry(
        indvar,
        {loop.init(), loop.condition(), loop.update()},
        loop.stride(),
        loop.num_iterations(),
        outer_indvar,
        tile_size,
        simplify_bounds
    );
    const auto& outer_condition = geometry.outer.condition;
    const auto& outer_update = geometry.outer.update;

    structured_control_flow::StructuredLoop* outer_loop = nullptr;
    if (auto map = dyn_cast<structured_control_flow::Map*>(&loop)) {
        outer_loop = &builder.add_map_before(
            *parent,
            loop,
            outer_indvar,
            outer_condition,
            loop.init(),
            outer_update,
            map->schedule_type(),
            loop.debug_info()
        );
    } else if (auto reduce = dyn_cast<structured_control_flow::Reduce*>(&loop)) {
        outer_loop = &builder.add_reduce_before(
            *parent,
            loop,
            outer_indvar,
            outer_condition,
            loop.init(),
            outer_update,
            reduce->reductions(),
            reduce->schedule_type(),
            loop.debug_info()
        );
    } else {
        outer_loop = &builder.add_for_before(
            *parent, loop, outer_indvar, outer_condition, loop.init(), outer_update, loop.debug_info()
        );
    }

    // Step 2: Redefine inner loop
    auto inner_indvar = indvar;
    builder.update_loop(loop, inner_indvar, geometry.inner.condition, geometry.inner.init, geometry.inner.update);

    // When tiling a Map, the outer tile loop inherits the schedule (created above
    // via add_map_before), but the inner element loop must become sequential.
    // Otherwise nested GPU Maps end up with repeated dimensions.
    if (dyn_cast<structured_control_flow::Map*>(&loop)) {
        builder.update_schedule_type(loop, structured_control_flow::ScheduleType_Sequential::create());
    }

    // Step 3: Move loop into tiling loop
    builder.move_child(*parent, index + 1, outer_loop->root());

    return *outer_loop;
};

void LoopTiling::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["parameters"] = nlohmann::json::object();
    j["parameters"] = {{"tile_size", tile_size_}, {"simplify_bounds", simplify_bounds_}};

    serializer::JSONSerializer ser_flat(false);
    j["subgraph"] = nlohmann::json::object();
    j["subgraph"]["0"] = nlohmann::json::object();
    ser_flat.serialize_node(j["subgraph"]["0"], loop_);
};

LoopTiling LoopTiling::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc) {
    auto loop_id = desc["subgraph"]["0"]["element_id"].get<size_t>();
    size_t tile_size = desc["parameters"]["tile_size"].get<size_t>();
    bool simplify_bounds = desc["parameters"].value("simplify_bounds", false);
    auto element = builder.find_element_by_id(loop_id);
    if (!element) {
        throw InvalidTransformationDescriptionException("Element with ID " + std::to_string(loop_id) + " not found.");
    }
    auto loop = dyn_cast<structured_control_flow::StructuredLoop*>(element);

    return LoopTiling(*loop, tile_size, simplify_bounds);
};

structured_control_flow::StructuredLoop* LoopTiling::inner_loop() {
    if (!applied_) {
        throw InvalidSDFGException("Accessing tiled loop before their creation.");
    }

    return inner_loop_;
}

structured_control_flow::StructuredLoop* LoopTiling::outer_loop() {
    if (!applied_) {
        throw InvalidSDFGException("Accessing tiled loop before their creation.");
    }

    return outer_loop_;
}

} // namespace transformations
} // namespace sdfg
