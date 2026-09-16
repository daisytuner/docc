#include "sdfg/transformations/loop_tiling.h"

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"

#include <symengine/integer.h>

namespace sdfg {
namespace transformations {

namespace {

/// A unit-stride loop whose exact trip is a positive integer multiple of `tile_size` fills every
/// tile, so the original bound is redundant after tiling and can be dropped. Cheap constant check
/// (no assumptions analysis); works progressively -- a first cut that drops its bound makes the
/// inner extent a constant, so a second (MultiLevelTiling) cut divides cleanly too.
bool tile_evenly_divides(const structured_control_flow::StructuredLoop& loop, size_t tile_size) {
    auto stride = loop.stride();
    if (stride.is_null() || !SymEngine::is_a<SymEngine::Integer>(*stride) ||
        SymEngine::rcp_static_cast<const SymEngine::Integer>(stride)->as_int() != 1) {
        return false;
    }
    auto trip = loop.num_iterations();
    if (trip.is_null() || !SymEngine::is_a<SymEngine::Integer>(*trip)) {
        return false;
    }
    long long trip_int = SymEngine::rcp_static_cast<const SymEngine::Integer>(trip)->as_int();
    return trip_int > 0 && (trip_int % static_cast<long long>(tile_size)) == 0;
}

} // namespace

LoopTiling::LoopTiling(structured_control_flow::StructuredLoop& loop, size_t tile_size, bool simplify_bounds)
    : loop_(loop), tile_size_(tile_size), simplify_bounds_(simplify_bounds) {};

std::string LoopTiling::name() const { return "LoopTiling"; };

bool LoopTiling::can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    if (this->tile_size_ <= 1) {
        return false;
    }
    return loop_.is_contiguous();
};

void LoopTiling::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
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

    // Whether the tile evenly divides this loop's (original) trip -- computed before tiling mutates it.
    // Only simplify when the caller opts in: dropping the guard changes the loop shape that later
    // passes (e.g. cooperative-copy vectorization) may depend on.
    bool drop_original_bound = simplify_bounds && tile_evenly_divides(loop, tile_size);

    // Step 1: Define new outer loop
    auto outer_indvar_str = builder.find_new_name(indvar->get_name() + "_tile");
    builder.add_container(outer_indvar_str, sdfg.type(loop.indvar()->get_name()));
    auto outer_indvar = symbolic::symbol(outer_indvar_str);
    auto outer_condition = symbolic::subs(loop.condition(), indvar, outer_indvar);
    auto outer_update = symbolic::add(outer_indvar, symbolic::integer(tile_size));

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
    auto inner_init = outer_indvar;
    auto inner_condition_tile = symbolic::Lt(inner_indvar, symbolic::add(outer_indvar, symbolic::integer(tile_size)));

    // Drop the redundant original bound for a perfectly dividing tile: the inner loop is then a clean
    // constant-trip tile that unrolls/vectorizes; otherwise keep the guard for the ragged remainder.
    symbolic::Condition inner_condition = drop_original_bound ? inner_condition_tile
                                                              : symbolic::And(inner_condition_tile, loop.condition());

    auto inner_update = symbolic::add(inner_indvar, symbolic::integer(1));
    builder.update_loop(loop, inner_indvar, inner_condition, inner_init, inner_update);

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
