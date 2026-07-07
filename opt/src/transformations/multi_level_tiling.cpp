#include "sdfg/transformations/multi_level_tiling.h"

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace transformations {

MultiLevelTiling::MultiLevelTiling(structured_control_flow::StructuredLoop& loop, size_t tile_size, size_t tile_size_2)
    : LoopTiling(loop, tile_size), tile_size_2_(tile_size_2) {};

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

    auto& sdfg = builder.subject();

    // Now tile the inner (point) loop again with tile_size_2_
    auto& inner = *inner_loop_;
    auto inner_indvar2 = inner.indvar();

    auto middle_indvar_str = builder.find_new_name(inner_indvar2->get_name() + "_tile");
    builder.add_container(middle_indvar_str, sdfg.type(inner_indvar2->get_name()));

    auto middle_indvar = symbolic::symbol(middle_indvar_str);
    auto middle_condition = symbolic::subs(inner.condition(), inner_indvar2, middle_indvar);
    auto middle_update = symbolic::add(middle_indvar, symbolic::integer(this->tile_size_2_));

    auto parent2 = static_cast<structured_control_flow::Sequence*>(inner.get_parent());
    size_t index2 = parent2->index(inner);
    auto& transition2 = parent2->at(index2).second;

    structured_control_flow::StructuredLoop* middle_loop = nullptr;
    if (auto map = dyn_cast<structured_control_flow::Map*>(&inner)) {
        middle_loop = &builder.add_map_before(
            *parent2,
            inner,
            middle_indvar,
            middle_condition,
            inner.init(),
            middle_update,
            map->schedule_type(),
            transition2.assignments(),
            inner.debug_info()
        );
    } else if (auto reduce = dyn_cast<structured_control_flow::Reduce*>(&inner)) {
        middle_loop = &builder.add_reduce_before(
            *parent2,
            inner,
            middle_indvar,
            middle_condition,
            inner.init(),
            middle_update,
            reduce->reductions(),
            reduce->schedule_type(),
            transition2.assignments(),
            inner.debug_info()
        );
    } else {
        middle_loop = &builder.add_for_before(
            *parent2,
            inner,
            middle_indvar,
            middle_condition,
            inner.init(),
            middle_update,
            transition2.assignments(),
            inner.debug_info()
        );
    }

    // Redefine the innermost loop
    auto innermost_init = middle_indvar;
    auto innermost_condition_tile =
        symbolic::Lt(inner_indvar2, symbolic::add(middle_indvar, symbolic::integer(this->tile_size_2_)));
    symbolic::Condition innermost_condition = symbolic::And(innermost_condition_tile, inner.condition());
    auto innermost_update = symbolic::add(inner_indvar2, symbolic::integer(1));
    builder.update_loop(inner, inner_indvar2, innermost_condition, innermost_init, innermost_update);

    // Move inner loop into middle loop
    transition2.assignments().clear();
    builder.move_child(*parent2, index2 + 1, middle_loop->root());

    analysis_manager.invalidate_all();
    middle_loop_ = middle_loop;
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
    auto element = builder.find_element_by_id(loop_id);
    if (!element) {
        throw InvalidTransformationDescriptionException("Element with ID " + std::to_string(loop_id) + " not found.");
    }
    auto loop = dyn_cast<structured_control_flow::StructuredLoop*>(element);

    return MultiLevelTiling(*loop, tile_size, tile_size_2);
};

structured_control_flow::StructuredLoop* MultiLevelTiling::middle_loop() {
    if (!applied_) {
        throw InvalidSDFGException("Accessing tiled loop before their creation.");
    }
    return middle_loop_;
}

} // namespace transformations
} // namespace sdfg
