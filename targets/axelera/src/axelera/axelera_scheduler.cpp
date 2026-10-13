#include "docc/target/axelera/axelera_scheduler.h"

#include <algorithm>
#include <vector>

#include <sdfg/analysis/loop_analysis.h>
#include <sdfg/structured_control_flow/block.h>
#include <sdfg/structured_control_flow/map.h>
#include <sdfg/structured_control_flow/reduce.h>
#include <sdfg/transformations/loop_tiling.h>

#include "docc/target/axelera/datapath.h"
#include "docc/target/axelera/schedule.h"

namespace sdfg {
namespace axelera {

namespace {

std::optional<int64_t> constant(const symbolic::Expression& expr) {
    if (expr.is_null() || !SymEngine::is_a<SymEngine::Integer>(*expr)) {
        return std::nullopt;
    }
    return SymEngine::down_cast<const SymEngine::Integer&>(*expr).as_int();
}

} // namespace

structured_control_flow::Map* flatten_dense_nest(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::Map& outer
) {
    std::vector<structured_control_flow::Map*> maps{&outer};
    while (maps.back()->root().size() == 1) {
        auto inner = dynamic_cast<structured_control_flow::Map*>(&maps.back()->root().at(0));
        if (!inner) {
            break;
        }
        maps.push_back(inner);
    }
    if (maps.size() < 2) {
        return nullptr;
    }

    std::vector<symbolic::Symbol> indvars;
    std::vector<int64_t> counts;
    for (auto* map : maps) {
        auto count = constant(map->num_iterations());
        if (constant(map->init()) != 0 || constant(map->stride()) != 1 || !count || *count <= 0) {
            return nullptr;
        }
        indvars.push_back(map->indvar());
        counts.push_back(*count);
    }
    symbolic::Expression linear = symbolic::zero();
    int64_t total = 1;
    for (size_t d = maps.size(); d-- > 0;) {
        linear = symbolic::add(linear, symbolic::mul(indvars[d], symbolic::integer(total)));
        total *= counts[d];
    }
    symbolic::ExpressionMapping origin, unit, relabel;
    for (auto& indvar : indvars) {
        origin[indvar] = symbolic::zero();
        unit[indvar] = symbolic::zero();
        relabel[indvar] = symbolic::zero();
    }
    unit[indvars.back()] = symbolic::one();

    // Only memlets may use the indvars, each index as an offset plus a multiple of the linear index.
    auto& body = maps.back()->root();
    for (size_t i = 0; i < body.size(); ++i) {
        auto block = dynamic_cast<structured_control_flow::Block*>(&body.at(i));
        if (!block || !block->dataflow().library_nodes().empty()) {
            return nullptr;
        }
        for (auto& node : block->dataflow().nodes()) {
            auto access = dynamic_cast<const data_flow::AccessNode*>(&node);
            if (access && std::ranges::any_of(indvars, [&](auto& indvar) {
                    return access->data() == indvar->get_name();
                })) {
                return nullptr;
            }
        }
        for (auto& memlet : block->dataflow().edges()) {
            for (auto& index : memlet.subset()) {
                auto step = symbolic::expand(symbolic::sub(index, symbolic::subs(index, origin)));
                auto multiple = symbolic::mul(symbolic::subs(step, unit), linear);
                if (!symbolic::eq(symbolic::expand(symbolic::sub(step, multiple)), symbolic::zero())) {
                    return nullptr;
                }
            }
        }
    }

    auto& parent = static_cast<structured_control_flow::Sequence&>(*outer.get_parent());
    auto name = builder.find_new_name(outer.indvar()->get_name() + "_flat");
    builder.add_container(name, builder.subject().type(outer.indvar()->get_name()));
    auto index = symbolic::symbol(name);
    auto& flat = builder.add_map_before(
        parent,
        outer,
        index,
        symbolic::Lt(index, symbolic::integer(total)),
        symbolic::zero(),
        symbolic::add(index, symbolic::one()),
        outer.schedule_type(),
        outer.debug_info()
    );
    builder.move_children(body, flat.root());
    relabel[indvars.back()] = index;
    flat.root().replace(relabel);
    builder.remove_child(parent, parent.index(outer));
    analysis_manager.invalidate_all();
    return &flat;
}

std::string datapath_obstacle(
    const StructuredSDFG& sdfg,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::StructuredLoop& loop,
    int64_t cores
) {
    auto reduce = dynamic_cast<structured_control_flow::Reduce*>(&loop);
    if (!dynamic_cast<structured_control_flow::Map*>(&loop) && !reduce) {
        return "it is neither a map nor a reduction";
    }
    if (!analysis_manager.get<analysis::LoopAnalysis>().children(&loop).empty()) {
        return "it contains loops";
    }
    auto count = constant(loop.num_iterations());
    if (constant(loop.init()) != 0 || constant(loop.stride()) != 1 || !count || *count <= 0) {
        return "it does not count from zero in unit steps for a fixed number of iterations";
    }
    if (*count % PWORD_LANES != 0) {
        return "its " + std::to_string(*count) + " iterations are not whole PWORDs of " + std::to_string(PWORD_LANES) +
               " lanes";
    }
    auto pwords = *count / static_cast<int64_t>(PWORD_LANES);
    if (cores < 1 || pwords % cores != 0 || (cores > 1 && pwords / cores < 2)) {
        return "its " + std::to_string(pwords) + " PWORDs do not split into chunks of at least two over " +
               std::to_string(cores) + " cores";
    }

    try {
        std::string accumulator;
        std::optional<DpuOp> combine;
        if (reduce) {
            auto kernel = match_reduction(sdfg, analysis_manager, *reduce);
            accumulator = kernel.accumulator;
            combine = kernel.combine;
        }
        auto plan = plan_datapath(sdfg, loop.root(), loop.indvar(), accumulator);
        if (plan.combine != combine) {
            return "its accumulator update does not match the reduction operation";
        }
        auto streams = plan.inputs;
        if (plan.output) {
            streams.push_back(*plan.output);
        }
        for (auto& stream : streams) {
            auto first = constant(symbolic::expand(symbolic::subs(stream.index, loop.indvar(), symbolic::zero())));
            if (!first || *first % static_cast<int64_t>(PWORD_LANES) != 0) {
                return "its access to " + stream.container + " is not PWORD-aligned";
            }
        }
    } catch (const std::runtime_error& e) {
        return e.what();
    }
    return "";
}

} // namespace axelera

namespace passes {
namespace scheduler {

SchedulerAction AxeleraScheduler::find(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::StructuredLoop& loop,
    bool offload_unknown_sizes
) {
    if (dyn_cast<structured_control_flow::Map*>(&loop) || dyn_cast<structured_control_flow::Reduce*>(&loop)) {
        return APPLY;
    }
    return CHILDREN;
}

SchedulerAction AxeleraScheduler::find(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::While& loop,
    bool offload_unknown_sizes
) {
    return NEXT;
}

void AxeleraScheduler::pre_schedule(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    std::vector<structured_control_flow::StructuredLoop*>& applicable_loops
) {
    for (auto& loop : applicable_loops) {
        if (auto map = dynamic_cast<structured_control_flow::Map*>(loop)) {
            if (auto flat = axelera::flatten_dense_nest(builder, analysis_manager, *map)) {
                loop = flat;
            }
        }
    }
}

bool AxeleraScheduler::can_apply_schedule(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::StructuredLoop& loop,
    bool offload_unknown_sizes
) {
    auto cores = analysis_manager.options().get(axelera::NUM_CORES, int64_t{1});
    if (!axelera::datapath_obstacle(builder.subject(), analysis_manager, loop, cores).empty()) {
        return false;
    }
    transformations::LoopTiling tiling(loop, axelera::PWORD_LANES);
    return tiling.can_be_applied(builder, analysis_manager);
}

void AxeleraScheduler::apply_schedule(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::StructuredLoop& loop,
    bool offload_unknown_sizes
) {
    using axelera::AxeleraLevel;
    using axelera::ScheduleType_Axelera;

    // LoopTiling needs unit-stride loops, so cut core chunks before PWORDs; `loop` stays the element loop.
    auto cores = analysis_manager.options().get(axelera::NUM_CORES, int64_t{1});
    if (cores > 1) {
        auto elements = SymEngine::down_cast<const SymEngine::Integer&>(*loop.num_iterations()).as_int();
        transformations::LoopTiling chunks(loop, elements / cores);
        chunks.apply(builder, analysis_manager);
        builder.update_schedule_type(*chunks.outer_loop(), ScheduleType_Axelera::create(AxeleraLevel::Core, cores));
    }

    transformations::LoopTiling lanes(loop, axelera::PWORD_LANES);
    lanes.apply(builder, analysis_manager);
    builder.update_schedule_type(*lanes.outer_loop(), ScheduleType_Axelera::create(AxeleraLevel::Stream));
    builder.update_schedule_type(*lanes.inner_loop(), ScheduleType_Axelera::create(AxeleraLevel::Lane));
}

std::unordered_set<ScheduleTypeCategory> AxeleraScheduler::compatible_types() {
    return {ScheduleTypeCategory::None};
}

} // namespace scheduler
} // namespace passes
} // namespace sdfg
