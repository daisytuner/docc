#include <utility>

#include "docc/target/tenstorrent/tenstorrent_transform.h"

#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/analysis/mem_access_range_analysis.h"
#include "sdfg/analysis/type_analysis.h"
#include "sdfg/analysis/users.h"

#include "sdfg/types/type.h"
#include "sdfg/types/utils.h"

#include "sdfg/transformations/loop_tiling.h"

namespace sdfg::tenstorrent {

std::string TenstorrentTransform::name() const { return "TenstorrentTransform"; }

void TenstorrentTransform::setup_device(builder::StructuredSDFGBuilder& builder, Block& global_alloc_block) {
    auto& sdfg = builder.subject();

    auto& block = builder.add_block_before(sdfg.root(), global_alloc_block, {}, {});
}

void TenstorrentTransform::teardown_device(builder::StructuredSDFGBuilder& builder, Block& global_alloc_block) {}

bool has_no_nested_loops(const structured_control_flow::ControlFlowNode& root) {
    // std::unordered_set<const data_flow::Tasklet*> tasklets;
    std::list<const structured_control_flow::ControlFlowNode*> queue = {&root};
    while (!queue.empty()) {
        auto node = queue.front();
        queue.pop_front();

        if (auto block = dynamic_cast<const structured_control_flow::Block*>(node)) {
            // for (auto& child : block->dataflow().nodes()) {
            //     if (auto tasklet = dynamic_cast<const data_flow::Tasklet*>(&child)) {
            //         tasklets.insert(tasklet);
            //     }
            // }
        } else if (auto sequence = dynamic_cast<const structured_control_flow::Sequence*>(node)) {
            for (size_t i = 0; i < sequence->size(); i++) {
                if (sequence->at(i).second.assignments().size() > 0) {
                    return false;
                }
                queue.push_back(&sequence->at(i).first);
            }
        } else {
            return false;
        }
    }
    return true;
}

void TenstorrentTransform::add_device_buffer(
    builder::StructuredSDFGBuilder& builder,
    std::string host_arg_name,
    std::string device_arg_name,
    symbolic::Expression arg_size
) {
    // TODO cannot model shared_ptr as ptr. But that is the actual return type of CreateBuffer()
    auto type = types::Structure("std::shared_ptr<tt::tt_metal::Buffer>");
    type.storage_type(global_device_storage_type(arg_size));
    builder.add_container(device_arg_name, type);
}

bool TenstorrentTransform::
    can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto plan = try_create_transform_plan(builder, analysis_manager);

    return !!plan;
}

std::unique_ptr<TransformPlan> TenstorrentTransform::
    try_create_transform_plan(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto stride = analysis::LoopAnalysis::stride(&map_);
    if (!symbolic::eq(stride, symbolic::one())) { // map stride must be 1 for convenient tiling
        if (report_) report_->transform_impossible(this, "non-1 stride");
        return {};
    }

    auto condition = map_.condition();
    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    auto bound = analysis::LoopAnalysis::canonical_bound(&map_, assumptions_analysis);
    if (bound == SymEngine::null) {
        if (report_) report_->transform_impossible(this, "upper bound unknown");
        return {};
    }

    // Criterion: Map must start at 0
    if (!symbolic::eq(this->map_.init(), symbolic::zero())) {
        if (report_) report_->transform_impossible(this, "non zero start");
        return {};
    }

    auto num_iterations = symbolic::div(bound, stride);
    num_iterations = symbolic::sub(num_iterations, map_.init());
    DEBUG_PRINTLN(
        "TTT: " << builder.subject().name() << ":n" << map_.element_id()
                << ": num strides: " << num_iterations->__str__()
    );

    auto& sdfg = builder.subject();
    auto& arguments_analysis = analysis_manager.get<analysis::ArgumentsAnalysis>();

    if (!arguments_analysis.inferred_types(analysis_manager, this->map_)) {
        if (report_) report_->transform_impossible(this, "confusing arg types");
        return {};
    }

    auto& arguments = arguments_analysis.arguments(analysis_manager, this->map_);

    // Criterion: arg Data Types must be continuous
    for (auto& [argument, meta] : arguments) {
        auto base_type = analysis::TypeAnalysis(sdfg, &map_, analysis_manager).get_outer_type(argument);
        if (base_type == nullptr) {
            if (report_) report_->transform_impossible(this, "cannot infer type");
            return {};
        }
        if (!types::is_contiguous_type(*base_type, sdfg)) {
            if (report_) report_->transform_impossible(this, "type is not contiguous");
            return {};
        }
        if (meta.is_scalar && meta.is_output) {
            if (report_) report_->transform_impossible(this, "scalar output");
            return {};
        }
    }

    // Criterion: Map cannot write to scalar arguments
    for (auto& [argument, meta] : arguments) {
        if (meta.is_scalar && meta.is_output) {
            if (report_) report_->transform_impossible(this, "scalar output");
            return {};
        }
    }

    auto& mem_access_ranges = analysis_manager.get<analysis::MemAccessRanges>();

    if (!arguments_analysis.argument_size_known(analysis_manager, this->map_, allow_dynamic_sizes_)) {
        if (report_) report_->transform_impossible(this, "transfer args not sized");
        return {};
    }

    std::unordered_map<std::string, symbolic::Expression> argument_sizes =
        arguments_analysis.argument_sizes(analysis_manager, this->map_, allow_dynamic_sizes_);

    auto& users = analysis_manager.get<analysis::Users>();
    analysis::UsersView scope_users(users, map_);

    std::optional<symbolic::Expression> input_ratio;
    std::optional<symbolic::Expression> output_ratio;
    std::unordered_map<std::string, int> arg_dims;
    for (auto& [arg, meta] : arguments) {
        if (meta.is_ptr) {
            auto& size = argument_sizes.at(arg);
            if (meta.is_input) {
                //                auto ratio = symbolic::div(num_iterations, size);
                //                if (!input_ratio.has_value()) {
                //                    input_ratio = ratio;
                //                } else if (!symbolic::eq(*input_ratio, ratio)) {
                //                    DEBUG_PRINTLN(
                //                        "TTT: " << builder.subject().name() << ":n" << map_.element_id() << ": arg "
                //                        << arg
                //                                << " ratio '" << ratio->__str__() << "' does match other in ratio '"
                //                                << input_ratio.value()->__str__() << "' between input sizes and map
                //                                iterations '"
                //                                << num_iterations->__str__() << "'"
                //                    );
                //                    return {};
                //                }
            }
            if (meta.is_output) {
                //                auto ratio = symbolic::div(num_iterations, size);
                //                if (!output_ratio.has_value()) {
                //                    output_ratio = ratio;
                //                } else if (!symbolic::eq(*output_ratio, ratio)) {
                //                    DEBUG_PRINTLN(
                //                        "TTT: " << builder.subject().name() << ":n" << map_.element_id() << ": arg "
                //                        << arg
                //                                << " ratio '" << ratio->__str__() << "' does match other out ratio '"
                //                                << output_ratio.value()->__str__() << "' between output sizes and map
                //                                iterations '"
                //                                << num_iterations->__str__() << "'"
                //                    );
                //                    return {};
                //                }
            }
            if (meta.is_input || meta.is_output) { // far too restrictive check. Full analysis requires reliable range
                                                   // analysis to find loop-invariant, quasi-scalar ranges and identify
                                                   // the various options for tile-sizes etc.
                for (auto* user : scope_users.uses(arg)) {
                    switch (user->use()) {
                        case analysis::MOVE:
                        case analysis::VIEW:
                            if (report_) report_->transform_impossible(this, "use of arg " + arg + " too complex");
                            return {};
                        default:
                            break;
                    }
                    for (auto& subset : user->subsets()) {
                        auto& outermost_idx = subset.at(0);
                        if (!symbolic::eq(outermost_idx, map_.indvar())) {
                            if (report_) {
                                std::stringstream ss;
                                ss << "use of arg " << arg << " with subset " << subset << " not indVar";
                                report_->transform_impossible(this, ss.str());
                            }
                            return {};
                        }
                        auto dims = static_cast<int>(subset.size());
                        auto prev_dims = arg_dims[arg];
                        if (dims > prev_dims) {
                            arg_dims[arg] = dims;
                        }
                    }
                }
            }
        }
    }

    auto& type_analysis = analysis_manager.get<analysis::TypeAnalysis>();

    auto plan_ptr = std::make_unique<TransformPlan>();
    auto& plan = *plan_ptr.get();

    plan.tile_entries_ = 1024;
    auto tile_entries = symbolic::integer(plan.tile_entries_);

    for (auto& [arg, meta] : arguments) {
        if (meta.is_ptr) {
            auto size = argument_sizes.at(arg);
            auto type = type_analysis.get_outer_type(arg);
            if (!type) {
                if (report_) {
                    report_->transform_impossible(this, "use of arg " + arg + " has unknown type");
                }
                return {};
            }
            if (type->primitive_type() != types::Int32 && type->primitive_type() != types::UInt32 &&
                type->primitive_type() != types::Float) {
                if (report_) {
                    report_->transform_impossible(this, "use of arg " + arg + " has unsupported cb type");
                }
                return {};
            }
            auto dims = arg_dims[arg];
            if (dims > 1) {
                if (report_)
                    report_
                        ->transform_impossible(this, arg + " is unsupported multi-dim (" + std::to_string(dims) + ")");
                return {};
            }
            auto page_size = symbolic::mul(tile_entries, types::get_contiguous_element_size(*type));
            plan.transferred_arguments_.emplace_back(arg, *type, size, page_size, meta);
        } else {
            auto type = type_analysis.get_outer_type(arg);
            if (type->type_id() != types::TypeID::Scalar) {
                if (report_) report_->transform_impossible(this, "use of arg " + arg + " is not a scalar");
                return {};
            }
            if (types::bit_width(type->primitive_type()) > 32) {
                if (report_) report_->transform_impossible(this, "use of arg " + arg + " does not map to uint32_t");
                return {};
            }
            plan.scalar_args_.emplace_back(arg, *type);
        }
    }

    auto& locals = arguments_analysis.locals(analysis_manager, this->map_);
    for (auto& local : locals) {
        plan.locals_.emplace_back(local);
    }

    // TODO check if no input is WAY smaller than this


    if (report_) report_->transform_possible(this);
    return std::move(plan_ptr);
}

void TenstorrentTransform::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto plan = try_create_transform_plan(builder, analysis_manager);

    apply_plan(builder, analysis_manager, std::move(plan));
}

bool TenstorrentTransform::try_apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto plan = try_create_transform_plan(builder, analysis_manager);

    if (!plan) {
        return false;
    }

    apply_plan(builder, analysis_manager, std::move(plan));
    return true;
}

void TenstorrentTransform::apply_plan(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    std::unique_ptr<TransformPlan> plan
) {
    transformations::LoopTiling tiler(map_, plan->tile_entries_);
    assert(tiler.can_be_applied(builder, analysis_manager) && "Cannot apply tiling");

    auto& parent_scope = require_parent_scope();

    create_offloaded_memory_handling(plan->transferred_arguments_);

    auto schedType = ScheduleType_Tenstorrent_Device::create();
    if (force_synchronous_) {
        ScheduleType_Tenstorrent_Device::set_blocking(schedType, true);
    }

    builder.update_schedule_type(map_, schedType); // will be copied by tiler to outer
                                                   // loop
    auto outer_map_idx = parent_scope.index(map_);

    tiler.apply(builder, analysis_manager);

    builder.update_schedule_type(map_, ScheduleType_Tenstorrent_Kernel::create()); // override the inner / original
                                                                                   // loops type

    allocate_locals_on_device_stack(builder, analysis_manager, plan->locals_);
    auto& outer_map = dynamic_cast<structured_control_flow::Map&>(parent_scope.at(outer_map_idx).first);

    builder.subject().type(outer_map.indvar()->get_name()).storage_type() = local_device_storage_type();
    if (report_) report_->transform_applied(this);
}

void TenstorrentTransform::set_report(sdfg::PassReportConsumer* report) {
    TenstorrentOffloadingExpansion::set_report(report);
    transformations::Transformation::set_report(report);
}


} // namespace sdfg::tenstorrent
