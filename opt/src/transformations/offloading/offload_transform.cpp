#include "sdfg/transformations/offloading/offload_transform.h"

#include <map>
#include <string>

#include "sdfg/analysis/mem_access_range_analysis.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/analysis/type_analysis.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/symbolic/symbolic.h"

#include "sdfg/optimization_report/pass_report_consumer.h"
#include "sdfg/types/utils.h"
#include "symengine/symengine_rcp.h"

namespace sdfg {
namespace transformations {

OffloadTransform::OffloadTransform(structured_control_flow::Map& map, bool allow_dynamic_sizes)
    : map_(map), allow_dynamic_sizes_(allow_dynamic_sizes) {}

bool OffloadTransform::can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& sdfg = builder.subject();

    auto& arguments_analysis = analysis_manager.get<analysis::ArgumentsAnalysis>();

    if (!arguments_analysis.inferred_types(analysis_manager, this->map_)) {
        if (report_) report_->transform_impossible(this, "unranged args");
        DEBUG_PRINTLN("Cannot apply transform: argument types not inferred");
        return false;
    }
    auto& arguments = arguments_analysis.arguments(analysis_manager, this->map_);

    // Criterion: arg Data Types must be continuous
    for (auto& [argument, meta] : arguments) {
        auto base_type = analysis::TypeAnalysis(sdfg, &map_, analysis_manager).get_outer_type(argument);
        if (base_type == nullptr) {
            if (report_) report_->transform_impossible(this, "cannot infer type");
            DEBUG_PRINTLN("Cannot apply transform: argument type cannot be inferred");
            return false;
        }
        if (!types::is_contiguous_type(*base_type, sdfg)) {
            if (report_) report_->transform_impossible(this, "type is not contiguous");
            DEBUG_PRINTLN("Cannot apply transform: argument type is not contiguous");
            return false;
        }
        if (meta.is_scalar && meta.is_output) {
            if (report_) report_->transform_impossible(this, "scalar output");
            DEBUG_PRINTLN("Cannot apply transform: map writes to scalar argument");
            return false;
        }
    }

    // Criterion: Map must start at 0
    if (!symbolic::eq(this->map_.init(), symbolic::zero())) {
        if (report_) report_->transform_impossible(this, "non zero start");
        DEBUG_PRINTLN("Cannot apply transform: map does not start at zero");
        return false;
    }

    // Criterion: Map cannot write to scalar arguments
    for (auto& [argument, meta] : arguments) {
        if (meta.is_scalar && meta.is_output) {
            if (report_) report_->transform_impossible(this, "scalar output");
            DEBUG_PRINTLN("Cannot apply transform: map writes to scalar argument");
            return false;
        }
    }

    // Criterion: arg ranges must be known
    auto& mem_access_ranges = analysis_manager.get<analysis::MemAccessRanges>();

    if (!arguments_analysis.argument_size_known(analysis_manager, this->map_, allow_dynamic_sizes_)) {
        if (report_) report_->transform_impossible(this, "args not understood");
        DEBUG_PRINTLN("Cannot apply transform: argument sizes not known");
        return false;
    }

    if (report_) report_->transform_possible(this);
    return true;
}

void OffloadTransform::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    // Schedule
    builder.update_schedule_type(this->map_, transformed_schedule_type());

    auto& sdfg = builder.subject();

    // Identify arguments and locals
    auto& arguments_analysis = analysis_manager.get<analysis::ArgumentsAnalysis>();

    auto& arguments = arguments_analysis.arguments(analysis_manager, this->map_);
    auto& locals = arguments_analysis.locals(analysis_manager, this->map_);

    // Infer subsets for arguments
    auto& mem_access_ranges = analysis_manager.get<analysis::MemAccessRanges>();
    auto& argument_sizes = arguments_analysis.argument_sizes(analysis_manager, this->map_, allow_dynamic_sizes_);

    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto parent_scope = static_cast<structured_control_flow::Sequence*>(scope_analysis.parent_scope(&this->map_));

    std::string container_prefix = copy_prefix() + std::to_string(parent_scope->element_id()) + "_";

    // Allocate arguments and locals
    allocate_locals_on_device_stack(builder, analysis_manager, locals);
    handle_device_setup_and_teardown(builder, arguments, argument_sizes, container_prefix);

    // Copy-in arguments to device memory & allocation
    for (auto& [argument, meta] : arguments) {
        if (!meta.is_ptr) {
            continue;
        }
        auto argument_device = container_prefix + argument;
        auto& new_block = builder.add_block_before(*parent_scope, this->map_, {}, this->map_.debug_info());
        auto& size = argument_sizes.at(argument);
        copy_to_device_with_allocation(builder, argument, argument_device, size, SymEngine::null, new_block);
    }

    update_map_containers(arguments, container_prefix);

    // Copy-out arguments to host memory & free
    for (auto& [argument, meta] : arguments) {
        if (!meta.is_ptr) {
            continue;
        }
        auto argument_device = container_prefix + argument;
        auto& new_block = builder.add_block_after(*parent_scope, this->map_, {}, this->map_.debug_info());
        auto& size = argument_sizes.at(argument);
        if (meta.is_output) {
            copy_from_device_with_free(builder, new_block, argument, argument_device, size, SymEngine::null);
        } else {
            deallocate_device_arg(builder, new_block, argument_device, size, SymEngine::null);
        }
    }

    if (report_) report_->transform_applied(this);
}

void OffloadTransform::handle_device_setup_and_teardown(
    builder::StructuredSDFGBuilder& builder,

    const std::map<std::string, analysis::RegionArgument>& arguments,
    const std::unordered_map<std::string, symbolic::Expression>& argument_sizes,
    std::string prefix
) {
    // Add managed buffers for pointer arguments
    for (auto& [argument, meta] : arguments) {
        if (!meta.is_ptr || builder.subject().exists(prefix + argument)) {
            continue;
        }
        auto argument_device = prefix + argument;

        auto arg_size = argument_sizes.at(argument);

        add_device_buffer(builder, argument, argument_device, arg_size);
    }
}

void OffloadTransform::
    update_map_containers(const std::map<std::string, analysis::RegionArgument>& arguments, std::string prefix) {
    for (auto& [argument, meta] : arguments) {
        if (meta.is_ptr) {
            auto argument_device = prefix + argument;
            this->map_.replace(symbolic::symbol(argument), symbolic::symbol(argument_device));
        }
    }
}

} // namespace transformations
} // namespace sdfg
