#include "sdfg/codegen/instrumentation/arg_capture_plan.h"

#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/arguments_analysis.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/analysis/type_analysis.h"
#include "sdfg/codegen/language_extension.h"
#include "sdfg/helpers/helpers.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/utils.h"

namespace sdfg {
namespace codegen {

CaptureVarPlan::CaptureVarPlan(
    bool capture_input,
    bool capture_output,
    int argIdx,
    bool isExternal,
    sdfg::types::PrimitiveType innerType,
    const sdfg::symbolic::Expression size,
    bool isScalar
)
    : capture_input(capture_input), capture_output(capture_output), arg_idx(argIdx), is_external(isExternal),
      inner_type(innerType), size(size), is_scalar(isScalar) {}

bool ArgCapturePlan::should_instrument(const structured_control_flow::ControlFlowNode& node) const {
    return this->nodes_.count(&node);
}

void ArgCapturePlan::begin_instrumentation(
    const structured_control_flow::ControlFlowNode& node, PrettyPrinter& stream, LanguageExtension& language_extension
) const {
    stream << "const bool __daisy_cap_en_" << node.element_id() << " = __daisy_capture_enter(__capture_ctx, "
           << node.element_id() << ");" << std::endl;

    stream << "if (__daisy_cap_en_" << node.element_id() << ")";
    stream << "{" << std::endl;

    auto& node_plan = this->nodes_.at(&node);
    this->emit_arg_captures(stream, language_extension, node_plan, false, std::to_string(node.element_id()));

    stream << "}" << std::endl;
}

void ArgCapturePlan::end_instrumentation(
    const structured_control_flow::ControlFlowNode& node, PrettyPrinter& stream, LanguageExtension& language_extension
) const {
    stream << "if (__daisy_cap_en_" << node.element_id() << ")";
    stream << "{" << std::endl;

    auto& node_plan = this->nodes_.at(&node);
    this->emit_arg_captures(stream, language_extension, node_plan, true, std::to_string(node.element_id()));

    stream << "}" << std::endl;

    stream << "__daisy_capture_end(__capture_ctx);" << std::endl;
}

std::unique_ptr<ArgCapturePlan> ArgCapturePlan::none(StructuredSDFG& sdfg) {
    return std::make_unique<ArgCapturePlan>(
        sdfg,
        std::unordered_map<
            const structured_control_flow::ControlFlowNode*,
            std::unordered_map<std::string, CaptureVarPlan>>{}
    );
}

std::unique_ptr<ArgCapturePlan> ArgCapturePlan::root(StructuredSDFG& sdfg) {
    analysis::AnalysisManager analysis_manager(sdfg);
    std::unordered_map<const structured_control_flow::ControlFlowNode*, std::unordered_map<std::string, CaptureVarPlan>>
        nodes;
    auto root_plan = create_capture_plan(sdfg, analysis_manager, sdfg.root());
    nodes.insert({&sdfg.root(), root_plan});
    return std::make_unique<ArgCapturePlan>(sdfg, nodes);
}

std::unique_ptr<ArgCapturePlan> ArgCapturePlan::outermost_loops_plan(StructuredSDFG& sdfg) {
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& loop_tree_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    auto ols = loop_tree_analysis.outermost_loops();

    std::unordered_map<const structured_control_flow::ControlFlowNode*, std::unordered_map<std::string, CaptureVarPlan>>
        nodes;
    for (size_t i = 0; i < ols.size(); i++) {
        auto& loop = ols[i];
        auto loop_plan = create_capture_plan(sdfg, analysis_manager, *loop);
        nodes.insert({loop, loop_plan});
    }

    DEBUG_PRINTLN("Created arg capture plan for " + std::to_string(nodes.size()) + " nodes.");

    return std::make_unique<ArgCapturePlan>(sdfg, nodes);
}

void ArgCapturePlan::emit_arg_captures(
    PrettyPrinter& stream,
    LanguageExtension& language_extension,
    const std::unordered_map<std::string, CaptureVarPlan>& plan,
    bool after,
    std::string element_id
) const {
    auto afterBoolStr = after ? "true" : "false";
    for (auto& [argName, varPlan] : plan) {
        auto argIdx = varPlan.arg_idx;
        if ((!after && varPlan.capture_input) || (after && varPlan.capture_output)) {
            std::string safe_name;
            symbolic::Expression size = varPlan.size;
            if (varPlan.is_external) {
                safe_name = language_extension.external_prefix() + argName;
                size = symbolic::subs(
                    size, symbolic::symbol(argName), symbolic::symbol(language_extension.external_prefix() + argName)
                );
            } else {
                safe_name = argName;
            }
            DEBUG_PRINTLN("size: " << size->__str__());
            stream << "\t__daisy_capture_raw(" << "__capture_ctx, " << argIdx << ", "
                   << (varPlan.is_scalar || varPlan.is_external ? "&" : "") << safe_name << ", "
                   << language_extension.expression(size) << ", " << varPlan.inner_type << ", " << afterBoolStr << ", "
                   << element_id << ");" << std::endl;
        }
    }
}

bool ArgCapturePlan::add_capture_plan(
    StructuredSDFG& sdfg,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::ControlFlowNode& node,
    const std::string& var_name,
    analysis::RegionArgument region_arg,
    int arg_idx,
    bool is_external,
    std::unordered_map<std::string, CaptureVarPlan>& plan
) {
    analysis::TypeAnalysis type_analysis(sdfg, &node, analysis_manager);
    auto type = type_analysis.get_outer_type(var_name);
    if (type == nullptr) {
        DEBUG_PRINTLN("Could not determine type for variable " + var_name + ", cannot add to capture plan.");
        return false;
    }

    auto& innermost_type = types::peel_to_innermost_element(*type);

    types::PrimitiveType inner_type = innermost_type.primitive_type();

    if (inner_type == types::Void) {
        return false;
    }

    auto& arguments_analysis = analysis_manager.get<analysis::ArgumentsAnalysis>();
    auto arg_sizes = arguments_analysis.argument_sizes(analysis_manager, node, true);
    if (arg_sizes.find(var_name) == arg_sizes.end() && !region_arg.is_scalar) {
        DEBUG_PRINTLN("Could not determine size for variable " + var_name + ", cannot add to capture plan.");
        return false;
    }
    auto size = arg_sizes.at(var_name);

    plan.insert(
        {var_name,
         CaptureVarPlan(
             region_arg.is_input, region_arg.is_output, arg_idx, is_external, inner_type, size, region_arg.is_scalar
         )}
    );

    DEBUG_PRINTLN("Successfully added capture plan for variable " + var_name);
    return true;
}

std::unordered_map<std::string, CaptureVarPlan> ArgCapturePlan::create_capture_plan(
    StructuredSDFG& sdfg, analysis::AnalysisManager& analysis_manager, structured_control_flow::ControlFlowNode& node
) {
    if (auto loop_node = dyn_cast<structured_control_flow::StructuredLoop*>(&node)) {
        DEBUG_PRINTLN("Creating capture plan for loop node " << loop_node->indvar()->__str__());
    } else {
        DEBUG_PRINTLN("Creating capture plan for node " << node.element_id());
    }
    auto& arguments_analysis = analysis_manager.get<analysis::ArgumentsAnalysis>();
    if (!arguments_analysis.inferred_types(analysis_manager, node)) {
        DEBUG_PRINTLN(
            "Could not create capture plan for node " << node.element_id()
                                                      << " because argument types could not be inferred."
        );
        return {};
    }

    if (!arguments_analysis.argument_size_known(analysis_manager, node, true)) {
        DEBUG_PRINTLN(
            "Could not create capture plan for node " << node.element_id() << " because argument sizes are not known."
        );
        return {};
    }

    auto& arguments = arguments_analysis.arguments(analysis_manager, node);
    analysis::TypeAnalysis type_analysis(sdfg, &node, analysis_manager);

    for (auto argument : arguments) {
        auto arg_type = type_analysis.get_outer_type(argument.first);
        if (!types::is_contiguous_type(*arg_type, sdfg)) {
            DEBUG_PRINTLN(
                "Could not create capture plan for node " << node.element_id() << " because argument " << argument.first
                                                          << " is not contiguous."
            );
            return {};
        }
    }

    // Sort arguments to have a deterministic order
    std::vector<std::string> args;
    for (auto& [arg_name, flags] : arguments) {
        args.push_back(arg_name);
    }
    std::sort(args.begin(), args.end());
    DEBUG_PRINTLN("Found " + std::to_string(args.size()) + " arguments for region " + std::to_string(node.element_id()));

    bool working = true;
    int arg_idx = 0;
    std::unordered_map<std::string, CaptureVarPlan> plan;
    for (auto& arg_name : args) {
        if (sdfg.type(arg_name).type_id() == types::TypeID::Function) {
            continue;
        }
        bool external = false;
        if (sdfg.is_external(arg_name)) {
            external = true;
        }

        working &=
            add_capture_plan(sdfg, analysis_manager, node, arg_name, arguments.at(arg_name), arg_idx, external, plan);
        ++arg_idx;
    }
    if (!working) {
        DEBUG_PRINTLN("could not create capture plan, returning empty plan");
        return std::unordered_map<std::string, CaptureVarPlan>{};
    }

    std::cout << "Created capture plan for " << plan.size() << " arguments." << std::endl;
    return plan;
}

} // namespace codegen
} // namespace sdfg
