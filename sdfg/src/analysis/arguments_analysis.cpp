#include "sdfg/analysis/arguments_analysis.h"
#include "sdfg/analysis/mem_access_range_analysis.h"
#include "sdfg/analysis/type_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/codegen/utils.h"
#include "sdfg/helpers/helpers.h"
#include "sdfg/types/utils.h"

namespace sdfg {
namespace analysis {

void ArgumentsAnalysis::find_arguments_and_locals(
    analysis::AnalysisManager& analysis_manager, structured_control_flow::ControlFlowNode& node
) {
    auto& users = analysis_manager.get<analysis::Users>();
    analysis::UsersView scope_users(users, node);

    analysis::TypeAnalysis type_analysis(sdfg_, &node, analysis_manager);

    std::unordered_map<std::string, DataRwFlags> all_containers;
    for (auto& user : scope_users.uses()) {
        if (user->container() == symbolic::__nullptr__()->get_name()) {
            continue;
        }

        DataRwFlags meta;
        switch (user->use()) {
            case analysis::READ:
                meta.found_explicit_read();
                break;
            case analysis::WRITE:
                meta.found_explicit_write();
                break;
            case analysis::MOVE:
            case analysis::VIEW:
            default:
                meta.found_analysis_escape();
        }
        auto it = all_containers.insert({user->container(), meta});
        if (!it.second) {
            it.first->second.merge(meta);
        }
    }

    bool inferred_types = true;
    std::map<std::string, RegionArgument> arguments;
    std::unordered_set<std::string> locals;
    for (auto& [container, rwFlags] : all_containers) {
        bool is_scalar = false;
        bool is_ptr = false;

        auto type = type_analysis.get_outer_type(container);
        if (type == nullptr) {
            inferred_types = false;
            is_ptr = true;
            is_scalar = false;
        } else {
            // Unwrap Reference types to check the underlying type
            const types::IType* effective_type = type;
            if (type->type_id() == types::TypeID::Reference) {
                auto* reference = dynamic_cast<const codegen::Reference*>(type);
                assert(reference != nullptr);
                effective_type = &reference->reference_type();
            }
            is_scalar = effective_type->type_id() == types::TypeID::Scalar;
            is_ptr = effective_type->type_id() == types::TypeID::Pointer ||
                     effective_type->type_id() == types::TypeID::Array;
        }

        if (sdfg_.is_argument(container) || sdfg_.is_external(container)) {
            arguments.insert({container, {rwFlags, is_scalar, is_ptr}});
            continue;
        }

        size_t total_uses = users.uses(container).size();
        size_t scope_uses = scope_users.uses(container).size();

        if (scope_uses < total_uses) {
            arguments.insert({container, {rwFlags, is_scalar, is_ptr}});
        } else {
            locals.insert(container);
        }
    }

    node_arguments_.insert({&node, arguments});
    node_locals_.insert({&node, locals});
    node_inferred_types_.insert({&node, inferred_types});
}

void ArgumentsAnalysis::collect_arg_sizes(
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::ControlFlowNode& node,
    bool allow_dynamic_sizes_,
    bool do_not_throw
) {
    std::unordered_set<std::string> internal_vars;
    argument_sizes_.insert({&node, {}});
    argument_element_sizes_.insert({&node, {}});

    auto& mem_access_ranges = analysis_manager.get<analysis::MemAccessRanges>();

    auto arguments = this->arguments(analysis_manager, node);
    auto locals = this->locals(analysis_manager, node);

    internal_vars.insert(locals.begin(), locals.end());
    std::ranges::for_each(arguments, [&internal_vars](const auto& pair) { internal_vars.insert(pair.first); });

    analysis::TypeAnalysis type_analysis(sdfg_, &node, analysis_manager);

    for (auto& [argument, meta] : arguments) {
        if (!meta.is_scalar) {
            auto range = mem_access_ranges.get(argument, node, internal_vars);
            if (range == nullptr) {
                if (do_not_throw) {
                    known_sizes_.insert({&node, false});
                    return;
                } else {
                    throw std::runtime_error("Range not found for " + argument);
                }
            }
            auto base_type = type_analysis.get_outer_type(argument);
            auto elem_size = types::get_contiguous_element_size(*base_type, true);
            if (range->is_undefined()) {
                if (!allow_dynamic_sizes_) {
                    if (do_not_throw) {
                        known_sizes_.insert({&node, false});
                        return;
                    } else {
                        throw std::runtime_error("Argument " + argument + " has undefined range");
                    }
                }
                DEBUG_PRINTLN("Argument " << argument << " has undefined range, using malloc_usable_size");
                argument_sizes_.at(&node).insert({argument, symbolic::malloc_usable_size(symbolic::symbol(argument))});
                argument_element_sizes_.at(&node).insert({argument, elem_size});
                continue;
            }

            symbolic::Expression size = symbolic::one();
            if (!range->dims().empty()) {
                size = symbolic::add(range->dims().at(0).second, symbolic::one());
            }

            bool is_nested_type = true;
            auto peeled_type = types::peel_to_next_element(*base_type);
            while (is_nested_type) {
                if (peeled_type == nullptr) {
                    if (do_not_throw) {
                        known_sizes_.insert({&node, false});
                        return;
                    } else {
                        throw std::runtime_error("Could not infer type for argument: " + argument);
                    }
                }
                if (peeled_type->type_id() == types::TypeID::Array) {
                    auto array_type = dynamic_cast<const types::Array*>(peeled_type);
                    size = symbolic::mul(size, array_type->num_elements());
                    peeled_type = &array_type->element_type();
                } else if (peeled_type->type_id() == types::TypeID::Pointer) {
                    if (do_not_throw) {
                        known_sizes_.insert({&node, false});
                        return;
                    } else {
                        throw std::runtime_error("Non-contiguous pointer type: " + argument);
                    }
                } else {
                    is_nested_type = false;
                }
            }


            size = symbolic::mul(size, elem_size);
            DEBUG_PRINTLN("Size of " << argument << " is " << size->__str__());
            if (size.is_null()) {
                if (do_not_throw) {
                    known_sizes_.insert({&node, false});
                    return;
                } else {
                    throw std::runtime_error("Cannot figure out access size of " + argument);
                }
            } else {
                argument_sizes_.at(&node).insert({argument, size});
                argument_element_sizes_.at(&node).insert({argument, elem_size});
            }
        } else {
            auto type = type_analysis.get_outer_type(argument);
            if (type == nullptr) {
                if (do_not_throw) {
                    known_sizes_.insert({&node, false});
                    return;
                } else {
                    throw std::runtime_error("Could not infer type for argument: " + argument);
                }
            }
            DEBUG_PRINTLN("type of argument: " << type->print());
            auto size = types::get_contiguous_element_size(*type);
            DEBUG_PRINTLN("Size of " << argument << " is " << size->__str__());
            argument_sizes_.at(&node).insert({argument, size});
            argument_element_sizes_.at(&node).insert({argument, size});
        }
    }

    known_sizes_.insert({&node, true});
}

ArgumentsAnalysis::ArgumentsAnalysis(StructuredSDFG& sdfg) : Analysis(sdfg) {}

void ArgumentsAnalysis::run(analysis::AnalysisManager& analysis_manager) {}

const std::map<std::string, RegionArgument>& ArgumentsAnalysis::
    arguments(analysis::AnalysisManager& analysis_manager, structured_control_flow::ControlFlowNode& node) {
    if (node_arguments_.find(&node) != node_arguments_.end()) {
        return node_arguments_.at(&node);
    } else {
        find_arguments_and_locals(analysis_manager, node);
        return node_arguments_.at(&node);
    }
}

const std::unordered_set<std::string>& ArgumentsAnalysis::
    locals(analysis::AnalysisManager& analysis_manager, structured_control_flow::ControlFlowNode& node) {
    if (node_locals_.find(&node) != node_locals_.end()) {
        return node_locals_.at(&node);
    } else {
        find_arguments_and_locals(analysis_manager, node);
        return node_locals_.at(&node);
    }
}

bool ArgumentsAnalysis::
    inferred_types(analysis::AnalysisManager& analysis_manager, structured_control_flow::ControlFlowNode& node) {
    if (node_inferred_types_.find(&node) != node_inferred_types_.end()) {
        return node_inferred_types_.at(&node);
    } else {
        find_arguments_and_locals(analysis_manager, node);
        return node_inferred_types_.at(&node);
    }
}

const std::unordered_map<std::string, symbolic::Expression>& ArgumentsAnalysis::argument_sizes(
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::ControlFlowNode& node,
    bool allow_dynamic_sizes_
) {
    if (argument_sizes_.find(&node) != argument_sizes_.end() && known_sizes_.at(&node)) {
        return argument_sizes_.at(&node);
    } else {
        collect_arg_sizes(analysis_manager, node, allow_dynamic_sizes_, false);
        return argument_sizes_.at(&node);
    }
}

const std::unordered_map<std::string, symbolic::Expression>& ArgumentsAnalysis::argument_element_sizes(
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::ControlFlowNode& node,
    bool allow_dynamic_sizes_
) {
    if (argument_element_sizes_.find(&node) != argument_element_sizes_.end() && known_sizes_.at(&node)) {
        return argument_element_sizes_.at(&node);
    } else {
        collect_arg_sizes(analysis_manager, node, allow_dynamic_sizes_, false);
        return argument_element_sizes_.at(&node);
    }
}

bool ArgumentsAnalysis::argument_size_known(
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::ControlFlowNode& node,
    bool allow_dynamic_sizes_
) {
    if (known_sizes_.find(&node) != known_sizes_.end()) {
        return known_sizes_.at(&node);
    } else {
        collect_arg_sizes(analysis_manager, node, allow_dynamic_sizes_, true);
        return known_sizes_.at(&node);
    }
}

} // namespace analysis
} // namespace sdfg
