#include "sdfg/structured_control_flow/reduce.h"

#include <string>

#include "sdfg/exceptions.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace structured_control_flow {

std::string reduction_operation_to_string(ReductionOperation op) {
    switch (op) {
        case ReductionOperation::Add:
            return "add";
        case ReductionOperation::Mul:
            return "mul";
        case ReductionOperation::Min:
            return "min";
        case ReductionOperation::Max:
            return "max";
    }
    throw InvalidSDFGException("Reduce: unknown reduction operation");
}

ReductionOperation reduction_operation_from_string(const std::string& value) {
    if (value == "add") {
        return ReductionOperation::Add;
    } else if (value == "mul") {
        return ReductionOperation::Mul;
    } else if (value == "min") {
        return ReductionOperation::Min;
    } else if (value == "max") {
        return ReductionOperation::Max;
    }
    throw InvalidSDFGException("Reduce: unknown reduction operation '" + value + "'");
}

Reduce::Reduce(
    size_t element_id,
    const DebugInfo& debug_info,
    ControlFlowNode* parent,
    symbolic::Symbol indvar,
    symbolic::Expression init,
    symbolic::Expression update,
    symbolic::Condition condition,
    std::vector<ReductionInfo> reductions,
    const ScheduleType& schedule_type
)
    : StructuredLoop(element_id, debug_info, parent, indvar, init, update, condition, schedule_type),
      reductions_(std::move(reductions)) {};

bool Reduce::accept(visitor::ActualStructuredSDFGVisitor& visitor) { return visitor.visit(*this); }

void Reduce::validate(const Function& function) const { StructuredLoop::validate(function); };

void Reduce::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    StructuredLoop::replace(old_expression, new_expression);

    if (SymEngine::is_a<SymEngine::Symbol>(*old_expression) && SymEngine::is_a<SymEngine::Symbol>(*new_expression)) {
        const auto& old_name = SymEngine::rcp_static_cast<const SymEngine::Symbol>(old_expression)->get_name();
        const auto& new_name = SymEngine::rcp_static_cast<const SymEngine::Symbol>(new_expression)->get_name();
        for (auto& reduction : this->reductions_) {
            if (reduction.container == old_name) {
                reduction.container = new_name;
            }
        }
    }
}

void Reduce::replace(const symbolic::ExpressionMapping& replacements) {
    StructuredLoop::replace(replacements);

    for (auto& reduction : this->reductions_) {
        auto it = replacements.find(symbolic::symbol(reduction.container));
        if (it != replacements.end() && SymEngine::is_a<SymEngine::Symbol>(*it->second)) {
            reduction.container = SymEngine::rcp_static_cast<const SymEngine::Symbol>(it->second)->get_name();
        }
    }
}

const std::vector<ReductionInfo>& Reduce::reductions() const { return this->reductions_; };

void Reduce::replace_reduction_container(const std::string& old_name, const std::string& new_name) {
    if (old_name == new_name) {
        return;
    }
    for (auto& reduction : this->reductions_) {
        if (reduction.container == old_name) {
            reduction.container = new_name;
        }
    }
};

} // namespace structured_control_flow
} // namespace sdfg
