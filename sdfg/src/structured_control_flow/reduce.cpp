#include "sdfg/structured_control_flow/reduce.h"

#include <string>

#include "sdfg/exceptions.h"
#include "sdfg/function.h"
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

std::string rescale_base_to_string(RescaleBase base) {
    switch (base) {
        case RescaleBase::E:
            return "e";
        case RescaleBase::Two:
            return "2";
    }
    throw InvalidSDFGException("Reduce: unknown rescale base");
}

RescaleBase rescale_base_from_string(const std::string& value) {
    if (value == "e") {
        return RescaleBase::E;
    } else if (value == "2") {
        return RescaleBase::Two;
    }
    throw InvalidSDFGException("Reduce: unknown rescale base '" + value + "'");
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

bool Reduce::accept(visitor::ActualStructuredSDFGVisitor& visitor) {
    return visitor.visit(*this);
}

void Reduce::validate(const Function& function) const {
    StructuredLoop::validate(function);
    for (const auto& reduction : reductions_) {
        if (reduction.rescale.empty()) {
            continue;
        }
        if (reduction.operation != ReductionOperation::Add) {
            throw InvalidSDFGException("Reduce: only an add reduction can be rescaled ('" + reduction.container + "')");
        }
        const ReductionInfo* anchor = nullptr;
        for (const auto& other : reductions_) {
            if (other.container == reduction.rescale) {
                anchor = &other;
            }
        }
        if (anchor == nullptr || anchor == &reduction) {
            throw InvalidSDFGException(
                "Reduce: rescale anchor '" + reduction.rescale + "' of '" + reduction.container +
                "' is not another reduction of this loop"
            );
        }
        if (anchor->operation != ReductionOperation::Max) {
            throw InvalidSDFGException("Reduce: rescale anchor '" + reduction.rescale + "' must be a max reduction");
        }
        if (!anchor->rescale.empty()) {
            throw InvalidSDFGException("Reduce: rescale anchor '" + reduction.rescale + "' is itself rescaled");
        }
        for (const auto& other : reductions_) {
            if (other.rescale == reduction.rescale && other.rescale_base != reduction.rescale_base) {
                throw InvalidSDFGException(
                    "Reduce: reductions rescaled by '" + reduction.rescale + "' use different bases"
                );
            }
        }
    }
    for (const auto& reduction : reductions_) {
        if (reduction.original_index.is_null()) {
            continue;
        }
        if (!function.exists(reduction.container)) {
            throw InvalidSDFGException("Reduce: missing original accumulator '" + reduction.container + "'");
        }
        if (symbolic::uses(reduction.original_index, indvar())) {
            throw InvalidSDFGException("Reduce: original index depends on the reduction variable");
        }
        for (const auto& symbol : symbolic::atoms(reduction.original_index)) {
            if (!function.exists(symbol->get_name())) {
                throw InvalidSDFGException("Reduce: undeclared original-index symbol '" + symbol->get_name() + "'");
            }
        }
    }
};

void Reduce::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    StructuredLoop::replace(old_expression, new_expression);

    for (auto& reduction : reductions_) {
        if (!reduction.original_index.is_null()) {
            reduction.original_index = symbolic::subs(reduction.original_index, old_expression, new_expression);
        }
    }

    if (SymEngine::is_a<SymEngine::Symbol>(*old_expression) && SymEngine::is_a<SymEngine::Symbol>(*new_expression)) {
        const auto& old_name = SymEngine::rcp_static_cast<const SymEngine::Symbol>(old_expression)->get_name();
        const auto& new_name = SymEngine::rcp_static_cast<const SymEngine::Symbol>(new_expression)->get_name();
        for (auto& reduction : this->reductions_) {
            if (reduction.container == old_name) {
                reduction.container = new_name;
            }
            if (reduction.rescale == old_name) {
                reduction.rescale = new_name;
            }
        }
    }
}

void Reduce::replace(const symbolic::ExpressionMapping& replacements) {
    StructuredLoop::replace(replacements);

    for (auto& reduction : this->reductions_) {
        if (!reduction.original_index.is_null()) {
            reduction.original_index = SymEngine::subs(reduction.original_index, replacements);
        }
        auto it = replacements.find(symbolic::symbol(reduction.container));
        if (it != replacements.end() && SymEngine::is_a<SymEngine::Symbol>(*it->second)) {
            reduction.container = SymEngine::rcp_static_cast<const SymEngine::Symbol>(it->second)->get_name();
        }
        if (!reduction.rescale.empty()) {
            auto anchor = replacements.find(symbolic::symbol(reduction.rescale));
            if (anchor != replacements.end() && SymEngine::is_a<SymEngine::Symbol>(*anchor->second)) {
                reduction.rescale = SymEngine::rcp_static_cast<const SymEngine::Symbol>(anchor->second)->get_name();
            }
        }
    }
}

const std::vector<ReductionInfo>& Reduce::reductions() const {
    return this->reductions_;
};

void Reduce::original_index(const std::string& container, symbolic::Expression index) {
    for (auto& reduction : reductions_) {
        if (reduction.container == container) {
            reduction.original_index = index;
            return;
        }
    }
    throw InvalidSDFGException("Reduce: unknown accumulator '" + container + "'");
}

void Reduce::replace_reduction_container(const std::string& old_name, const std::string& new_name) {
    if (old_name == new_name) {
        return;
    }
    for (auto& reduction : this->reductions_) {
        if (reduction.container == old_name) {
            reduction.container = new_name;
        }
        if (reduction.rescale == old_name) {
            reduction.rescale = new_name;
        }
    }
};

} // namespace structured_control_flow
} // namespace sdfg
