#include "sdfg/visualizer/visualizer.h"

#include <cassert>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "sdfg/data_flow/tasklet.h"
#include "sdfg/helpers/helpers.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/return.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/while.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/type.h"
#include "symengine/basic.h"

namespace sdfg {
namespace visualizer {

constexpr const char* code_to_string(data_flow::TaskletCode c) {
    switch (c) {
        case data_flow::TaskletCode::assign:
            return "=";
        case data_flow::TaskletCode::int_add:
        case data_flow::TaskletCode::fp_add:
            return "+";
        case data_flow::TaskletCode::int_sub:
        case data_flow::TaskletCode::fp_sub:
            return "-";
        case data_flow::TaskletCode::int_mul:
        case data_flow::TaskletCode::fp_mul:
            return "*";
        case data_flow::TaskletCode::int_udiv:
        case data_flow::TaskletCode::int_sdiv:
        case data_flow::TaskletCode::fp_div:
            return "/";
        case data_flow::TaskletCode::int_urem:
        case data_flow::TaskletCode::int_srem:
        case data_flow::TaskletCode::fp_rem:
            return "%";
        case data_flow::TaskletCode::fp_fma:
            return "fma";
        case data_flow::TaskletCode::fp_one:
            return "of!=";
        default:
            return "?";
    };
};

std::string Visualizer::expression(const std::string expr) {
    if (this->replacements_.empty()) return expr;
    std::string res = expr;
    size_t pos1 = 0, pos2 = 0;
    for (std::pair<const std::string, const std::string> replace : this->replacements_) {
        pos2 = res.find(replace.first);
        if (pos2 == res.npos) continue;
        pos1 = 0;
        std::stringstream res_tmp;
        while (pos2 < res.npos) {
            res_tmp << res.substr(pos1, pos2 - pos1) << replace.second;
            pos1 = pos2 + replace.first.size();
            pos2 = res.find(replace.first, pos1);
        }
        if (pos1 < res.npos) res_tmp << res.substr(pos1);
        res = res_tmp.str();
    }
    return res;
}

void Visualizer::visualizeNode(const StructuredSDFG& sdfg, const structured_control_flow::ControlFlowNode& node) {
    if (auto block = dynamic_cast<const structured_control_flow::Block*>(&node)) {
        this->visualizeBlock(sdfg, *block);
        return;
    }
    if (auto sequence = dynamic_cast<const structured_control_flow::Sequence*>(&node)) {
        this->visualizeSequence(sdfg, *sequence);
        return;
    }
    if (auto if_else = dynamic_cast<const structured_control_flow::IfElse*>(&node)) {
        this->visualizeIfElse(sdfg, *if_else);
        return;
    }
    if (auto while_loop = dynamic_cast<const structured_control_flow::While*>(&node)) {
        this->visualizeWhile(sdfg, *while_loop);
        return;
    }
    if (auto loop = dynamic_cast<const structured_control_flow::For*>(&node)) {
        this->visualizeFor(sdfg, *loop);
        return;
    }
    if (auto return_node = dynamic_cast<const structured_control_flow::Return*>(&node)) {
        this->visualizeReturn(sdfg, *return_node);
        return;
    }
    if (auto break_node = dynamic_cast<const structured_control_flow::Break*>(&node)) {
        this->visualizeBreak(sdfg, *break_node);
        return;
    }
    if (auto continue_node = dynamic_cast<const structured_control_flow::Continue*>(&node)) {
        this->visualizeContinue(sdfg, *continue_node);
        return;
    }
    if (auto map_node = dynamic_cast<const structured_control_flow::Map*>(&node)) {
        this->visualizeMap(sdfg, *map_node);
        return;
    }
    throw std::runtime_error("Unsupported control flow node");
}

void Visualizer::visualizeTasklet(data_flow::Tasklet const& tasklet) {
    std::string op = code_to_string(tasklet.code());
    std::vector<std::string> arguments;
    for (size_t i = 0; i < tasklet.inputs().size(); ++i) {
        arguments.push_back(this->expression(tasklet.input(i)));
    }

    if (tasklet.code() == data_flow::TaskletCode::assign) {
        this->stream_ << arguments.at(0);
    } else if (tasklet.code() == data_flow::TaskletCode::fp_fma) {
        if (arguments.size() != 3) throw std::runtime_error("FMA requires 3 arguments");
        this->stream_ << arguments.at(0) << " * " << arguments.at(1) << " + " << arguments.at(2);
    } else {
        this->stream_ << op << "(" << helpers::join(arguments, ", ") << ")";
    }
}

void Visualizer::visualizeForBounds(
    symbolic::Symbol const& indvar,
    symbolic::Expression const& init,
    symbolic::Condition const& condition,
    symbolic::Expression const& update
) {
    this->stream_ << indvar->get_name() << " = " << this->expression(init->__str__()) << "; "
                  << this->expression(condition->__str__()) << "; " << indvar->get_name() << " = "
                  << this->expression(update->__str__());
}

std::string Visualizer::subsetRangeString(data_flow::Subset const& subset, int subIdx) {
    auto& dim = subset.at(subIdx);
    return this->expression(dim->__str__());
}

/// @brief If known, use the type to better visualize structures. Then track the type as far as it goes.
void Visualizer::
    visualizeSubset(Function const& function, data_flow::Subset const& sub, types::IType const* type, int subIdx) {
    if (static_cast<int>(sub.size()) <= subIdx) {
        return;
    }
    if (auto structure_type = dynamic_cast<const types::Structure*>(type)) {
        types::StructureDefinition const& definition = function.structure(structure_type->name());

        auto memberIdx = SymEngine::rcp_dynamic_cast<const SymEngine::Integer>(sub.at(subIdx));
        if (!memberIdx.is_null()) {
            this->stream_ << ".member_" << this->expression(memberIdx->__str__());
            auto& member_type = definition.member_type(memberIdx);
            this->visualizeSubset(function, sub, &member_type, subIdx + 1);
        } else {
            this->stream_ << ".member[" << subsetRangeString(sub, subIdx) << "]";
            this->visualizeSubset(function, sub, nullptr, subIdx + 1);
        }
    } else if (auto tensor_type = dynamic_cast<const types::Tensor*>(type)) {
        auto& shape = tensor_type->shape();
        int tensor_dims = shape.size();
        int i = 0;
        while (i < tensor_dims && subIdx < sub.size()) {
            this->stream_ << "[" << subsetRangeString(sub, i) << ":" << subsetRangeString(shape, i) << "]";
            ++subIdx;
            ++i;
        }
        if (subIdx < sub.size()) {
            this->visualizeSubset(function, sub, &tensor_type->element_type(), subIdx);
        }
    } else if (auto array_type = dynamic_cast<const types::Array*>(type)) {
        this->stream_ << "[" << subsetRangeString(sub, subIdx) << "]";
        types::IType const& element_type = array_type->element_type();
        this->visualizeSubset(function, sub, &element_type, subIdx + 1);
    } else if (auto pointer_type = dynamic_cast<const types::Pointer*>(type)) {
        this->stream_ << "[" << subsetRangeString(sub, subIdx) << "]";
        const types::IType* pointee_type;
        if (pointer_type->has_pointee_type()) {
            pointee_type = &pointer_type->pointee_type();
        } else {
            auto z = symbolic::zero();
            if (!symbolic::eq(sub.at(subIdx), z)) {
                this->stream_ << "#illgl";
            }
            pointee_type = nullptr;
        }
        this->visualizeSubset(function, sub, pointee_type, subIdx + 1);
    } else {
        if (type == nullptr) {
            this->stream_ << "(rogue)";
        }
        this->stream_ << "[" << subsetRangeString(sub, subIdx) << "]";
        visualizeSubset(function, sub, nullptr, subIdx + 1);
    }
}

} // namespace visualizer
} // namespace sdfg
