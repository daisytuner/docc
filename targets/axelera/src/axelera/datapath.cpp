#include "docc/target/axelera/datapath.h"

#include <bit>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <limits>
#include <numbers>
#include <ranges>
#include <stdexcept>
#include <unordered_map>

#include <sdfg/analysis/users.h>
#include <sdfg/data_flow/access_node.h>
#include <sdfg/data_flow/library_nodes/math/cmath/cmath_node.h>
#include <sdfg/data_flow/tasklet.h>
#include <sdfg/structured_control_flow/block.h>
#include <sdfg/types/pointer.h>
#include <sdfg/types/scalar.h>

namespace sdfg {
namespace axelera {

namespace {

[[noreturn]] void unsupported(const std::string& what) {
    throw std::runtime_error("Axelera datapath: " + what + " is not supported");
}

bool is_float_scalar(const types::IType& type) {
    auto scalar = dynamic_cast<const types::Scalar*>(&type);
    return scalar && scalar->primitive_type() == types::PrimitiveType::Float;
}

bool is_float_pointer(const types::IType& type) {
    auto pointer = dynamic_cast<const types::Pointer*>(&type);
    return pointer && pointer->has_pointee_type() && is_float_scalar(pointer->pointee_type());
}

/// The operation a reduction body folds into its accumulator.
std::optional<DpuOp> combine_op(const data_flow::CodeNode& node) {
    if (auto tasklet = dynamic_cast<const data_flow::Tasklet*>(&node)) {
        return tasklet->code() == data_flow::TaskletCode::fp_add ? std::optional(DpuOp::Add) : std::nullopt;
    }
    if (auto cmath = dynamic_cast<const math::cmath::CMathNode*>(&node);
        cmath && cmath->primitive_type() == types::PrimitiveType::Float) {
        switch (cmath->function()) {
            case math::cmath::CMathFunction::fmax:
                return DpuOp::Max;
            case math::cmath::CMathFunction::fmin:
                return DpuOp::Min;
            default:
                return std::nullopt;
        }
    }
    return std::nullopt;
}

using Emit = std::function<size_t(DpuOp, std::vector<size_t>)>;
using Constant = std::function<size_t(double)>;

/// Emits the DPU statements computing @p node from registers @p x; returns the result register, if supported.
std::optional<size_t>
lower(const data_flow::CodeNode& node, const std::vector<size_t>& x, const Emit& emit, const Constant& constant) {
    if (auto tasklet = dynamic_cast<const data_flow::Tasklet*>(&node)) {
        switch (tasklet->code()) {
            case data_flow::TaskletCode::assign:
                return emit(DpuOp::Copy, {x.at(0)});
            case data_flow::TaskletCode::fp_neg:
                return emit(DpuOp::Neg, {x.at(0)});
            case data_flow::TaskletCode::fp_add:
                return emit(DpuOp::Add, {x.at(0), x.at(1)});
            case data_flow::TaskletCode::fp_sub:
                return emit(DpuOp::Sub, {x.at(0), x.at(1)});
            case data_flow::TaskletCode::fp_mul:
                return emit(DpuOp::Mul, {x.at(0), x.at(1)});
            case data_flow::TaskletCode::fp_div:
                return emit(DpuOp::Mul, {x.at(0), emit(DpuOp::Recip, {x.at(1)})});
            case data_flow::TaskletCode::fp_fma:
                return emit(DpuOp::Fma, {x.at(0), x.at(1), x.at(2)});
            default:
                return std::nullopt;
        }
    }
    auto cmath = dynamic_cast<const math::cmath::CMathNode*>(&node);
    if (!cmath || cmath->primitive_type() != types::PrimitiveType::Float) {
        return std::nullopt;
    }
    switch (cmath->function()) {
        case math::cmath::CMathFunction::fmax:
            return emit(DpuOp::Max, {x.at(0), x.at(1)});
        case math::cmath::CMathFunction::fmin:
            return emit(DpuOp::Min, {x.at(0), x.at(1)});
        case math::cmath::CMathFunction::fabs:
            return emit(DpuOp::Max, {x.at(0), emit(DpuOp::Neg, {x.at(0)})});
        case math::cmath::CMathFunction::sqrt:
            return emit(DpuOp::Sqrt, {x.at(0)});
        case math::cmath::CMathFunction::sin:
            return emit(DpuOp::Sin, {x.at(0)});
        case math::cmath::CMathFunction::cos:
            return emit(DpuOp::Cos, {x.at(0)});
        case math::cmath::CMathFunction::log2:
            return emit(DpuOp::Log2, {x.at(0)});
        case math::cmath::CMathFunction::exp2:
            return emit(DpuOp::Exp2, {x.at(0)});
        case math::cmath::CMathFunction::exp:
            return emit(DpuOp::Exp2, {emit(DpuOp::Mul, {x.at(0), constant(std::numbers::log2e)})});
        case math::cmath::CMathFunction::log:
            return emit(DpuOp::Mul, {emit(DpuOp::Log2, {x.at(0)}), constant(std::numbers::ln2)});
        default:
            return std::nullopt;
    }
}

/// The value of a numeric constant such as "2.0", "-inf" or "1.5f".
std::optional<double> constant_value(const data_flow::ConstantNode& node) {
    const char* begin = node.data().c_str();
    char* end = nullptr;
    double value = std::strtod(begin, &end);
    std::string_view rest(end);
    if (end == begin || !(rest.empty() || rest == "f" || rest == "F")) {
        return std::nullopt;
    }
    return value;
}

/// `0 - x`, which the frontend emits for `-x` on arrays.
bool is_negation(const data_flow::CodeNode& node, const std::vector<const data_flow::Memlet*>& inputs) {
    auto tasklet = dynamic_cast<const data_flow::Tasklet*>(&node);
    if (!tasklet || tasklet->code() != data_flow::TaskletCode::fp_sub) {
        return false;
    }
    auto zero = dynamic_cast<const data_flow::ConstantNode*>(&inputs.at(0)->src());
    return zero && constant_value(*zero) == 0.0;
}

std::string describe(const data_flow::CodeNode& node) {
    if (auto tasklet = dynamic_cast<const data_flow::Tasklet*>(&node)) {
        return "tasklet code " + std::to_string(static_cast<int>(tasklet->code()));
    }
    if (auto cmath = dynamic_cast<const math::cmath::CMathNode*>(&node)) {
        return "math function " + cmath->name() + " on " + types::primitive_type_to_string(cmath->primitive_type());
    }
    if (auto library_node = dynamic_cast<const data_flow::LibraryNode*>(&node)) {
        return "library node " + library_node->code().value();
    }
    return "code node";
}

DatapathStream stream_of(
    const StructuredSDFG& sdfg,
    const data_flow::AccessNode& access,
    const data_flow::Memlet& memlet,
    const symbolic::Symbol& lane
) {
    auto& type = sdfg.type(access.data());
    if (!is_float_pointer(type)) {
        unsupported("streaming " + access.data() + " of type " + type.print());
    }
    if (memlet.subset().size() != 1) {
        unsupported("multi-dimensional access to " + access.data());
    }
    auto index = memlet.subset().at(0);
    if (symbolic::uses(symbolic::expand(symbolic::sub(index, lane)), lane)) {
        unsupported("non-contiguous access " + access.data() + "[" + index->__str__() + "]");
    }
    return {access.data(), index};
}

/// The single assignment `dst = src` a block holds, as (source node, memlet into the destination).
std::pair<const data_flow::DataFlowNode*, const data_flow::Memlet*>
single_assignment(structured_control_flow::Block& block, const std::string& what) {
    auto& dataflow = block.dataflow();
    if (std::ranges::distance(dataflow.nodes()) != 3) {
        unsupported(what + " that does more than one assignment");
    }
    for (auto& node : dataflow.nodes()) {
        auto tasklet = dynamic_cast<const data_flow::Tasklet*>(&node);
        if (!tasklet) {
            continue;
        }
        if (tasklet->code() != data_flow::TaskletCode::assign) {
            unsupported(what + " that computes instead of assigning");
        }
        auto& iedge = *dataflow.in_edges(*tasklet).begin();
        auto& oedge = *dataflow.out_edges(*tasklet).begin();
        return {&iedge.src(), &oedge};
    }
    unsupported(what + " without an assignment");
}

double identity(DpuOp combine) {
    switch (combine) {
        case DpuOp::Add:
            return 0.0;
        case DpuOp::Max:
            return -std::numeric_limits<double>::infinity();
        case DpuOp::Min:
            return std::numeric_limits<double>::infinity();
        default:
            unsupported("this reduction operation");
    }
}

} // namespace

DatapathPlan plan_datapath(
    const StructuredSDFG& sdfg,
    structured_control_flow::Sequence& body,
    const symbolic::Symbol& lane,
    const std::string& accumulator
) {
    DatapathPlan plan;
    plan.num_registers = 0;
    std::vector<size_t> popped;
    std::unordered_map<std::string, size_t> scalars;

    auto input_register = [&](const DatapathStream& stream) {
        for (size_t k = 0; k < plan.inputs.size(); ++k) {
            if (plan.inputs[k].container == stream.container && symbolic::eq(plan.inputs[k].index, stream.index)) {
                return popped[k];
            }
        }
        if (plan.inputs.size() == 2) {
            unsupported("a third input stream (" + stream.container + ")");
        }
        plan.inputs.push_back(stream);
        size_t reg = plan.num_registers++;
        plan.program.push_back({.destination = reg, .input_stream = plan.inputs.size() - 1});
        popped.push_back(reg);
        return reg;
    };

    std::vector<std::pair<double, size_t>> loaded;
    auto constant_register = [&](double value) {
        for (auto& [known, reg] : loaded) {
            if (std::bit_cast<uint64_t>(known) == std::bit_cast<uint64_t>(value)) {
                return reg;
            }
        }
        size_t reg = plan.num_registers++;
        plan.program.push_back({.destination = reg, .constant = value});
        loaded.emplace_back(value, reg);
        return reg;
    };

    for (size_t i = 0; i < body.size(); ++i) {
        auto block = dynamic_cast<structured_control_flow::Block*>(&body.at(i));
        if (!block) {
            unsupported("control flow or symbol assignments inside a datapath");
        }
        const auto& dataflow = block->dataflow();
        for (auto* node : dataflow.topological_sort()) {
            if (dynamic_cast<const data_flow::AccessNode*>(node)) {
                continue;
            }
            auto& code_node = static_cast<const data_flow::CodeNode&>(*node);
            if (plan.combine) {
                unsupported("computing after the accumulator update");
            }

            std::vector<size_t> operands;
            bool reads_accumulator = false;
            auto inputs = dataflow.in_edges_by_connector(code_node);
            bool negation = is_negation(code_node, inputs);
            for (auto* iedge : inputs) {
                if (auto constant = dynamic_cast<const data_flow::ConstantNode*>(&iedge->src())) {
                    if (negation && iedge == inputs.front()) {
                        continue;
                    }
                    auto value = constant_value(*constant);
                    if (!value) {
                        unsupported("the constant " + constant->data());
                    }
                    operands.push_back(constant_register(*value));
                    continue;
                }
                auto& src = static_cast<const data_flow::AccessNode&>(iedge->src());
                auto& type = sdfg.type(src.data());
                if (!accumulator.empty() && src.data() == accumulator) {
                    reads_accumulator = true;
                } else if (dynamic_cast<const types::Pointer*>(&type)) {
                    operands.push_back(input_register(stream_of(sdfg, src, *iedge, lane)));
                } else if (is_float_scalar(type)) {
                    auto it = scalars.find(src.data());
                    if (it == scalars.end()) {
                        unsupported("reading " + src.data() + " before the datapath computes it");
                    }
                    operands.push_back(it->second);
                } else {
                    unsupported("operands of type " + type.print());
                }
            }

            auto& oedge = *dataflow.out_edges(code_node).begin();
            auto& dst = static_cast<const data_flow::AccessNode&>(oedge.dst());
            auto& dst_type = sdfg.type(dst.data());
            bool writes_accumulator = !accumulator.empty() && dst.data() == accumulator;
            if (reads_accumulator || writes_accumulator) {
                auto op = combine_op(code_node);
                if (!reads_accumulator || !writes_accumulator || operands.size() != 1 || !op) {
                    unsupported("using the accumulator " + accumulator + " other than to add to, max or min into it");
                }
                plan.combine = op;
                plan.result = operands.front();
                continue;
            }

            auto emit = [&](DpuOp op, std::vector<size_t> args) {
                size_t reg = plan.num_registers++;
                plan.program.push_back({.destination = reg, .op = op, .operands = std::move(args)});
                return reg;
            };
            auto lowered = negation ? std::optional(emit(DpuOp::Neg, {operands.at(0)}))
                                    : lower(code_node, operands, emit, constant_register);
            if (!lowered) {
                unsupported(describe(code_node) + " on the DPU");
            }
            size_t reg = *lowered;
            if (dynamic_cast<const types::Pointer*>(&dst_type)) {
                if (!accumulator.empty()) {
                    unsupported("storing to memory inside a reduction");
                }
                if (plan.output) {
                    unsupported("a second output");
                }
                plan.output = stream_of(sdfg, dst, oedge, lane);
                plan.result = reg;
            } else if (is_float_scalar(dst_type)) {
                scalars[dst.data()] = reg;
            } else {
                unsupported("results of type " + dst_type.print());
            }
        }
    }

    if (accumulator.empty() && !plan.output) {
        unsupported("a datapath without output");
    }
    if (!accumulator.empty() && !plan.combine) {
        unsupported("a reduction that never updates its accumulator " + accumulator);
    }
    for (auto& input : plan.inputs) {
        if (plan.output && input.container == plan.output->container) {
            unsupported("reading and writing " + input.container + " in one datapath");
        }
    }
    return plan;
}

ReductionKernel match_reduction(
    const StructuredSDFG& sdfg, analysis::AnalysisManager& analysis_manager, structured_control_flow::Reduce& loop
) {
    if (loop.reductions().size() != 1) {
        unsupported("a loop with several reductions");
    }
    auto& reduction = loop.reductions().front();
    ReductionKernel kernel;
    kernel.accumulator = reduction.container;
    switch (reduction.operation) {
        case structured_control_flow::ReductionOperation::Add:
            kernel.combine = DpuOp::Add;
            break;
        case structured_control_flow::ReductionOperation::Max:
            kernel.combine = DpuOp::Max;
            break;
        case structured_control_flow::ReductionOperation::Min:
            kernel.combine = DpuOp::Min;
            break;
        default:
            unsupported("this reduction operation");
    }
    if (!is_float_scalar(sdfg.type(kernel.accumulator)) || !sdfg.is_transient(kernel.accumulator)) {
        unsupported("reducing into " + kernel.accumulator + ", which is not a local float");
    }

    auto parent = dynamic_cast<structured_control_flow::Sequence*>(loop.get_parent());
    int position = parent ? parent->index(loop) : -1;
    auto init = position > 0 ? dynamic_cast<structured_control_flow::Block*>(&parent->at(position - 1)) : nullptr;
    auto writeback = parent && position + 1 < static_cast<int>(parent->size())
                         ? dynamic_cast<structured_control_flow::Block*>(&parent->at(position + 1))
                         : nullptr;
    if (!init || !writeback) {
        unsupported("a reduction without its initialization right before and its store right after");
    }
    kernel.init = position - 1;
    kernel.writeback = position + 1;

    auto [start, init_memlet] = single_assignment(*init, "an accumulator initialization");
    auto constant = dynamic_cast<const data_flow::ConstantNode*>(start);
    auto& init_dst = static_cast<const data_flow::AccessNode&>(init_memlet->dst());
    if (!constant || init_dst.data() != kernel.accumulator) {
        unsupported("initializing " + kernel.accumulator + " from anything but a constant");
    }
    auto value = constant_value(*constant);
    if (!value) {
        unsupported("the accumulator start value " + constant->data());
    }
    // The datapath seeds the accumulator with the first value, which only matches starting at the identity.
    if (*value != identity(kernel.combine)) {
        unsupported("the accumulator start value " + constant->data() + ", which is not the reduction's identity,");
    }

    auto [result, store] = single_assignment(*writeback, "a reduction result store");
    auto result_access = dynamic_cast<const data_flow::AccessNode*>(result);
    auto& out = static_cast<const data_flow::AccessNode&>(store->dst());
    if (!result_access || result_access->data() != kernel.accumulator || !is_float_pointer(sdfg.type(out.data())) ||
        store->subset().size() != 1 || !symbolic::eq(store->subset().at(0), symbolic::zero())) {
        unsupported("storing the reduction result other than to element 0 of a float array");
    }
    // The ODR stores a whole PWORD, so nothing else may write the lanes next to the result.
    if (analysis_manager.get<analysis::Users>().writes(out.data()).size() != 1) {
        unsupported("storing the reduction result into " + out.data() + ", which is also written elsewhere,");
    }
    kernel.output = {out.data(), store->subset().at(0)};
    return kernel;
}

} // namespace axelera
} // namespace sdfg
