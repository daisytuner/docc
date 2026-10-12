#include "docc/target/axelera/datapath.h"

#include <stdexcept>
#include <unordered_map>

#include <sdfg/data_flow/access_node.h>
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

bool is_dpu_operation(data_flow::TaskletCode code) {
    switch (code) {
        case data_flow::TaskletCode::assign:
        case data_flow::TaskletCode::fp_add:
        case data_flow::TaskletCode::fp_sub:
        case data_flow::TaskletCode::fp_mul:
            return true;
        default:
            return false;
    }
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

} // namespace

DatapathPlan
plan_datapath(const StructuredSDFG& sdfg, structured_control_flow::Sequence& body, const symbolic::Symbol& lane) {
    DatapathPlan plan;
    plan.num_registers = 0;
    std::optional<DatapathStream> output;
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
        plan.program.push_back({reg, plan.inputs.size() - 1});
        popped.push_back(reg);
        return reg;
    };

    for (size_t i = 0; i < body.size(); ++i) {
        auto block = dynamic_cast<structured_control_flow::Block*>(&body.at(i));
        if (!block) {
            unsupported("control flow or symbol assignments inside a datapath");
        }
        const auto& dataflow = block->dataflow();
        for (auto* node : dataflow.topological_sort()) {
            if (dynamic_cast<const data_flow::ConstantNode*>(node)) {
                unsupported("constants");
            }
            if (dynamic_cast<const data_flow::AccessNode*>(node)) {
                continue;
            }
            auto tasklet = dynamic_cast<const data_flow::Tasklet*>(node);
            if (!tasklet) {
                unsupported("library nodes inside a datapath");
            }

            std::vector<size_t> operands;
            for (auto* iedge : dataflow.in_edges_by_connector(*tasklet)) {
                auto& src = static_cast<const data_flow::AccessNode&>(iedge->src());
                auto& type = sdfg.type(src.data());
                if (dynamic_cast<const types::Pointer*>(&type)) {
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
            if (!is_dpu_operation(tasklet->code())) {
                unsupported("tasklet code " + std::to_string(static_cast<int>(tasklet->code())) + " on the DPU");
            }

            size_t reg = plan.num_registers++;
            plan.program.push_back({reg, std::nullopt, tasklet->code(), operands});

            auto& oedge = *dataflow.out_edges(*tasklet).begin();
            auto& dst = static_cast<const data_flow::AccessNode&>(oedge.dst());
            auto& dst_type = sdfg.type(dst.data());
            if (dynamic_cast<const types::Pointer*>(&dst_type)) {
                if (output) {
                    unsupported("a second output");
                }
                output = stream_of(sdfg, dst, oedge, lane);
                plan.result = reg;
            } else if (is_float_scalar(dst_type)) {
                scalars[dst.data()] = reg;
            } else {
                unsupported("results of type " + dst_type.print());
            }
        }
    }

    if (!output) {
        unsupported("a datapath without output");
    }
    for (auto& input : plan.inputs) {
        if (input.container == output->container) {
            unsupported("reading and writing " + input.container + " in one datapath");
        }
    }
    plan.output = *output;
    return plan;
}

} // namespace axelera
} // namespace sdfg
