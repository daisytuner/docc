#include "sdfg/passes/dataflow/float_widening.h"

#include <functional>
#include <memory>
#include <vector>

#include <nlohmann/json.hpp>

#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/batchnorm_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/layernorm_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/matmul_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/spatial_tensor_node.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/structured_control_flow/while.h"
#include "sdfg/types/array.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/tensor.h"

namespace sdfg {
namespace passes {

namespace {

/// @p type with its elements of the @p widened types replaced by Float; null if it has none.
std::unique_ptr<types::IType> widen(const types::IType& type, const std::set<types::PrimitiveType>& widened) {
    if (auto* scalar = dynamic_cast<const types::Scalar*>(&type)) {
        if (!widened.contains(scalar->primitive_type())) {
            return nullptr;
        }
        return std::make_unique<types::Scalar>(
            scalar->storage_type(), scalar->alignment(), scalar->initializer(), types::PrimitiveType::Float
        );
    }
    if (auto* tensor = dynamic_cast<const types::Tensor*>(&type)) {
        auto element = widen(tensor->element_type(), widened);
        if (!element) {
            return nullptr;
        }
        return std::make_unique<types::Tensor>(
            tensor->storage_type(),
            tensor->alignment(),
            tensor->initializer(),
            static_cast<const types::Scalar&>(*element),
            tensor->layout()
        );
    }
    if (auto* pointer = dynamic_cast<const types::Pointer*>(&type)) {
        if (!pointer->has_pointee_type()) {
            return nullptr;
        }
        auto pointee = widen(pointer->pointee_type(), widened);
        if (!pointee) {
            return nullptr;
        }
        return std::make_unique<
            types::Pointer>(pointer->storage_type(), pointer->alignment(), pointer->initializer(), *pointee);
    }
    if (auto* array = dynamic_cast<const types::Array*>(&type)) {
        auto element = widen(array->element_type(), widened);
        if (!element) {
            return nullptr;
        }
        return std::make_unique<types::Array>(
            array->storage_type(), array->alignment(), array->initializer(), *element, array->num_elements()
        );
    }
    return nullptr;
}

/// Widens the element types a library node stores itself (casts, quantizations, contraction types).
bool widen_library_node(data_flow::LibraryNode& node, const std::set<types::PrimitiveType>& widened) {
    auto wide = [&](types::PrimitiveType type) {
        return widened.contains(type) ? types::PrimitiveType::Float : type;
    };
    if (auto* elementwise = dynamic_cast<math::tensor::ElementWiseDataflowTensorNode*>(&node)) {
        auto type = elementwise->fixed_quantization();
        elementwise->set_fixed_quantization(wide(type));
        return wide(type) != type;
    }
    if (auto* matmul = dynamic_cast<math::tensor::MatMulNode*>(&node)) {
        auto types = matmul->types();
        math::tensor::ContractionTypes widened_types{wide(types.input), wide(types.accumulator), wide(types.output)};
        matmul->set_types(widened_types);
        return widened_types.input != types.input || widened_types.accumulator != types.accumulator ||
               widened_types.output != types.output;
    }
    if (auto* spatial = dynamic_cast<math::tensor::SpatialTensorNode*>(&node)) {
        auto type = spatial->fixed_quantization();
        spatial->set_fixed_quantization(wide(type));
        return wide(type) != type;
    }
    if (auto* layernorm = dynamic_cast<math::tensor::LayerNormNode*>(&node)) {
        auto type = layernorm->quantization();
        layernorm->set_quantization(wide(type));
        return wide(type) != type;
    }
    if (auto* batchnorm = dynamic_cast<math::tensor::BatchNormNode*>(&node)) {
        auto type = batchnorm->quantization();
        batchnorm->set_quantization(wide(type));
        return wide(type) != type;
    }
    if (auto* cmath = dynamic_cast<math::cmath::CMathNode*>(&node)) {
        auto type = cmath->primitive_type();
        cmath->set_primitive_type(wide(type));
        return wide(type) != type;
    }
    return false;
}

/// Every block under @p node, while loops included.
void for_each_block(
    structured_control_flow::ControlFlowNode& node, const std::function<void(structured_control_flow::Block&)>& fn
) {
    if (auto* block = dynamic_cast<structured_control_flow::Block*>(&node)) {
        fn(*block);
    } else if (auto* sequence = dynamic_cast<structured_control_flow::Sequence*>(&node)) {
        for (size_t i = 0; i < sequence->size(); ++i) {
            for_each_block(sequence->at(i), fn);
        }
    } else if (auto* loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&node)) {
        for_each_block(loop->root(), fn);
    } else if (auto* loop = dynamic_cast<structured_control_flow::While*>(&node)) {
        for_each_block(loop->root(), fn);
    } else if (auto* branches = dynamic_cast<structured_control_flow::IfElse*>(&node)) {
        for (size_t i = 0; i < branches->size(); ++i) {
            for_each_block(branches->at(i).first, fn);
        }
    }
}

} // namespace

FloatWidening::FloatWidening(std::set<types::PrimitiveType> widened) : widened_(std::move(widened)) {
}

std::string FloatWidening::name() {
    return "FloatWidening";
}

bool FloatWidening::run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    if (widened_.empty()) {
        return false;
    }
    auto& sdfg = builder.subject();
    bool changed = false;
    nlohmann::json arguments = nlohmann::json::object();
    if (auto found = sdfg.metadata().find(ARGUMENT_TYPES); found != sdfg.metadata().end()) {
        arguments = nlohmann::json::parse(found->second);
    }
    for (auto& name : std::vector<std::string>(sdfg.containers().begin(), sdfg.containers().end())) {
        auto& type = sdfg.type(name);
        if (auto wide = widen(type, widened_)) {
            if (sdfg.is_argument(name) && !arguments.contains(name)) {
                arguments[name] = types::primitive_type_to_string(type.primitive_type());
            }
            builder.change_type(name, *wide);
            changed = true;
        }
    }
    for_each_block(sdfg.root(), [&](structured_control_flow::Block& block) {
        auto& graph = block.dataflow();
        for (auto& memlet : graph.edges()) {
            if (auto wide = widen(memlet.base_type(), widened_)) {
                memlet.set_base_type(*wide);
                changed = true;
            }
        }
        for (auto& node : graph.nodes()) {
            if (auto* library_node = dynamic_cast<data_flow::LibraryNode*>(&node)) {
                changed |= widen_library_node(*library_node, widened_);
            }
        }
        // Constants carry their own type: each is replaced by a Float constant of the same literal.
        std::vector<data_flow::ConstantNode*> constants;
        for (auto& node : graph.nodes()) {
            if (auto* constant = dynamic_cast<data_flow::ConstantNode*>(&node);
                constant && widen(constant->type(), widened_)) {
                constants.push_back(constant);
            }
        }
        for (auto* constant : constants) {
            auto& replacement =
                builder
                    .add_constant(block, constant->data(), *widen(constant->type(), widened_), constant->debug_info());
            std::vector<const data_flow::Memlet*> uses;
            for (auto& memlet : graph.out_edges(*constant)) {
                uses.push_back(&memlet);
            }
            for (auto* memlet : uses) {
                builder.add_memlet(
                    block,
                    replacement,
                    memlet->src_conn(),
                    const_cast<data_flow::DataFlowNode&>(memlet->dst()),
                    memlet->dst_conn(),
                    memlet->subset(),
                    memlet->base_type(),
                    memlet->debug_info()
                );
                builder.remove_memlet(block, *memlet);
            }
            builder.remove_node(block, *constant);
            changed = true;
        }
    });
    if (!arguments.empty()) {
        sdfg.add_metadata(ARGUMENT_TYPES, arguments.dump());
    }
    if (changed) {
        analysis_manager.invalidate_all();
    }
    return changed;
}

} // namespace passes
} // namespace sdfg
