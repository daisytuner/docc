#include "sdfg/passes/dataflow/tensor_to_pointer_conversion.h"

#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/array.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/utils.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {

TensorToPointerConversion::
    TensorToPointerConversion(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
    : visitor::NonStoppingStructuredSDFGVisitor(builder, analysis_manager) {}

bool TensorToPointerConversion::accept(structured_control_flow::Block& block) {
    bool applied = false;
    auto& dfg = block.dataflow();

    for (auto& memlet : dfg.edges()) {
        auto& base_type = memlet.base_type();
        if (base_type.type_id() != types::TypeID::Tensor) {
            continue;
        }
        auto& tensor_type = static_cast<const types::Tensor&>(base_type);

        // Distinguish between pointer and scalar buffers
        const types::IType* container_type = nullptr;
        if (!dynamic_cast<const data_flow::AccessNode*>(&memlet.src())) {
            auto& dst_node = static_cast<const data_flow::AccessNode&>(memlet.dst());
            if (auto const_node = dynamic_cast<const data_flow::ConstantNode*>(&dst_node)) {
                container_type = &const_node->type();
            } else {
                container_type = &builder_.subject().type(dst_node.data());
            }
        } else {
            auto& src_node = static_cast<const data_flow::AccessNode&>(memlet.src());
            if (auto const_node = dynamic_cast<const data_flow::ConstantNode*>(&src_node)) {
                container_type = &const_node->type();
            } else {
                container_type = &builder_.subject().type(src_node.data());
            }
        }

        // Handle scalar containers (including scalar tensor memlets with empty subset)
        if (container_type->type_id() == types::TypeID::Scalar) {
            memlet.set_base_type(*container_type);
            memlet.set_subset({});
            applied = true;
            continue;
        }

        // For pointer containers, we need a non-empty subset to linearize
        if (memlet.subset().empty()) {
            continue;
        }

        auto& element_type = tensor_type.element_type();
        auto& shape = tensor_type.shape();
        auto& strides = tensor_type.strides();

        auto& tensor_subset = memlet.subset();
        symbolic::Expression linearized_access = tensor_type.offset();
        for (size_t i = 0; i < shape.size(); ++i) {
            linearized_access = symbolic::add(linearized_access, symbolic::mul(tensor_subset.at(i), strides.at(i)));
        }

        data_flow::Subset pointer_subset = {linearized_access};
        memlet.set_subset(pointer_subset);

        types::Pointer pointer_type(element_type);
        memlet.set_base_type(pointer_type);

        applied = true;
    }

    return applied;
}

} // namespace passes
} // namespace sdfg
