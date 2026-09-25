#include "sdfg/passes/dataflow/memlet_linearization.h"

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

MemletLinearization::
    MemletLinearization(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
    : visitor::NonStoppingStructuredSDFGVisitor(builder, analysis_manager) {
}

bool MemletLinearization::accept(structured_control_flow::Block& block) {
    bool applied = false;
    auto& dfg = block.dataflow();

    for (auto& memlet : dfg.edges()) {
        if (memlet.subset().empty()) {
            continue;
        }
        auto& base_type = memlet.base_type();
        if (base_type.type_id() != types::TypeID::Pointer) {
            continue;
        }
        auto& pointer_type = dynamic_cast<const types::Pointer&>(base_type);
        if (!pointer_type.has_pointee_type()) {
            continue;
        }

        // Check if pointee type contains arrays
        auto& pointee_type = pointer_type.pointee_type();
        const types::IType* current_type = &pointee_type;
        std::vector<symbolic::Expression> array_dimensions = {symbolic::one()};

        // Collect all array dimensions
        while (current_type->type_id() == types::TypeID::Array) {
            auto array_type = dynamic_cast<const types::Array*>(current_type);
            array_dimensions.push_back(array_type->num_elements());
            current_type = &array_type->element_type();
        }
        if (current_type->type_id() != types::TypeID::Scalar) {
            continue;
        }
        if (array_dimensions.size() == 1) {
            continue;
        }

        auto storage = pointer_type.storage_type();
        auto alignment = pointer_type.alignment();
        auto initializer = pointer_type.initializer();
        types::Pointer new_pointer_type(storage, alignment, initializer, *current_type);

        // Linearize subset
        auto old_subset = memlet.subset();
        data_flow::Subset new_subset;

        symbolic::Expression linearized_index = symbolic::zero();

        for (size_t i = 0; i < old_subset.size(); ++i) {
            symbolic::Expression stride = symbolic::one();

            // Calculate stride: product of all dimensions after current dimension
            for (size_t j = i + 1; j < array_dimensions.size(); ++j) {
                stride = symbolic::mul(stride, array_dimensions[j]);
            }
            linearized_index = symbolic::add(linearized_index, symbolic::mul(old_subset[i], stride));
        }

        new_subset.push_back(linearized_index);
        memlet.set_base_type(new_pointer_type);
        memlet.set_subset(new_subset);

        applied = true;
    }

    return applied;
}

} // namespace passes
} // namespace sdfg
