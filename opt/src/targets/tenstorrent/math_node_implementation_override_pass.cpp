#include "sdfg/targets/tenstorrent/math_node_implementation_override_pass.h"

#include "sdfg/data_flow/library_nodes/math/math.h"
#include "sdfg/targets/tenstorrent/plugin.h"

namespace sdfg {
namespace tenstorrent {

MathNodeImplementationOverride::
    MathNodeImplementationOverride(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
    : visitor::StructuredSDFGVisitor(builder, analysis_manager) {}

bool MathNodeImplementationOverride::accept(structured_control_flow::Block& node) {
    auto& dataflow = node.dataflow();
    for (auto& library_node : dataflow.nodes()) {
        if (auto lib_node = dynamic_cast<math::blas::BLASNode*>(&library_node)) {
            auto implType = try_map_blas_node_implementation(*lib_node);

            if (implType) {
                lib_node->implementation_type() = implType.value();
            }
        }
    }
    return false;
}

} // namespace tenstorrent
} // namespace sdfg
