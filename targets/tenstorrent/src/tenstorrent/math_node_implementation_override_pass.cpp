#include "docc/target/tenstorrent/math_node_implementation_override_pass.h"

#include "docc/target/tenstorrent/library_node_mapping.h"
#include "docc/target/tenstorrent/plugin.h"
#include "sdfg/data_flow/library_nodes/math/math.h"

namespace sdfg {
namespace tenstorrent {

bool TTLibNodeMapper::try_map(
    builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager, data_flow::LibraryNode& node
) const {
    auto code = node.code();
    if (code == math::blas::LibraryNodeType_GEMM.value()) {
        auto* gemm_node = dynamic_cast<math::blas::GEMMNode*>(&node);

        auto data_type = gemm_node->scalar_primitive();
        if (data_type == types::PrimitiveType::Float && gemm_node->trans_a() == math::blas::No &&
            gemm_node->trans_b() == math::blas::No && gemm_node->layout() == math::blas::RowMajor) {
            gemm_node->implementation_type() = ImplementationType_Tenstorrent_WithTransfers;
            return true;
        }
    } else if (code == math::blas::LibraryNodeType_DOT.value()) {
        auto* dot_node = dynamic_cast<math::blas::DotNode*>(&node);

        auto data_type = dot_node->scalar_primitive();
        if (data_type == types::PrimitiveType::Float && symbolic::null_safe_eq(dot_node->incx(), symbolic::one()) &&
            symbolic::null_safe_eq(dot_node->incy(), symbolic::one())) {
            dot_node->implementation_type() = ImplementationType_Tenstorrent_WithTransfers;
            return true;
        }
    }

    return false;
}

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
