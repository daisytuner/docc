#include "docc/target/et/et_lib_node_mapper.h"

#include <docc/target/et/target.h>
#include <sdfg/data_flow/library_nodes/math/blas/gemm_node.h>

namespace docc::target::et {

using namespace sdfg;

bool EtLibNodeMapper::try_map(
    builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager, data_flow::LibraryNode& node
) const {
    if (node.code() == math::blas::LibraryNodeType_GEMM.value()) {
        auto* gemm_node = dynamic_cast<math::blas::GEMMNode*>(&node);
        auto data_type = gemm_node->scalar_primitive();
        if (data_type == types::PrimitiveType::Float) {
            gemm_node->implementation_type() = ImplementationType_ETSOC_WithTransfers;
            return true;
        }
    }

    return false;
}

} // namespace docc::target::et
