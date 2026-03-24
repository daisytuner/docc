#include "sdfg/targets/tenstorrent/library_node_mapping.h"

#include "sdfg/data_flow/library_nodes/math/blas/dot_node.h"
#include "sdfg/data_flow/library_nodes/math/blas/gemm_node.h"

#include "sdfg/targets/tenstorrent/plugin.h"

namespace sdfg::tenstorrent {

std::optional<data_flow::ImplementationType> try_map_library_node_implementation(const data_flow::LibraryNodeCode& code
) {
    if (code == math::blas::LibraryNodeType_GEMM.value()) {
        return ImplementationType_Tenstorrent_WithTransfers;
    } else if (code == math::blas::LibraryNodeType_DOT.value()) {
        return ImplementationType_Tenstorrent_WithTransfers;
    } else {
        return std::nullopt;
    }
}

std::optional<data_flow::ImplementationType>
try_map_library_node_implementation(const data_flow::LibraryNodeCode& code, types::PrimitiveType data_type) {
    if (data_type == types::PrimitiveType::Float) {
        if (code == math::blas::LibraryNodeType_GEMM.value()) {
            return ImplementationType_Tenstorrent_WithTransfers;
        } else if (code == math::blas::LibraryNodeType_DOT.value()) {
            return ImplementationType_Tenstorrent_WithTransfers;
        } else {
            return std::nullopt;
        }
    } else {
        return std::nullopt;
    }
}

std::optional<data_flow::ImplementationType> try_map_blas_node_implementation(const math::blas::BLASNode& node) {
    auto data_type = node.scalar_primitive();
    auto& code = node.code();

    return try_map_library_node_implementation(code, data_type);
}

} // namespace sdfg::tenstorrent
