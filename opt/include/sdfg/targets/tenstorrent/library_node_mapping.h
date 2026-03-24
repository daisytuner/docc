#pragma once

#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/library_nodes/math/math.h"

namespace sdfg::tenstorrent {

std::optional<data_flow::ImplementationType> try_map_library_node_implementation(const data_flow::LibraryNodeCode& code
);
std::optional<data_flow::ImplementationType>
try_map_library_node_implementation(const data_flow::LibraryNodeCode& code, types::PrimitiveType data_type);

std::optional<data_flow::ImplementationType> try_map_blas_node_implementation(const math::blas::BLASNode& node);

// TODO function to generate SDFG contents for implementation type to be compatible with einsum-recognition

} // namespace sdfg::tenstorrent
