#pragma once

#include "sdfg/metadata/loop_provenance.h"

namespace sdfg {
namespace passes::rpc {

inline void record_rpc_loop_result(
    Element& element,
    double expected_speedup,
    double vector_distance
) {
    metadata::set_rpc_optimization(element, expected_speedup, vector_distance);
}

} // namespace passes::rpc
} // namespace sdfg
