#pragma once

#include <sdfg/plugins/plugins.h>
#include "sdfg/fp_math/library_nodes/ozaki_matmul_node.h"
#include "sdfg/serializer/json_serializer.h"

namespace sdfg {
namespace fp_math {

inline void register_fp_math_plugin(plugins::Context& context) {
    context.library_node_serializer_registry.register_library_node_serializer(LibraryNodeType_OzakiMatMul.value(), []() {
        return std::make_unique<OzakiMatMulNodeSerializer>();
    });
}

} // namespace fp_math
} // namespace sdfg
