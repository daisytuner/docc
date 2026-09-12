#pragma once

#include <functional>
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/control_flow_node.h"

namespace sdfg {
namespace visitor {

/// Visit every Block reachable under @p node (recursing through sequences,
/// loops, and if-else branches).
void for_each_block(
    structured_control_flow::ControlFlowNode& node, const std::function<void(structured_control_flow::Block&)>& fn
);

} // namespace visitor
} // namespace sdfg
