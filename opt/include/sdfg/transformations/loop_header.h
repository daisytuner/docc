#pragma once

#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace transformations {

/// Bounds and step of a loop, keeping its induction variable.
struct LoopHeader {
    symbolic::Expression init;
    symbolic::Condition condition;
    symbolic::Expression update;
};

/// Two directly nested loops and their headers after swapping them.
struct LoopSwap {
    structured_control_flow::StructuredLoop& outer;
    structured_control_flow::StructuredLoop& inner;
    LoopHeader new_outer;
    LoopHeader new_inner;
};

} // namespace transformations
} // namespace sdfg
