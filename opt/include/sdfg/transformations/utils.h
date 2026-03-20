#pragma once

#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {

sdfg::symbolic::Integer get_iteration_count(sdfg::structured_control_flow::StructuredLoop& loop);

} // namespace sdfg
