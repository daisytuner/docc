#pragma once

#include "sdfg/structured_sdfg.h"

namespace sdfg {
namespace passes {

bool promote_device_residency(StructuredSDFG& sdfg, bool is_rocm);

} // namespace passes
} // namespace sdfg
