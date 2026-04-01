#pragma once

#include "sdfg/codegen/instrumentation/instrumentation_plan.h"

namespace sdfg::auto_util {

void add_offloading_instrumentations(codegen::InstrumentationPlan& plan, sdfg::StructuredSDFG& sdfg);

}
