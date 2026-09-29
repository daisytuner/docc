#pragma once

#include <memory>
#include <nlohmann/json_fwd.hpp>
#include <string>

#include "sdfg/analysis/analysis.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_sdfg.h"

namespace sdfg {
namespace serializer {

class CutoutSerializer : public JSONSerializer {
public:
    CutoutSerializer() {
    }

    nlohmann::json serialize(
        const sdfg::StructuredSDFG& sdfg,
        analysis::AnalysisManager* analysis_manager,
        structured_control_flow::Sequence* cutout_root
    ) override;
};
} // namespace serializer
} // namespace sdfg
