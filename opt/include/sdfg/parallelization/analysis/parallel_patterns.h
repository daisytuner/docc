#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/structured_control_flow/map.h"

namespace sdfg::parallelization {

using NeighborhoodOffset = std::vector<int64_t>;

struct NeighborhoodReads {
    std::set<NeighborhoodOffset> offsets;
    NeighborhoodOffset lower_halo;
    NeighborhoodOffset upper_halo;
};

struct StencilNeighborhood {
    std::vector<const structured_control_flow::Map*> axes;
    data_flow::Subset output_index;
    std::set<std::string> outputs;
    std::map<std::string, NeighborhoodReads> inputs;
};

class ParallelPatterns : public analysis::Analysis {
private:
    analysis::AnalysisManager* analysis_manager_ = nullptr;
    std::unordered_map<const structured_control_flow::Map*, std::optional<StencilNeighborhood>> neighborhoods_;

    std::optional<StencilNeighborhood> detect(structured_control_flow::Map& map);

public:
    explicit ParallelPatterns(StructuredSDFG& sdfg);

    std::string name() const override {
        return "ParallelPatterns";
    }

    void run(analysis::AnalysisManager& analysis_manager) override;

    const std::optional<StencilNeighborhood>& stencil(structured_control_flow::Map& map);
};

} // namespace sdfg::parallelization
