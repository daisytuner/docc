#pragma once

#include <sdfg/analysis/analysis.h>
#include <sdfg/builder/structured_sdfg_builder.h>
#include <sdfg/plugins/target_mapping.h>

namespace docc::target::et {

class EtLibNodeMapper : public sdfg::plugins::TargetMapper {
public:
    bool try_map(
        sdfg::builder::StructuredSDFGBuilder& builder,
        sdfg::analysis::AnalysisManager& analysis_manager,
        sdfg::data_flow::LibraryNode& node
    ) const override;
};

} // namespace docc::target::et
