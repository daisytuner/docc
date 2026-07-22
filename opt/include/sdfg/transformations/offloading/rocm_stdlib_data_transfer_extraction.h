#pragma once

#include <nlohmann/json_fwd.hpp>
#include <string>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/library_nodes/stdlib/memcpy.h"
#include "sdfg/data_flow/library_nodes/stdlib/memset.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/transformations/transformation.h"
#include "sdfg/types/pointer.h"

namespace sdfg {
namespace rocm {

class ROCMStdlibDataTransferExtraction : public transformations::Transformation {
private:
    data_flow::LibraryNode& lib_node_;

    void apply_memset(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        data_flow::DataFlowGraph& dfg,
        structured_control_flow::Sequence& sequence,
        structured_control_flow::Block& block
    );

    void apply_memcpy(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        data_flow::DataFlowGraph& dfg,
        structured_control_flow::Sequence& sequence,
        structured_control_flow::Block& block
    );

public:
    ROCMStdlibDataTransferExtraction(data_flow::LibraryNode& lib_node);

    virtual std::string name() const override;

    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        override;

    virtual void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual void to_json(nlohmann::json& json) const override;
};

} // namespace rocm
} // namespace sdfg
