#pragma once

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/concat_node.h"
#include "sdfg/targets/offloading/target_expansion.h"

namespace sdfg {
namespace offloading {

class RocmConcatExpander : public TargetLibNodeExpander {
private:
    math::tensor::ConcatNode& node_;

public:
    RocmConcatExpander(math::tensor::ConcatNode& library_node)
        : TargetLibNodeExpander(library_node), node_(library_node) {};

    bool expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);
};

} // namespace offloading
} // namespace sdfg
