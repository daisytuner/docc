#pragma once

#include "sdfg/passes/dataflow/byte_reference_elimination.h"
#include "sdfg/passes/dataflow/constant_propagation.h"
#include "sdfg/passes/dataflow/dead_data_elimination.h"
#include "sdfg/passes/dataflow/dead_reference_elimination.h"
#include "sdfg/passes/dataflow/reference_propagation.h"
#include "sdfg/passes/dataflow/trivial_array_elimination.h"
#include "sdfg/passes/debug_info_propagation.h"
#include "sdfg/passes/memory/allocation_management.h"
#include "sdfg/passes/pass.h"
#include "sdfg/passes/structured_control_flow/block_fusion.h"
#include "sdfg/passes/structured_control_flow/common_assignment_elimination.h"
#include "sdfg/passes/structured_control_flow/condition_elimination.h"
#include "sdfg/passes/structured_control_flow/dead_cfg_elimination.h"
#include "sdfg/passes/structured_control_flow/for2map.h"
#include "sdfg/passes/structured_control_flow/sequence_fusion.h"
#include "sdfg/passes/symbolic/symbol_evolution.h"
#include "sdfg/passes/symbolic/symbol_propagation.h"
#include "sdfg/passes/transformations/map_fusion_pass.h"

namespace sdfg {
namespace passes {

class Pipeline : public Pass {
private:
    std::vector<std::unique_ptr<Pass>> passes_;
    std::string name_;

public:
    Pipeline(const std::string& name);

    virtual std::string name();

    size_t size() const;

    virtual bool run(builder::SDFGBuilder& builder);

    virtual bool run(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);

    template<class T>
    void register_pass() {
        this->passes_.push_back(std::make_unique<T>());
    };

    static Pipeline dataflow_simplification();

    static Pipeline symbolic_simplification();

    static Pipeline dead_code_elimination();

    static Pipeline expression_combine();

    static Pipeline memlet_combine();

    static Pipeline controlflow_simplification();

    static Pipeline data_parallelism();

    static Pipeline memory();

    static Pipeline expansion();

    static Pipeline map_fusion();
};

} // namespace passes
} // namespace sdfg
