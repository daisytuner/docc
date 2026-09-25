#pragma once

#include <string>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/element.h"
#include "sdfg/passes/pass.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {

class TaskletFusion : public visitor::NonStoppingStructuredSDFGVisitor {
private:
    analysis::Users& users_analysis_;

    bool container_allowed_accesses(const std::string& container, const Element* allowed_read_and_writes);

public:
    TaskletFusion(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);

    static std::string name() {
        return "TaskletFusion";
    }

    virtual bool accept(structured_control_flow::Block& block) override;
};

typedef VisitorPass<TaskletFusion> TaskletFusionPass;

} // namespace passes
} // namespace sdfg
