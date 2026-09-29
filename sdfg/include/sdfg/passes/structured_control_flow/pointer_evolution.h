#pragma once

#include "sdfg/passes/pass.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {

class IteratorToIndvar : public visitor::NonStoppingStructuredSDFGVisitor {
public:
    IteratorToIndvar(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);

    static std::string name() {
        return "IteratorToIndvar";
    };

    bool accept(structured_control_flow::For& node) override;
};

typedef VisitorPass<IteratorToIndvar> PointerEvolution;

} // namespace passes
} // namespace sdfg
