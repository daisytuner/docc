#pragma once

#include <string>
#include <unordered_map>

#include <sdfg/visitor/structured_sdfg_visitor.h>

namespace sdfg {
namespace codegen {

class LoopReport : public sdfg::visitor::StructuredSDFGVisitor {
private:
    std::unordered_map<std::string, size_t> report_;

public:
    LoopReport(sdfg::builder::StructuredSDFGBuilder& builder, sdfg::analysis::AnalysisManager& analysis_manager);

    const std::unordered_map<std::string, size_t>& report() const { return this->report_; }

    bool accept(sdfg::structured_control_flow::Block& node) override;

    bool accept(sdfg::structured_control_flow::For& node) override;

    bool accept(sdfg::structured_control_flow::While& node) override;

    bool accept(sdfg::structured_control_flow::Map& node) override;

    bool accept(sdfg::structured_control_flow::Reduce& node) override;
};


} // namespace codegen
} // namespace sdfg
