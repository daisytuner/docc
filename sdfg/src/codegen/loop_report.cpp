#include "sdfg/codegen/loop_report.h"

namespace sdfg {
namespace codegen {

LoopReport::LoopReport(sdfg::builder::StructuredSDFGBuilder& builder, sdfg::analysis::AnalysisManager& analysis_manager)
    : sdfg::visitor::StructuredSDFGVisitor(builder, analysis_manager) {}

bool LoopReport::accept(sdfg::structured_control_flow::Block& node) {
    auto& dataflow = node.dataflow();
    for (auto& dnode : dataflow.nodes()) {
        if (auto library_node = dynamic_cast<const sdfg::data_flow::LibraryNode*>(&dnode)) {
            if (this->report_.find(library_node->code().value()) == this->report_.end()) {
                this->report_[library_node->code().value()] = 0;
            }
            this->report_[library_node->code().value()]++;
        }
    }

    return false;
}

bool LoopReport::accept(sdfg::structured_control_flow::For& node) {
    if (this->report_.find("FOR") == this->report_.end()) {
        this->report_["FOR"] = 0;
    }
    this->report_["FOR"]++;

    if (this->report_.find(node.schedule_type().value()) == this->report_.end()) {
        this->report_[node.schedule_type().value()] = 0;
    }
    this->report_[node.schedule_type().value()]++;

    return false;
}

bool LoopReport::accept(sdfg::structured_control_flow::While& node) {
    if (this->report_.find("WHILE") == this->report_.end()) {
        this->report_["WHILE"] = 0;
    }
    this->report_["WHILE"]++;

    return false;
}

bool LoopReport::accept(sdfg::structured_control_flow::Map& node) {
    if (this->report_.find("MAP") == this->report_.end()) {
        this->report_["MAP"] = 0;
    }
    this->report_["MAP"]++;

    if (this->report_.find(node.schedule_type().value()) == this->report_.end()) {
        this->report_[node.schedule_type().value()] = 0;
    }
    this->report_[node.schedule_type().value()]++;

    return false;
}

bool LoopReport::accept(sdfg::structured_control_flow::Reduce& node) {
    if (this->report_.find("REDUCE") == this->report_.end()) {
        this->report_["REDUCE"] = 0;
    }
    this->report_["REDUCE"]++;

    if (this->report_.find(node.schedule_type().value()) == this->report_.end()) {
        this->report_[node.schedule_type().value()] = 0;
    }
    this->report_[node.schedule_type().value()]++;

    return false;
}

} // namespace codegen
} // namespace sdfg
