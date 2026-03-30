#pragma once

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/users.h"

namespace sdfg {
namespace analysis {

class DominanceAnalysis : public Analysis {
private:
    std::unordered_map<graph::Vertex, graph::Vertex> dom_tree_;
    std::unordered_map<graph::Vertex, graph::Vertex> pdom_tree_;

protected:
    void run(analysis::AnalysisManager& analysis_manager) override;

public:
    DominanceAnalysis(StructuredSDFG& sdfg);

    std::string name() const override { return "DominanceAnalysis"; }

    bool dominates(User& user1, User& user2);

    bool post_dominates(User& user1, User& user2);
};

} // namespace analysis
} // namespace sdfg
