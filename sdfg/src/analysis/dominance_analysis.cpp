#include "sdfg/analysis/dominance_analysis.h"

namespace sdfg {
namespace analysis {

DominanceAnalysis::DominanceAnalysis(StructuredSDFG& sdfg) : Analysis(sdfg) {
}

void DominanceAnalysis::run(AnalysisManager& analysis_manager) {
    this->dom_tree_.clear();
    this->pdom_tree_.clear();

    auto& users_analysis = analysis_manager.get<analysis::Users>();

    this->dom_tree_ = graph::dominator_tree(users_analysis.graph_, users_analysis.source_->vertex_);
    this->pdom_tree_ = graph::post_dominator_tree(users_analysis.graph_);
}

bool DominanceAnalysis::dominates(User& user1, User& user2) {
    auto dominator = this->dom_tree_.at(user2.vertex_);
    while (dominator != boost::graph_traits<graph::Graph>::null_vertex()) {
        if (dominator == user1.vertex_) {
            return true;
        }
        dominator = this->dom_tree_.at(dominator);
    }
    return false;
};

bool DominanceAnalysis::post_dominates(User& user1, User& user2) {
    auto dominator = this->pdom_tree_.at(user2.vertex_);
    while (dominator != boost::graph_traits<graph::Graph>::null_vertex()) {
        if (dominator == user1.vertex_) {
            return true;
        }
        dominator = this->pdom_tree_.at(dominator);
    }
    return false;
};

} // namespace analysis
} // namespace sdfg
