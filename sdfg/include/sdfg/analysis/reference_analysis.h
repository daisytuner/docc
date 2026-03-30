#pragma once

#include <unordered_map>
#include <unordered_set>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/structured_sdfg.h"

namespace sdfg {
namespace analysis {

class ReferenceAnalysis : public Analysis {
    friend class AnalysisManager;

private:
    structured_control_flow::Sequence& node_;
    std::unordered_map<std::string, std::unordered_map<User*, std::unordered_set<User*>>> results_;

public:
    ReferenceAnalysis(StructuredSDFG& sdfg);

    std::string name() const override { return "ReferenceAnalysis"; }

    void run(analysis::AnalysisManager& analysis_manager) override;

    /****** Visitor API ******/

    void visit_block(
        analysis::Users& users,
        structured_control_flow::Block& block,
        std::unordered_set<User*>& undefined,
        std::unordered_map<User*, std::unordered_set<User*>>& open_definitions,
        std::unordered_map<User*, std::unordered_set<User*>>& closed_definitions
    );


    void visit_sequence(
        analysis::Users& users,
        structured_control_flow::Sequence& sequence,
        std::unordered_set<User*>& undefined,
        std::unordered_map<User*, std::unordered_set<User*>>& open_definitions,
        std::unordered_map<User*, std::unordered_set<User*>>& closed_definitions
    );

    void visit_if_else(
        analysis::Users& users,
        structured_control_flow::IfElse& if_loop,
        std::unordered_set<User*>& undefined,
        std::unordered_map<User*, std::unordered_set<User*>>& open_definitions,
        std::unordered_map<User*, std::unordered_set<User*>>& closed_definitions
    );

    void visit_for(
        analysis::Users& users,
        structured_control_flow::StructuredLoop& for_loop,
        std::unordered_set<User*>& undefined,
        std::unordered_map<User*, std::unordered_set<User*>>& open_definitions,
        std::unordered_map<User*, std::unordered_set<User*>>& closed_definitions
    );

    void visit_while(
        analysis::Users& users,
        structured_control_flow::While& while_loop,
        std::unordered_set<User*>& undefined,
        std::unordered_map<User*, std::unordered_set<User*>>& open_definitions,
        std::unordered_map<User*, std::unordered_set<User*>>& closed_definitions
    );

    void visit_return(
        analysis::Users& users,
        structured_control_flow::Return& return_statement,
        std::unordered_set<User*>& undefined,
        std::unordered_map<User*, std::unordered_set<User*>>& open_definitions,
        std::unordered_map<User*, std::unordered_set<User*>>& closed_definitions
    );

    std::unordered_set<User*> defines(User& move);

    std::unordered_map<User*, std::unordered_set<User*>> definitions(const std::string& container);

    std::unordered_map<User*, std::unordered_set<User*>> defined_by(const std::string& container);

    std::unordered_set<User*> defined_by(User& user);
};

} // namespace analysis
} // namespace sdfg
