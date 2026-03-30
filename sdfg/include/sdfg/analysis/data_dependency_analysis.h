#pragma once

#include <unordered_map>
#include <unordered_set>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/dominance_analysis.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/symbolic/maps.h"

namespace sdfg {
namespace analysis {

enum LoopCarriedDependency {
    LOOP_CARRIED_DEPENDENCY_READ_WRITE,
    LOOP_CARRIED_DEPENDENCY_WRITE_WRITE,
};

/**
 * @brief Extended loop-carried dependency information including distance vectors.
 *
 * Combines the dependency type (read-write or write-write) with the full
 * ISL delta set representing all possible iteration-distance vectors.
 */
struct LoopCarriedDependencyInfo {
    LoopCarriedDependency type;
    symbolic::maps::DependenceDeltas deltas;
};

/**
 * @brief Computes the read-after-write relation between users.
 *
 * This analysis computes for each definition (write) the set of users (reads).
 *
 * A definition open when a container is written to.
 * A definition remains open until a new definition to the same container is encountered,
 * that dominates the previous definition in terms of:
 * - control-flow
 * - subsets (multi-dimensional containers)
 *
 */
class DataDependencyAnalysis : public Analysis {
    friend class AnalysisManager;

private:
    structured_control_flow::Sequence& node_;
    bool detailed_;

    std::unordered_map<std::string, std::unordered_map<User*, std::unordered_set<User*>>> results_;

    std::unordered_map<
        structured_control_flow::StructuredLoop*,
        std::unordered_map<std::string, LoopCarriedDependencyInfo>>
        loop_carried_dependencies_;

    std::list<std::unique_ptr<User>> undefined_users_;

    symbolic::maps::DependenceDeltas loop_deltas(
        User& previous,
        User& current,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::StructuredLoop& loop
    );

    bool supersedes_restrictive(User& previous, User& current, analysis::AnalysisManager& analysis_manager);

    bool intersects(User& previous, User& current, analysis::AnalysisManager& analysis_manager);

    bool closes(analysis::AnalysisManager& analysis_manager, User& previous, User& current, bool requires_dominance);

    bool depends(analysis::AnalysisManager& analysis_manager, User& previous, User& current);

public:
    DataDependencyAnalysis(StructuredSDFG& sdfg, bool detailed = false);

    DataDependencyAnalysis(StructuredSDFG& sdfg, structured_control_flow::Sequence& node, bool detailed = false);

    std::string name() const override { return "DataDependencyAnalysis"; }

    void run(analysis::AnalysisManager& analysis_manager) override;

    /****** Visitor API ******/

    void visit_block(
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::Block& block,
        std::unordered_set<User*>& undefined,
        std::unordered_map<User*, std::unordered_set<User*>>& open_definitions,
        std::unordered_map<User*, std::unordered_set<User*>>& closed_definitions
    );

    void visit_for(
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::StructuredLoop& for_loop,
        std::unordered_set<User*>& undefined,
        std::unordered_map<User*, std::unordered_set<User*>>& open_definitions,
        std::unordered_map<User*, std::unordered_set<User*>>& closed_definitions
    );

    void visit_if_else(
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::IfElse& if_loop,
        std::unordered_set<User*>& undefined,
        std::unordered_map<User*, std::unordered_set<User*>>& open_definitions,
        std::unordered_map<User*, std::unordered_set<User*>>& closed_definitions
    );

    void visit_while(
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::While& while_loop,
        std::unordered_set<User*>& undefined,
        std::unordered_map<User*, std::unordered_set<User*>>& open_definitions,
        std::unordered_map<User*, std::unordered_set<User*>>& closed_definitions
    );

    void visit_return(
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::Return& return_statement,
        std::unordered_set<User*>& undefined,
        std::unordered_map<User*, std::unordered_set<User*>>& open_definitions,
        std::unordered_map<User*, std::unordered_set<User*>>& closed_definitions
    );

    void visit_sequence(
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::Sequence& sequence,
        std::unordered_set<User*>& undefined,
        std::unordered_map<User*, std::unordered_set<User*>>& open_definitions,
        std::unordered_map<User*, std::unordered_set<User*>>& closed_definitions
    );

    /****** Defines & Use ******/

    /**
     * @brief Get the users (reads) of a definition (write).
     *
     * @param write the definition (write)
     * @return The users (reads) of the definition.
     */
    std::unordered_set<User*> defines(User& write);

    /**
     * @brief Get all definitions (writes) and their users (reads).
     *
     * @param The container of the definitions.
     * @return The definitions and their users.
     */
    std::unordered_map<User*, std::unordered_set<User*>> definitions(const std::string& container);

    /**
     * @brief Get all definitions (writes) for each user (reads).
     *
     * @param The container of the definitions.
     * @return The users (reads) and their definitions (writes).
     */
    std::unordered_map<User*, std::unordered_set<User*>> defined_by(const std::string& container);

    /**
     * @brief Get all definitions (writes) for a user (read).
     *
     * @param The user (read).
     * @return The definitions (writes) of the user.
     */
    std::unordered_set<User*> defined_by(User& read);

    bool is_undefined_user(User& user) const;

    /****** Loop-carried dependencies ******/

    bool available(structured_control_flow::StructuredLoop& loop) const;

    const std::unordered_map<std::string, LoopCarriedDependencyInfo>& dependencies(structured_control_flow::StructuredLoop&
                                                                                       loop) const;
};

} // namespace analysis
} // namespace sdfg
