#pragma once

#include <unordered_map>
#include <unordered_set>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/dominance_analysis.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/structured_sdfg.h"

namespace sdfg {
namespace analysis {

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

    std::unordered_map<std::string, std::unordered_map<User*, std::unordered_set<User*>>> results_;

    // Per-loop boundary snapshots taken at the moment of `visit_for`.
    // - first  = `undefined_for` (reads inside the loop body that look outside
    //            the loop body for their definition; equivalently, upward-exposed
    //            reads at the loop's boundary).
    // - second = `open_definitions_for` (writes inside the loop body that may
    //            escape the loop body or feed a future iteration; equivalently,
    //            live-out definitions at the loop's boundary).
    // This is the primitive consumed by `LoopCarriedDependencyAnalysis`.
    std::unordered_map<
        structured_control_flow::StructuredLoop*,
        std::pair<std::unordered_set<User*>, std::unordered_map<User*, std::unordered_set<User*>>>>
        loop_boundaries_;

    std::list<std::unique_ptr<User>> undefined_users_;

    // When false (default), the symbolic-subset/disjointness helpers below
    // (`supersedes_restrictive`, `intersects`, `closes`, `depends`) take
    // conservative shortcuts and skip the expensive ISL queries. Sound but
    // less precise. `LoopCarriedDependencyAnalysis` constructs its own
    // detailed instance manually with `detailed_ = true` to obtain the
    // precise boundary information it requires.
    bool detailed_ = false;

    bool supersedes_restrictive(User& previous, User& current, analysis::AnalysisManager& analysis_manager);

    bool intersects(User& previous, User& current, analysis::AnalysisManager& analysis_manager);

    bool closes(analysis::AnalysisManager& analysis_manager, User& previous, User& current, bool requires_dominance);

    bool depends(analysis::AnalysisManager& analysis_manager, User& previous, User& current);

    // True iff every read-subset of `current` is contained in some single
    // open writer's subset (per-subset coverage). For scalars / undefined
    // users / symbol uses this degrades to `depends`. Used to detect partial
    // coverage that `depends` (intersect-any) would otherwise hide, which
    // would cause upward-exposed reads to be silently swallowed.
    bool fully_covered(
        analysis::AnalysisManager& analysis_manager,
        User& current,
        const std::unordered_map<User*, std::unordered_set<User*>>& open_definitions
    );

public:
    DataDependencyAnalysis(StructuredSDFG& sdfg);

    DataDependencyAnalysis(StructuredSDFG& sdfg, structured_control_flow::Sequence& node);

    // Enable detailed symbolic subset/disjointness checks. Off by default;
    // `LoopCarriedDependencyAnalysis` flips it on for its own manually
    // constructed instance.
    void set_detailed(bool detailed) { detailed_ = detailed; }

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

    /****** Loop boundaries (consumed by LoopCarriedDependencyAnalysis) ******/

    /**
     * @brief Whether boundary snapshots are available for the given loop.
     */
    bool has_loop_boundary(structured_control_flow::StructuredLoop& loop) const;

    /**
     * @brief Reads inside `loop`'s body that are upward-exposed at the loop
     * boundary (i.e. not satisfied by a definition inside the body).
     */
    const std::unordered_set<User*>& upward_exposed_reads(structured_control_flow::StructuredLoop& loop) const;

    /**
     * @brief Writes inside `loop`'s body that escape the body (i.e. are
     * live-out at the loop boundary or feed a future iteration of `loop`).
     */
    const std::unordered_map<User*, std::unordered_set<User*>>&
    escaping_definitions(structured_control_flow::StructuredLoop& loop) const;
};

} // namespace analysis
} // namespace sdfg
