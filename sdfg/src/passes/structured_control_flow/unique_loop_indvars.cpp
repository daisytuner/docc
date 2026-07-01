#include "sdfg/passes/structured_control_flow/unique_loop_indvars.h"

#include <string>
#include <unordered_set>

#include "sdfg/analysis/users.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {

namespace {

/**
 * @brief Renames induction variables so that every loop has a unique one.
 *
 * The traversal is a pre-order walk (a loop is visited before its body), so an
 * outer/earlier loop reserves its induction variable name before any nested or
 * later loop is considered.
 */
class UniqueLoopIndvarsVisitor : public visitor::NonStoppingStructuredSDFGVisitor {
private:
    std::unordered_set<std::string> used_indvars_;

    /**
     * @brief Checks whether the loop's induction variable is genuinely read.
     *
     * Uses the Users analysis: starting from the loop's update user (the write
     * to the induction variable performed at the end of each iteration), it looks
     * for read users of the same container that depend on that update and are not
     * the loop-control use of another loop (i.e. an actual data read).
     */
    bool indvar_is_read(structured_control_flow::StructuredLoop& loop, const std::string& name) {
        auto& users = analysis_manager_.get<analysis::Users>();

        auto* update_user = users.get_user(name, &loop, analysis::Use::WRITE, false, false, true);
        if (update_user == nullptr) {
            return false;
        }

        for (auto* user : users.all_uses_after(*update_user)) {
            if (user->use() != analysis::Use::READ || user->container() != name) {
                continue;
            }
            // Skip reads that are the init/condition/update of another loop.
            if (auto* for_user = dynamic_cast<analysis::ForUser*>(user)) {
                if (for_user->element() != &loop) {
                    continue;
                }
            }
            return true;
        }
        return false;
    }

    /**
     * @brief Checks whether the induction variable's value escapes the loop.
     *
     * A genuine (non loop-control) read of the same container that depends on the
     * loop's update user but lies outside the loop subtree consumes the final
     * induction-variable value. Renaming only rewrites uses within the loop, so
     * such an escaping read would be left dangling; in that case the loop must not
     * be renamed.
     */
    bool indvar_escapes(structured_control_flow::StructuredLoop& loop, const std::string& name) {
        auto& users = analysis_manager_.get<analysis::Users>();

        auto* update_user = users.get_user(name, &loop, analysis::Use::WRITE, false, false, true);
        if (update_user == nullptr) {
            return false;
        }

        // Reads of the induction variable that live inside the loop subtree.
        analysis::UsersView loop_view(users, loop);
        auto inner_reads = loop_view.reads(name);
        std::unordered_set<analysis::User*> inner(inner_reads.begin(), inner_reads.end());

        for (auto* user : users.all_uses_after(*update_user)) {
            if (user->use() != analysis::Use::READ || user->container() != name) {
                continue;
            }
            // Loop-control reads (init/condition/update) are not data escapes.
            if (dynamic_cast<analysis::ForUser*>(user) != nullptr) {
                continue;
            }
            // A read still inside the loop is just the body using the variable.
            if (inner.find(user) != inner.end()) {
                continue;
            }
            return true;
        }
        return false;
    }

    bool handle_loop(structured_control_flow::StructuredLoop& loop) {
        const std::string name = loop.indvar()->get_name();

        // First loop to claim this induction variable: just reserve the name.
        if (used_indvars_.insert(name).second) {
            return false;
        }

        // Name already in use by another loop. Only disambiguate if the variable
        // is actually read within this loop; otherwise the clash is harmless.
        if (!indvar_is_read(loop, name)) {
            return false;
        }

        // The induction variable is consumed after the loop: a local rename would
        // break that downstream user, so leave the loop untouched.
        if (indvar_escapes(loop, name)) {
            return false;
        }

        auto& sdfg = builder_.subject();
        std::string new_name = builder_.find_new_name(name + "_");
        builder_.add_container(new_name, sdfg.type(name));

        loop.replace(loop.indvar(), symbolic::symbol(new_name));
        used_indvars_.insert(new_name);

        // The rename rewrote the loop subtree, so user information is now stale.
        analysis_manager_.invalidate_all();
        return true;
    }

public:
    UniqueLoopIndvarsVisitor(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        : visitor::NonStoppingStructuredSDFGVisitor(builder, analysis_manager) {}

    bool accept(structured_control_flow::For& node) override { return handle_loop(node); }

    bool accept(structured_control_flow::Map& node) override { return handle_loop(node); }
};

} // namespace

UniqueLoopIndvars::UniqueLoopIndvars() : Pass() {}

std::string UniqueLoopIndvars::name() { return "UniqueLoopIndvars"; }

bool UniqueLoopIndvars::run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    UniqueLoopIndvarsVisitor visitor(builder, analysis_manager);
    return visitor.visit();
}

} // namespace passes
} // namespace sdfg
