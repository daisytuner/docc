
#include "sdfg/analysis/mem_access_range_analysis.h"

#include <stdbool.h>
#include <symengine/basic.h>
#include <symengine/functions.h>
#include <symengine/infinity.h>
#include <symengine/number.h>
#include <symengine/symengine_rcp.h>

#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/mem_access_range_analysis_internal.h"
#include "sdfg/analysis/users.h"
#include "sdfg/helpers/helpers.h"
#include "sdfg/symbolic/extreme_values.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace analysis {

MemAccessRanges::MemAccessRanges(StructuredSDFG& sdfg) : Analysis(sdfg), graph_() {}

void MemAccessRanges::
    run(structured_control_flow::ControlFlowNode& node, std::unordered_set<std::string> target_containers) {
    auto& users = analysis_manager_->get<Users>();
    auto& assumptions_analysis = analysis_manager_->get<AssumptionsAnalysis>();

    auto builder = MemAccessRangesBuilder(sdfg_, node, users, assumptions_analysis);

    auto& worklist = builder.worklist_;

    // Initialize worklist with containers
    for (const auto& container : target_containers) {
        worklist.push_back(new WorkItem{&container});
    }

    // Iterate over all variables and their users
    while (!worklist.empty()) {
        auto* workItem = worklist.front();
        builder.process_workItem(workItem);
        worklist.pop_front();
        delete workItem;
    }

    this->ranges_.insert_or_assign(&node, std::move(builder.ranges_));
}

void MemAccessRanges::run(analysis::AnalysisManager& analysis_manager) {
    this->analysis_manager_ = &analysis_manager;
    std::unordered_set<std::string> containers;

    // Collect argument names
    for (auto& arg : sdfg_.arguments()) {
        if (sdfg_.type(arg).type_id() != types::TypeID::Scalar) {
            containers.insert(arg);
        }
    }

    // Collect external names
    for (auto& ext : sdfg_.externals()) {
        if (sdfg_.type(ext).type_id() != types::TypeID::Scalar) {
            containers.insert(ext);
        }
    }

    this->run(sdfg_.root(), containers);
}

const MemAccessRange* MemAccessRanges::get(const std::string& varName) const {
    auto ranges = this->ranges_.find(&sdfg_.root());
    if (ranges == this->ranges_.end()) {
        return nullptr;
    }
    auto res = ranges->second.find(varName);
    if (res != ranges->second.end()) {
        return &res->second;
    } else {
        return nullptr;
    }
}

const MemAccessRange* MemAccessRanges::
    get(const std::string& varName,
        structured_control_flow::ControlFlowNode& node,
        std::unordered_set<std::string> target_nodes) {
    auto ranges = this->ranges_.find(&node);
    this->run(node, target_nodes);
    ranges = this->ranges_.find(&node);
    if (ranges == this->ranges_.end()) {
        return nullptr;
    }
    auto res = ranges->second.find(varName);
    if (res != ranges->second.end()) {
        return &res->second;
    } else {
        return nullptr;
    }
}

MemAccessRange::MemAccessRange(
    const std::string& name,
    bool saw_read,
    bool saw_write,
    bool undefined,
    const std::vector<std::pair<symbolic::Expression, symbolic::Expression>>&& dims
)
    : name_(name), saw_read_(saw_read), saw_write_(saw_write), undefined_(undefined), dims_(dims) {}

const std::string& MemAccessRange::get_name() const { return name_; }

bool MemAccessRange::saw_read() const { return saw_read_; }
bool MemAccessRange::saw_write() const { return saw_write_; }
bool MemAccessRange::is_undefined() const { return undefined_; }

const std::vector<std::pair<symbolic::Expression, symbolic::Expression>>& MemAccessRange::dims() const { return dims_; }

void MemAccessRangesBuilder::process_workItem(WorkItem* item) {
    analysis::UsersView users_(users_analysis_, node_);

    const auto* varName = item->var_name;

    const auto& reads = users_.reads(*varName);
    process_direct_users(item, false, reads);

    const auto& writes = users_.writes(*varName);
    process_direct_users(item, true, writes);

    const auto& views = users_.views(*varName);
    if (!views.empty()) {
        DEBUG_PRINTLN("Found views for " << *varName << " => not rangeable!");
        item->undefined = true;
    }

    const auto& moves = users_.moves(*varName);
    if (!moves.empty()) {
        DEBUG_PRINTLN("Found moves for " << *varName << " => not rangeable!");
        item->undefined = true;
    }

    if (!item->dims.empty()) {
        std::vector<std::pair<symbolic::Expression, symbolic::Expression>> finalDims;
        finalDims.reserve(item->dims.size());

        for (auto& dim : item->dims) {
            auto& lowerExprs = std::get<0>(dim);
            bool isLowerUndefined = std::get<1>(dim);
            symbolic::Expression lb = (!lowerExprs.empty() && !isLowerUndefined)
                                          ? SymEngine::min(lowerExprs)
                                          : SymEngine::RCP<const SymEngine::Basic>();
            auto& upperExprs = std::get<2>(dim);
            bool isUpperUndefined = std::get<3>(dim);
            symbolic::Expression ub = (!upperExprs.empty() && !isUpperUndefined)
                                          ? SymEngine::max(upperExprs)
                                          : SymEngine::RCP<const SymEngine::Basic>();

            if (lb.is_null() || ub.is_null()) {
                item->undefined = true;
            }
            if (!lb.is_null() && SymEngine::is_a<SymEngine::Infty>(*lb)) {
                lb = SymEngine::null;
                item->undefined = true;
            }
            if (!ub.is_null() && SymEngine::is_a<SymEngine::Infty>(*ub)) {
                ub = SymEngine::null;
                item->undefined = true;
            }

            finalDims.emplace_back(std::move(lb), std::move(ub));
        }

        this->ranges_.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(*varName),
            std::forward_as_tuple(*varName, item->saw_read, item->saw_write, item->undefined, std::move(finalDims))
        );
    }
}

void MemAccessRangesBuilder::process_direct_users(WorkItem* item, bool is_write, std::vector<User*> accesses) {
    for (auto& access : accesses) {
        // The actual range analysis replaces symbols used in subsets
        // by their lower/upper bounds according to the assumptions analysis.
        // For this, we take the immediate scope to get the richest assumptions.
        const auto& user_scope = analysis::Users::scope(access);
        auto assums = assumptions_analysis_.get(*user_scope, false);

        // The final expression must be an expression w.r.t parameters,
        // i.e., constant symbols w.r.t the actual node.
        // Note we can compute this more efficiently once, but
        // we want to move this to the assumptions analysis anyway
        analysis::UsersView users_view(users_analysis_, node_);
        symbolic::SymbolSet params;
        for (auto& user : users_view.uses()) {
            if (user->container() == symbolic::__nullptr__()->get_name()) {
                continue;
            }
            auto& type = sdfg_.type(user->container());
            if (type.type_id() != types::TypeID::Scalar) {
                continue;
            }
            auto& scalar_type = static_cast<const types::Scalar&>(type);
            if (!types::is_integer(scalar_type.primitive_type())) {
                continue;
            }
            if (users_view.writes(user->container()).size() > 0) {
                continue;
            }
            params.insert(symbolic::symbol(user->container()));
        }

        item->saw_read |= !is_write;
        item->saw_write |= is_write;

        auto subsets = access->subsets();
        for (const auto& subset : subsets) {
            auto subsetDims = subset.size();
            item->dims.reserve(subsetDims);
            for (size_t i = item->dims.size(); i < subsetDims; ++i) {
                item->dims.emplace_back(std::make_tuple<
                                        std::vector<symbolic::Expression>,
                                        bool,
                                        std::vector<symbolic::Expression>,
                                        bool>({}, false, {}, false));
            }
            int dimIdx = 0;
            for (auto& dim : subset) {
                auto lb = symbolic::minimum(dim, params, assums, true);
                auto ub = symbolic::maximum(dim, params, assums, true);

                if (lb.is_null() || symbolic::has<SymEngine::Infty>(lb)) {
                    std::get<1>(item->dims[dimIdx]) = true;
                } else {
                    std::get<0>(item->dims[dimIdx]).push_back(lb);
                }
                if (ub.is_null() || symbolic::has<SymEngine::Infty>(ub)) {
                    std::get<3>(item->dims[dimIdx]) = true;
                } else {
                    std::get<2>(item->dims[dimIdx]).push_back(ub);
                }

                ++dimIdx;
            }
        }
    }
}

} // namespace analysis
} // namespace sdfg
