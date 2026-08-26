#pragma once

#include "sdfg/analysis/arguments_analysis.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg::passes::loop_fusion {

struct LoopFusionConfig {
    bool allow_init_hoist = true;
    bool map_fusion_by_domain = true;
    bool map_fusion_by_access = true;
};

struct FusionRegCandidate {
    std::string container;
    data_flow::Subset consumer_subset;
    std::vector<std::pair<symbolic::Symbol, symbolic::Expression>> index_mappings;
    bool integrated_rle = false;
};

typedef std::string RegId;

struct FusionArgCommonBlock {
    structured_control_flow::Block* common_block = nullptr;
    bool block_conflict = false;

    bool merge_into(structured_control_flow::Block* other_block, bool write_not_read);

    bool merge_into(const FusionArgCommonBlock& other);
};

struct FusionArgCommonAccesses {
    bool subsets_conflict = false;
    std::optional<data_flow::Subset> common_subset;
    FusionArgCommonBlock wr_block;
    FusionArgCommonBlock rd_block;

    bool merge_into(
        structured_control_flow::Block* block,
        const std::optional<data_flow::Subset>& subset,
        bool not_understood,
        bool write_not_read,
        const symbolic::ExpressionMapping* lower_indvars = nullptr
    );

    bool merge_into(const FusionArgCommonAccesses& other, const symbolic::ExpressionMapping* lower_indvars = nullptr);

    bool subset_conflicts_with(
        const FusionArgCommonAccesses& other_common_acceses, symbolic::ExpressionMapping& lower_indvars
    ) const;

protected:
    bool merge_subset(
        const std::optional<data_flow::Subset>& subset,
        bool not_understood,
        const symbolic::ExpressionMapping* lower_indvars
    );
};

/**
 * Represents one "Argument" per ArgumentAnalysis of a FusionCandidate. I.e. contains Live-Ins, Live-Outs of the loop
 * @property arg contains the ArgumentAnalysis metadata that identifies how the arg is used (read, written) inside the
 * loop
 * @property local_access for accesses that happen directly inside the candidate (not in further nested candidates)
 */
struct FusionArg {
    analysis::RegionArgument arg;
    FusionArgCommonAccesses local_access;
    FusionArgCommonAccesses nested_access;

    FusionArg(const analysis::RegionArgument& arg) : arg(arg) {}
    FusionArg(const FusionArg& arg) : arg(arg.arg), local_access(arg.local_access), nested_access(arg.nested_access) {}

    bool saw_access_locally() const;
};

/**
 * Represents pre-cached data for 1 Loop that fullfills the prerequisits for fusing with another candidate
 * Goal is to collect every harder to get data once for all involved loops.
 * Because we likely need to compare a candidate against many others, reusing the same data.
 * Also structured in a way, that the data can be merged & updated if fusing between candidates actually happens
 *
 * Also, since not every loop is eligible to be a candidate, this flattens the loop-tree.
 * Relevant data of nested loops that are not themselves candidates will need to be folded into its nearest parent
 * candidate
 */
struct FusionLoopCandidate {
    structured_control_flow::StructuredLoop* loop;
    const symbolic::Assumption* indvar_boundaries;
    symbolic::Assumptions assumptions;
    bool is_map = false;
    bool is_by_domain_candidate = false;
    std::unordered_map<RegId, FusionArg> args;
    bool incompatible = false;
    bool nested_incompatible = false;

    FusionLoopCandidate(
        structured_control_flow::StructuredLoop* loop,
        const symbolic::Assumption* indvar_boundaries,
        symbolic::Assumptions assumptions,
        bool is_map = false,
        bool is_by_domain_candidate = false
    )
        : loop(loop), indvar_boundaries(indvar_boundaries), assumptions(std::move(assumptions)), is_map(is_map),
          is_by_domain_candidate(is_by_domain_candidate) {}

    void non_indvar_writes();

    void aliasing_encountered();

    void replace(const symbolic::ExpressionMapping& mapping);
};

class PatternHandler {
public:
    virtual ~PatternHandler() = default;

    struct MatchResult {
        bool removed_first = false;
        bool visit_second_body = false;
        structured_control_flow::ControlFlowNode* second_root_replacement = nullptr;
    };

    virtual MatchResult match(
        structured_control_flow::StructuredLoop& first,
        structured_control_flow::StructuredLoop& second,
        bool no_uses_between
    ) = 0;

    virtual bool check_no_overlap(
        const structured_control_flow::StructuredLoop& first,
        const structured_control_flow::StructuredLoop& second,
        const std::unordered_set<RegId>& skipped_containers
    ) = 0;
};

struct ComplexFusionResult {
    PatternHandler::MatchResult pattern_result;
    bool fused = false;
};

} // namespace sdfg::passes::loop_fusion
