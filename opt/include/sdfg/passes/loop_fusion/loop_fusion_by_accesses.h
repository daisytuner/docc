#pragma once

#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/passes/loop_fusion/fusion_types.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/assumptions.h"

namespace sdfg::passes::loop_fusion {

class LoopFusionByAccessWorker {
public:
    enum class FusionDirection {
        None = 0,
        ProducerIntoConsumer,
        ConsumerIntoProducer,
    };

protected:
    bool allow_init_hoist_ = true;

    LoopFusionByAccessWorker(bool allow_init_hoist = true) : allow_init_hoist_(allow_init_hoist) {}

    struct Plan {
        structured_control_flow::Map& first;
        structured_control_flow::StructuredLoop& second;

        FusionDirection direction_;
        std::vector<structured_control_flow::StructuredLoop*> producer_loops_;
        structured_control_flow::Sequence* producer_body_ = nullptr;
        structured_control_flow::Block* producer_block_ = nullptr;
        FusionLoopCandidate* producer_fusion_candidate_ = nullptr;
        std::vector<structured_control_flow::StructuredLoop*> consumer_loops_;
        structured_control_flow::Sequence* consumer_body_ = nullptr;
        FusionLoopCandidate* consumer_fusion_candidate_ = nullptr;
        std::vector<FusionRegCandidate> fusion_candidates_;

        Plan(structured_control_flow::Map& first, structured_control_flow::StructuredLoop& second)
            : first(first), second(second), direction_() {}

        // Case 2 (init-into-reduction): when true, the producer is hoisted to the
        // reduction's outer parallel band (before the innermost sequential loop) and
        // keeps writing the accumulator array, instead of being scalarized and inlined
        // element-by-element inside the reduction loop (Case 1).
        bool init_hoist_ = false;
        structured_control_flow::StructuredLoop* consumer_target_loop() const;

        structured_control_flow::Sequence& consumer_target_sequence() const;

        // The outer parallel-band body that hosts the hoisted init (Case 2 only).
        structured_control_flow::Sequence* hoist_body_ = nullptr;
        // The loops being fused match domains exactly, so we can remove the original loop, if we do
        bool domains_match = false;
    };

    virtual ~LoopFusionByAccessWorker() = default;

    virtual analysis::LoopAnalysis& get_loop_analysis() = 0;

    virtual FusionLoopCandidate* get_fuse_candidate(structured_control_flow::StructuredLoop& loop) = 0;

    virtual builder::StructuredSDFGBuilder& builder() = 0;

    /**
     * Moved non-loop contents from first_current to second_current.
     * @param plan
     * @param first_current
     * @param second_current
     */
    virtual void update_copied_leaf_contents_from_first_to_second(
        const Plan& plan, FusionLoopCandidate* first_current, FusionLoopCandidate* second_current
    ) = 0;

public:
    static std::vector<std::pair<symbolic::Symbol, symbolic::Expression>> solve_subsets(
        const data_flow::Subset& producer_subset,
        const data_flow::Subset& consumer_subset,
        const std::vector<structured_control_flow::StructuredLoop*>& producer_loops,
        const std::vector<structured_control_flow::StructuredLoop*>& consumer_loops,
        const symbolic::Assumptions& producer_assumptions,
        const symbolic::Assumptions& consumer_assumptions,
        bool invert_range_check = false
    );

protected:
    struct FusionRegs {
        std::unordered_set<RegId> fusion_regs;
        std::unordered_set<RegId> second_outputs;
        bool conflicts;
    };

    FusionRegs find_fusion_regs(const FusionLoopCandidate& first, const FusionLoopCandidate& second);


    /**
     * Returns a vector of loops from `top` to its innermost StructuredLoop,
     * assuming they are all perfectly nested
     */
    std::vector<structured_control_flow::StructuredLoop*>
    collect_structured_sub_tree(structured_control_flow::StructuredLoop& top);

    std::vector<structured_control_flow::StructuredLoop*>
    collect_loop_parents(structured_control_flow::Sequence* sequence, structured_control_flow::StructuredLoop* loop);

public:
    std::unique_ptr<Plan> try_create_fusion_by_access_plan(
        FusionLoopCandidate& first, FusionLoopCandidate& second, bool domains_match = false
    );

    ComplexFusionResult apply_fusion_by_access_plan(std::unique_ptr<Plan> plan);

protected:
    virtual ComplexFusionResult apply_producer_into_consumer(Plan& plan);
    virtual ComplexFusionResult apply_consumer_into_producer(Plan& plan);

public:
    ComplexFusionResult try_fuse_by_access(FusionLoopCandidate& first, FusionLoopCandidate& second, bool domains_match);
};


} // namespace sdfg::passes::loop_fusion
