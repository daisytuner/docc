#pragma once

#include <memory>
#include <string>
#include <vector>

#include "sdfg/analysis/memory_layout_analysis.h"
#include "sdfg/passes/pass.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/type.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg::passes {

/**
 * @brief Replaces perfectly-nested Map loops that zero out an entire array with a single memset.
 *
 * The pass detects a perfect nest of Map loops whose only effect is to overwrite the
 * complete contents of a (multi-dimensional) array container with the literal value 0
 * (and nothing else). Such a nest is semantically equivalent to a `memset(A, 0, sizeof(A))`
 * and is rewritten to a single stdlib::MemsetNode.
 *
 * A nest qualifies when:
 * - Each loop level is a Map with `init == 0`, unit stride and an extractable exclusive
 *   upper bound; the nest is perfect (each Map body contains exactly the next level).
 * - The innermost body is a single Block whose only computation assigns the constant 0
 *   to one element of the array, indexed exactly by the induction variables of the nest.
 * - The MemoryLayoutAnalysis tile for the target container at the outermost Map covers the
 *   whole array contiguously from offset 0 (bounded dimensions match the declared extent,
 *   and the covered elements form a dense block), guaranteeing the whole array is covered.
 *   Relying on the layout analysis also transparently handles linearized accesses such as
 *   `A[i*M + j]`, which are delinearized to the underlying multi-dimensional shape.
 */
class ZeroFillToMemsetPass : public sdfg::passes::Pass {
public:
    struct State {
        size_t applied = 0;
    };

    std::string name() override { return "ZeroFillToMemset"; }

    bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;
};

class ZeroFillToMemsetVisitor : public sdfg::visitor::ActualStructuredSDFGVisitor {
public:
    struct Candidate {
        structured_control_flow::Map* map;
        std::string array;
        symbolic::Expression num;
        std::unique_ptr<types::IType> ptr_type;
    };

private:
    builder::StructuredSDFGBuilder& builder_;
    ZeroFillToMemsetPass::State& state_;
    analysis::MemoryLayoutAnalysis& memory_layout_analysis_;
    std::vector<Candidate> candidates_;

    bool match(structured_control_flow::Map& node, Candidate& candidate);

public:
    ZeroFillToMemsetVisitor(
        builder::StructuredSDFGBuilder& builder,
        ZeroFillToMemsetPass::State& state,
        analysis::MemoryLayoutAnalysis& memory_layout_analysis
    );

    bool visit(structured_control_flow::Map& node) override;

    void apply();
};

} // namespace sdfg::passes
