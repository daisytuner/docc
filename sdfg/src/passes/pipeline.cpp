#include "sdfg/passes/pipeline.h"

#include <chrono>

#include "sdfg/helpers/helpers.h"
#include "sdfg/passes/dataflow/tasklet_fusion.h"
#include "sdfg/passes/dataflow/trivial_reference_conversion.h"
#include "sdfg/passes/schedules/expansion_pass.h"
#include "sdfg/passes/statistics.h"

namespace sdfg {
namespace passes {

Pipeline::Pipeline(const std::string& name)
    : Pass(), name_(name) {

      };

std::string Pipeline::name() { return this->name_; };

size_t Pipeline::size() const { return this->passes_.size(); };

bool Pipeline::run(builder::SDFGBuilder& builder) {
    std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds> start;
    if (PipelineStatistics::instance().enabled()) {
        start = std::chrono::high_resolution_clock::now();
#ifndef NDEBUG
        DEBUG_PRINTLN("Started SDFG Pipeline '" << this->name() << "' on '" << builder.subject().name() << "'");
#endif
    }

    bool applied = false;
    bool applied_pipeline;
    do {
        applied_pipeline = false;
        for (auto& pass : this->passes_) {
            bool applied_pass = false;
            do {
                applied_pass = pass->run(builder);
                applied_pipeline |= applied_pass;
            } while (applied_pass);
        }
        applied |= applied_pipeline;
    } while (applied_pipeline);

    if (PipelineStatistics::instance().enabled()) {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        PipelineStatistics::instance().add_sdfg_pipeline(this->name(), duration);
#ifndef NDEBUG
        DEBUG_PRINTLN("Finished SDFG Pipeline '" << this->name() << "' in " << duration << " ms");
#endif
    }

    return applied;
};

bool Pipeline::run(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds> start;
    if (PipelineStatistics::instance().enabled()) {
        start = std::chrono::high_resolution_clock::now();
#ifndef NDEBUG
        DEBUG_PRINTLN("Started Structured SDFG Pipeline '" << this->name() << "' on '" << builder.subject().name() << "'");
#endif
    }

    bool applied = false;
    bool applied_pipeline;
    do {
        applied_pipeline = false;
        for (auto& pass : this->passes_) {
            bool applied_pass = false;
            do {
                applied_pass = pass->run(builder, analysis_manager);
                applied_pipeline |= applied_pass;
            } while (applied_pass);
        }
        applied |= applied_pipeline;
    } while (applied_pipeline);

    if (PipelineStatistics::instance().enabled()) {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        PipelineStatistics::instance().add_structured_sdfg_pipeline(this->name(), duration);
#ifndef NDEBUG
        DEBUG_PRINTLN("Finished Structured SDFG Pipeline '" << this->name() << "' in " << duration << " ms");
#endif
    }

    return applied;
};

Pipeline Pipeline::dataflow_simplification() {
    Pipeline p("DataflowSimplification");

    p.register_pass<BlockFusionPass>();
    p.register_pass<TaskletFusionPass>();
    p.register_pass<SequenceFusion>();

    return p;
};

Pipeline Pipeline::symbolic_simplification() {
    Pipeline p("SymbolicSimplification");

    p.register_pass<SymbolPropagation>();

    return p;
};

Pipeline Pipeline::dead_code_elimination() {
    Pipeline p("DeadCodeElimination");

    p.register_pass<DeadCFGElimination>();
    p.register_pass<SequenceFusion>();

    return p;
};

Pipeline Pipeline::expression_combine() {
    Pipeline p("ExpressionCombine");

    p.register_pass<SymbolPropagation>();
    p.register_pass<ConstantPropagation>();
    p.register_pass<DeadDataElimination>();
    p.register_pass<SymbolEvolution>();
    p.register_pass<TaskletFusionPass>();

    return p;
};

Pipeline Pipeline::memlet_combine() {
    Pipeline p("MemletCombine");

    p.register_pass<ReferencePropagation>();
    p.register_pass<DeadReferenceElimination>();
    p.register_pass<ByteReferenceElimination>();
    p.register_pass<TrivialReferenceConversionPass>();

    return p;
};

Pipeline Pipeline::controlflow_simplification() {
    Pipeline p("ControlFlowSimplification");

    p.register_pass<DeadCFGElimination>();
    p.register_pass<BlockFusionPass>();
    p.register_pass<SequenceFusion>();
    p.register_pass<ConditionEliminationPass>();

    return p;
};

Pipeline Pipeline::data_parallelism() {
    Pipeline p("DataParallelism");

    p.register_pass<For2MapPass>();
    p.register_pass<SymbolPropagation>();
    p.register_pass<DeadDataElimination>();

    return p;
};

Pipeline Pipeline::memory() {
    Pipeline p("Memory");

    p.register_pass<AllocationManagementPass>();

    return p;
};

Pipeline Pipeline::expansion() {
    Pipeline p("Expansion");

    p.register_pass<ExpansionPass>();

    return p;
};

} // namespace passes
} // namespace sdfg
