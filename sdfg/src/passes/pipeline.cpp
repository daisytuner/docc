#include "sdfg/passes/pipeline.h"

#include <chrono>

#include "sdfg/helpers/helpers.h"
#include "sdfg/passes/dataflow/tasklet_fusion.h"
#include "sdfg/passes/dataflow/trivial_reference_conversion.h"
#include "sdfg/passes/schedules/expansion_pass.h"
#include "sdfg/passes/statistics.h"
#include "sdfg/passes/structured_control_flow/block_fusion.h"
#include "sdfg/visualizer/dot_visualizer.h"

namespace sdfg {
namespace passes {

Pipeline::Pipeline(const std::string& name) : Pass(), name_(name) {}

void Pipeline::set_debug_logging(bool enable) { debug_logging_ = enable; }

std::string Pipeline::name() { return this->name_; };

size_t Pipeline::size() const { return this->passes_.size(); };

bool Pipeline::run(builder::SDFGBuilder& builder) {
    CompileStatistics::enter_pipeline_if_enabled(name_);

    bool applied = false;
    bool applied_pipeline;
    uint32_t pipe_iterations = 0;
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
        ++pipe_iterations;
    } while (applied_pipeline);

    CompileStatistics::add_metric_if_enabled("pipeline_iterations", pipe_iterations);
    CompileStatistics::exit_pipeline_if_enabled();
    return applied;
};

bool Pipeline::run(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    CompileStatistics::enter_pipeline_if_enabled(name_);

    static uint32_t runs = 0;
    auto dir = builder.subject().metadata_if_exists("output_dir");
    std::optional<std::filesystem::path> output_dir;
    if (dir) {
        std::filesystem::path p = *dir;
        output_dir = p / ("pipeline_" + this->name_ + "_" + std::to_string(runs));
    }

    bool applied = false;
    bool applied_pipeline;
    uint32_t pipe_iterations = 0;
    do {
        applied_pipeline = false;
        uint32_t pass_idx = 0;
        for (auto& pass : this->passes_) {
            bool applied_pass = false;
            uint32_t pass_iterations = 0;
            do {
                if (debug_logging_) {
                    if (output_dir.has_value())
                        visualizer::DotVisualizer::writeToFile(
                            builder.subject(),
                            output_dir.value() /
                                ("pipe_" + std::to_string(pass_iterations) + "_" + std::to_string(pass_idx) + "_" +
                                 pass->name() + "_" + std::to_string(pass_iterations) + ".sdfg.dot")
                        );
                }
                applied_pass = pass->run(builder, analysis_manager);
                applied_pipeline |= applied_pass;
                ++pass_iterations;
            } while (applied_pass);
            ++pass_idx;
        }
        applied |= applied_pipeline;
        ++pipe_iterations;
    } while (applied_pipeline);
    CompileStatistics::add_metric_if_enabled("pipeline_iterations", pipe_iterations);

    CompileStatistics::exit_pipeline_if_enabled();

    return applied;
};

Pipeline Pipeline::dataflow_simplification(bool block_fusion_ignor_libnodes) {
    Pipeline p("DataflowSimplification");

    if (block_fusion_ignor_libnodes) {
        p.register_pass<NoLibnodesBlockFusionPass>();
    } else {
        p.register_pass<BlockFusionPass>();
    }
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

    p.register_pass<ForClassificationPass>();
    p.register_pass<SymbolPropagation>();
    p.register_pass<DeadDataElimination>();

    return p;
};

Pipeline Pipeline::memory() {
    Pipeline p("Memory");

    p.register_pass<AllocationManagementPass>();

    return p;
};

Pipeline Pipeline::constant_elimination() {
    Pipeline p("ConstantElimination");

    p.register_pass<ConstantPropagation>();
    p.register_pass<DeadDataElimination>();

    return p;
};

} // namespace passes
} // namespace sdfg
