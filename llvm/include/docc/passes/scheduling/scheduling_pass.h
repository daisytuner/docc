#pragma once

#include <llvm/IR/PassManager.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>

#include "docc/analysis/analysis.h"
#include "sdfg/optimization_report/pass_report_consumer.h"
#include "sdfg/plugins/plugins.h"

namespace docc {
namespace passes {

/**
 * @brief Schedules every SDFG for the --docc-tune target (see sdfg::scheduling::schedule), as the Python frontend does.
 *
 * Records each SDFG in the target's normal form in its "sdfg_file", for instrumentation and tuning.
 */
class SchedulingPass : public llvm::PassInfoMixin<SchedulingPass> {
private:
    const sdfg::plugins::Context& context_;
    const bool dump_visualization_;
    const bool force_synchronous_;
    bool transfer_opt_;
    sdfg::PassReportConsumer* const report_ = nullptr;

public:
    SchedulingPass(
        const sdfg::plugins::Context& context,
        bool force_synchronous = false,
        bool dump_visualization = false,
        bool transfer_opt = true,
        sdfg::PassReportConsumer* report = nullptr
    );

    static bool available(analysis::AnalysisManager& AM) {
        return true;
    }

    llvm::PreservedAnalyses run(llvm::Module& Module, llvm::ModuleAnalysisManager& MAM, analysis::AnalysisManager& AM);
};

} // namespace passes
} // namespace docc
