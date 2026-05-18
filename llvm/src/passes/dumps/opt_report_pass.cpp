#include "docc/passes/dumps/opt_report_pass.h"

#include <docc/target/tenstorrent/tenstorrent_transform.h>
#include <sdfg/analysis/analysis.h>
#include <sdfg/analysis/loop_analysis.h>
#include <sdfg/builder/structured_sdfg_builder.h>
#include <sdfg/optimization_report/optimization_report.h>
#include <sdfg/serializer/json_serializer.h>
#include <sdfg/targets/cuda/cuda_data_offloading_node.h>
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/targets/omp/schedule.h"
#include "sdfg/transformations/offloading/cuda_transform.h"
#include "sdfg/transformations/omp_transform.h"

namespace docc {
namespace passes {

llvm::PreservedAnalyses OPTReportPass::
    run(llvm::Module& Module, llvm::ModuleAnalysisManager& MAM, analysis::AnalysisManager& AM) {
    auto& registry = AM.get<analysis::SDFGRegistry>();
    if (!registry.has_module(Module)) {
        return llvm::PreservedAnalyses::all();
    }

    registry.for_each_sdfg_modifiable(Module, [&](sdfg::StructuredSDFG& sdfg) {
        sdfg::builder::StructuredSDFGBuilder builder(sdfg);
        sdfg::analysis::AnalysisManager manager(builder.subject());
        auto& loop_analysis = manager.get<sdfg::analysis::LoopAnalysis>();

        auto* reports_in_sdfg = report_->get_scope_reports(&sdfg);

        auto outermost_loops = loop_analysis.outermost_loops();


        sdfg::OptimizationReport opt_report(builder.subject());
        if (std::getenv("DOCC_OFFLOAD_REPORT")) {
            std::string env_var_value = std::string(std::getenv("DOCC_OFFLOAD_REPORT"));
            if (env_var_value == "1" || env_var_value == "true" || env_var_value == "yes") {
                for (size_t loopnest_index = 0; loopnest_index < outermost_loops.size(); loopnest_index++) {
                    ControlFlowNode* loop_nest = outermost_loops[loopnest_index];

                    RegionReport* reports_in_loopnest = nullptr;
                    if (reports_in_sdfg) {
                        auto it = reports_in_sdfg->find(loopnest_index);
                        if (it != reports_in_sdfg->end()) {
                            reports_in_loopnest = it->second.get();
                        }
                    }

                    if (reports_in_loopnest) {
                        for (auto& [ttype, report] : reports_in_loopnest->transform_results) {
                            opt_report.add_transformation_entry(loopnest_index, ttype, 0L, report);
                        }
                        for (auto& [target, outcome] : reports_in_loopnest->targets_possible) {
                            opt_report.add_target_test(loopnest_index, target, outcome);
                        }
                    }

                    if (auto map = dynamic_cast<sdfg::structured_control_flow::Map*>(loop_nest)) {
                        sdfg::transformations::OMPTransform omp_transform(*map);
                        opt_report.add_target_test(
                            loopnest_index,
                            sdfg::omp::ScheduleType_OMP::create().value(),
                            omp_transform.can_be_applied(builder, manager)
                        );
                        sdfg::cuda::CUDATransform cuda_transform(*map);
                        opt_report.add_target_test(
                            loopnest_index,
                            sdfg::cuda::ScheduleType_CUDA::create().value(),
                            cuda_transform.can_be_applied(builder, manager)
                        );
                        sdfg::tenstorrent::TenstorrentTransform tt_transform(builder, manager, *map);
                        opt_report.add_target_test(
                            loopnest_index,
                            sdfg::tenstorrent::ScheduleType_Tenstorrent_Kernel::create().value(),
                            tt_transform.can_be_applied(builder, manager)
                        );
                    } else {
                        opt_report.add_target_test(loopnest_index, sdfg::omp::ScheduleType_OMP::create().value(), false);
                        opt_report
                            .add_target_test(loopnest_index, sdfg::cuda::ScheduleType_CUDA::create().value(), false);
                        opt_report.add_target_test(
                            loopnest_index, sdfg::tenstorrent::ScheduleType_Tenstorrent_Kernel::create().value(), false
                        );
                    }
                }
            }
        }

        // Add optimization report path metadata
        std::filesystem::path sdfg_path = builder.subject().metadata("sdfg_file");
        std::filesystem::path opt_report_file = sdfg_path.parent_path() /
                                                (sdfg_path.stem().string() + ".opt_report.json");
        builder.subject().add_metadata("opt_report_file", opt_report_file.string());

        auto report = opt_report.get_report();
        std::ofstream ofs(opt_report_file);
        if (!ofs.is_open()) {
            throw std::runtime_error("Failed to open file: " + opt_report_file.string());
        }
        ofs << report.dump(2);
        ofs.close();
    });

    return llvm::PreservedAnalyses::all();
}

} // namespace passes
} // namespace docc
