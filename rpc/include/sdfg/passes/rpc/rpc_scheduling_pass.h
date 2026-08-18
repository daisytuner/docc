#pragma once

#include <sdfg/passes/pass.h>
#include <sdfg/plugins/targets.h>
#include <string>
#include "sdfg/optimization_report/pass_report_consumer.h"
#include "sdfg/passes/rpc/rpc_context.h"

namespace sdfg {
namespace transformations {
class Recorder;
}
namespace passes {
namespace scheduler {

class RpcOptimizationPass : public Pass {
private:
    std::shared_ptr<rpc::RpcContext> rpc_context_;
    docc::target::TargetOptions options_;
    sdfg::PassReportConsumer* report_ = nullptr;
    bool enable_fusion_ = true;
    bool schedule_loops_ = true;

public:
    RpcOptimizationPass(
        std::shared_ptr<rpc::RpcContext> rpc_context,
        docc::target::TargetOptions options,
        bool enable_fusion = true,
        bool schedule_loops = true,
        sdfg::PassReportConsumer* report = nullptr
    )
        : rpc_context_(rpc_context), options_(std::move(options)), report_(report), enable_fusion_(enable_fusion),
          schedule_loops_(schedule_loops) {}
    ~RpcOptimizationPass() override = default;

    bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    std::string name() override { return "RpcOptimizationPass"; }
};


} // namespace scheduler
} // namespace passes
} // namespace sdfg
