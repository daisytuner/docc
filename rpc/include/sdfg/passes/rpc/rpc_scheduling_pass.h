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
    bool normalize_ = true;
    bool schedule_loops_ = true;

public:
    /**
     * @param normalize whether the server brings the SDFG into normal form before tuning it; false sends it as it is
     */
    RpcOptimizationPass(
        std::shared_ptr<rpc::RpcContext> rpc_context,
        docc::target::TargetOptions options,
        bool normalize = true,
        bool schedule_loops = true,
        sdfg::PassReportConsumer* report = nullptr
    )
        : rpc_context_(rpc_context), options_(std::move(options)), report_(report), normalize_(normalize),
          schedule_loops_(schedule_loops) {
    }
    ~RpcOptimizationPass() override = default;

    bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    std::string name() override {
        return "RpcOptimizationPass";
    }
};


} // namespace scheduler
} // namespace passes
} // namespace sdfg
