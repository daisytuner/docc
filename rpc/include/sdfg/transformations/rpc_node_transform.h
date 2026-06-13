#pragma once
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>
#include <string>
#include <variant>
#include "sdfg/analysis/users.h"
#include "sdfg/passes/rpc/rpc_responses.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_sdfg.h"

#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/passes/rpc/rpc_context.h"
#include "sdfg/transformations/replayer.h"
#include "sdfg/transformations/transformation.h"
#include "sdfg/util/utils_curl.h"

namespace sdfg {
namespace transformations {

class RPCNodeTransform : public sdfg::transformations::Transformation {
private:
    structured_control_flow::ControlFlowNode& node_;
    std::string target_;
    std::string category_;

    sdfg::passes::rpc::RpcContext& rpc_context_;

    std::unique_ptr<passes::rpc::RpcOptResponse> opt_resp_;

    bool dump_steps_;

    std::string get_node_id_str() const;

    std::variant<std::unique_ptr<passes::rpc::RpcOptResponse>, std::string>
    query_rpc_server(passes::rpc::RpcOptRequest request, sdfg::passes::rpc::RpcContext& context);

    void print_transformation_sequence(const nlohmann::json& sequence) const;

public:
    RPCNodeTransform(
        structured_control_flow::ControlFlowNode& node,
        const std::string& target,
        const std::string& category,
        sdfg::passes::rpc::RpcContext& rpc_context,
        bool print_steps = false
    );

    virtual std::string name() const override;

    virtual bool can_be_applied(
        sdfg::builder::StructuredSDFGBuilder& builder, sdfg::analysis::AnalysisManager& analysis_manager
    ) override;

    virtual void apply(sdfg::builder::StructuredSDFGBuilder& builder, sdfg::analysis::AnalysisManager& analysis_manager)
        override;

    passes::rpc::RpcOptResponse& applied_recipe() const { return *opt_resp_; }

    void to_json(nlohmann::json& j) const override;

    std::variant<std::unique_ptr<passes::rpc::RpcOptResponse>, std::string> parse_rpc_response(HttpResult result);
};


} // namespace transformations
} // namespace sdfg
