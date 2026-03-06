#include "sdfg/transformations/rpc_node_transform.h"

#include <curl/curl.h>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>
#include <string>
#include <variant>

#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/cutouts/cutouts.h"
#include "sdfg/optimization_report/pass_report_consumer.h"
#include "sdfg/passes/rpc/rpc_context.h"
#include "sdfg/passes/rpc/rpc_responses.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/transformations/replayer.h"
#include "sdfg/transformations/rpc_node_transform.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {


RPCNodeTransform::RPCNodeTransform(
    structured_control_flow::ControlFlowNode& node,
    const std::string& target,
    const std::string& category,
    sdfg::passes::rpc::RpcContext& rpc_context,
    bool dump_steps
)
    : node_(node), target_(target), category_(category), rpc_context_(rpc_context), dump_steps_(dump_steps) {}

std::string RPCNodeTransform::name() const { return "RPCNodeTransform"; }

std::string RPCNodeTransform::get_node_id_str() const { return std::to_string(this->node_.element_id()); }

bool RPCNodeTransform::
    can_be_applied(sdfg::builder::StructuredSDFGBuilder& builder, sdfg::analysis::AnalysisManager& analysis_manager) {

    // Get loop info

    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();

    auto loop_info = loop_analysis.loop_info(&this->node_);

    // Re-check for side effects with fresh loop info
    if (loop_info.has_side_effects) {
        if (report_) {
            report_->transform_impossible(
                this->name(), "Loopnest side effects (" + get_node_id_str() + ")"
            );
        }
        return false;
    }

    // Create cutout SDFG
    std::unique_ptr<sdfg::StructuredSDFG> loop_sdfg = util::cutout(builder.subject(), analysis_manager, this->node_);

    // Loop info is only used for information on the loop structure
    auto opt_resp = query_rpc_server(
        {.sdfg = *loop_sdfg,
         .category = this->category_,
         .target = this->target_,
         .loop_info = loop_info},
        rpc_context_
    );

    // In case query was successful, store response
    if (std::holds_alternative<std::unique_ptr<passes::rpc::RpcOptResponse>>(opt_resp)) {
        this->opt_resp_ = std::move(std::get<std::unique_ptr<passes::rpc::RpcOptResponse>>(opt_resp));
    }

    bool can_apply = this->opt_resp_ != nullptr && (this->opt_resp_->sdfg_result.has_value() ||
                                                    this->opt_resp_->local_replay.has_value());

    if (report_) {
        if (!can_apply) {
            std::string error_msg;
            if (std::holds_alternative<std::string>(opt_resp)) {
                error_msg = std::get<std::string>(opt_resp);
            } else if (this->opt_resp_->error.has_value()) {
                error_msg = this->opt_resp_->error.value();
            }
            report_->transform_impossible(
                this->name(), "No opt. SDFG received (" + get_node_id_str() + ", " + error_msg + ")"
            );
        }
    }
    return can_apply;
}

std::variant<std::unique_ptr<passes::rpc::RpcOptResponse>, std::string> RPCNodeTransform::
    query_rpc_server(passes::rpc::RpcOptRequest request, sdfg::passes::rpc::RpcContext& context) {

    CURL* curl_handle = curl_easy_init();
    if (!curl_handle) {
        std::cerr << "[ERROR] Could not initialize CURL!" << std::endl;
        return {"CurlInit"};
    }

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    // Add all headers provided by the RPC context (auth and optional testing headers).
    auto context_headers = context.get_auth_headers();
    for (const auto& [key, value] : context_headers) {
        std::string hdr = key + ": " + value;
        headers = curl_slist_append(headers, hdr.c_str());
    }
    curl_easy_setopt(curl_handle, CURLOPT_HTTPHEADER, headers);

    sdfg::serializer::JSONSerializer serializer;
    nlohmann::json sdfg_json = serializer.serialize(request.sdfg);

    // Construct query payload
    nlohmann::json payload = {{"sdfg", sdfg_json},
                              {"category", request.category},
                              {"target", request.target},
                              {"loop_info", analysis::loop_info_to_json(request.loop_info)}};
    std::string payload_str = payload.dump();

    // Send query
    HttpResult res = post_json(curl_handle, context.get_remote_address(), payload_str, headers);

    auto rpc_response = parse_rpc_response(res);

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl_handle);

    return std::move(rpc_response);
}

std::variant<std::unique_ptr<passes::rpc::RpcOptResponse>, std::string> RPCNodeTransform::parse_rpc_response(HttpResult result) {

    // Check for HTTP errors first (including authentication issues)
    if (!result.error_message.empty()) {
        std::cerr << result.error_message << std::endl;
        return {result.error_message};
    }

    auto rpc_response = std::make_unique<passes::rpc::RpcOptResponse>();

    try {
            // Parse response
    nlohmann::json parsed;
        try {
            parsed = nlohmann::json::parse(result.body);
        } catch (const std::exception& e) {
            std::cerr << "[ERROR] RPC optimization response failed to parse: " << e.what() << std::endl;
            return {"InvalidJsonResp"};
        }

        auto json_error = parsed.find("error");
        if (json_error != parsed.end()) {
                DEBUG_PRINTLN("[ERROR] RPC optimization query returned error: " << json_error->get<std::string>());
                return {json_error->get<std::string>()};
        }

        auto json_sdfg_result = parsed.find("sdfg_result");
        if (json_sdfg_result != parsed.end()) {
            auto sdfg_field = json_sdfg_result->at("sdfg");
            passes::rpc::RpcSdfgResult result;
            sdfg::serializer::JSONSerializer serializer;
            result.sdfg = serializer.deserialize(sdfg_field);
            rpc_response->sdfg_result = std::move(result);
        }

        auto json_local_replay = parsed.find("local_replay");
        if (json_local_replay != parsed.end()) {
            passes::rpc::RpcLocalReplayRecipe recipe;
            recipe.sequence = json_local_replay->at("sequence");
            rpc_response->local_replay = std::move(recipe);
        }

        auto json_metadata = parsed.find("metadata");
        if (json_metadata != parsed.end()) {
            auto& meta = rpc_response->metadata;
            auto json_region_id = json_metadata->find("region_id");
            if (json_region_id != json_metadata->end()) {
                meta.region_id = json_region_id->get<std::string>();
            }
            auto json_speedup = json_metadata->find("speedup");
            if (json_speedup != json_metadata->end()) {
                meta.speedup = json_speedup->get<double>();
            }
            auto json_vector_distance = json_metadata->find("vector_distance");
            if (json_vector_distance != json_metadata->end()) {
                meta.vector_distance = json_vector_distance->get<double>();
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Failed to parse RPC optimization response: " << e.what() << std::endl;
        return {"RpcRespParseError"};
    }
    return std::move(rpc_response);
}


void RPCNodeTransform::
    apply(sdfg::builder::StructuredSDFGBuilder& builder, sdfg::analysis::AnalysisManager& analysis_manager) {
    auto& opt = *this->opt_resp_;

    if (!opt.sdfg_result.has_value() && !opt.local_replay.has_value()) {
        throw std::runtime_error("RPCNodeTransform: No SDFG result or replay to apply.");
    }

    int element_id = this->node_.element_id();

    if (opt.sdfg_result.has_value()) {
        auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
        auto parent_scope = static_cast<structured_control_flow::Sequence*>(scope_analysis.parent_scope(&this->node_));
        size_t index = parent_scope->index(this->node_);

        // this consumes the SDFG result

        auto& sdfg_response = opt.sdfg_result->sdfg;

        // TODO: add transitions from after loop to tmp_scope

        builder.remove_child(*parent_scope, index); // remove old loop
        builder.move_child(opt.sdfg_result->sdfg->root(), 0, *parent_scope, index); // move optimized loop into place

        for (auto& container : sdfg_response->containers()) {
            if (builder.subject().exists(container)) {
                continue;
            }
            auto& type = sdfg_response->type(container);
            builder.add_container(
                container,
                type,
                false,
                false
            );
        }

        opt.sdfg_result->sdfg.reset();
    } else if (opt.local_replay.has_value()) {
        try {
            Replayer replayer;
            replayer.replay(builder, analysis_manager, opt.local_replay.value().sequence, false);
        } catch (const std::exception& e) {
            std::cerr << "[ERROR] Failed to replay rpc optimization: " << e.what() << std::endl;
            if (report_) {
                report_
                    ->transform_impossible(this->name(), "Failed to replay local optimization: " + std::string(e.what()));
            }
            return;
        }
    }

    if (report_) {
        nlohmann::json j;
        this->to_json(j);
        report_->transform_applied(this->name(), j);
    }

    if (opt.local_replay.has_value()) {
        auto recipe = opt.local_replay.value();
        std::cout << "Applied RPC optimization sequence with speedup " << opt.metadata.speedup << " and vector distance "
        << opt.metadata.vector_distance << ":\n";

        if (dump_steps_) {
            print_transformation_sequence(recipe.sequence);
        }

    } else {
        std::cout << "RPC: Applied plain SDFG with speedup " << opt.metadata.speedup << " and vector distance "
        << opt.metadata.vector_distance << "\n";
    }
}


void RPCNodeTransform::to_json(nlohmann::json& j) const {
    j["transformation_type"] = name();
    nlohmann::json params = {{"target", target_}, {"category", category_}, {"speedup", opt_resp_->metadata.speedup}};
    if (opt_resp_->metadata.region_id.has_value()) {
        params["region_id"] = opt_resp_->metadata.region_id.value();
    }
    params["vector_distance"] = opt_resp_->metadata.vector_distance;
    j["parameters"] = params;
}

void RPCNodeTransform::print_transformation_sequence(const nlohmann::json& sequence) const {
    if (sequence.empty()) {
        std::cerr << "Nothing to tune, code already optimized" << std::endl;
    } else {
        for (auto& desc : sequence) {
            bool fail = false;
            auto transformation_type = desc.find("transformation_type");
            std::cout << "\t" << transformation_type->get<std::string>();
            auto transformation_parameter = desc.find("parameters");
            if (transformation_parameter != desc.end()) {
                std::cout << " (";
                for (auto& [key, value] : transformation_parameter->items()) {
                    std::cout << key << "=" << value << ", ";
                }
                std::cout << ")";
            }
            std::cout << "\n";
        }
    }
}


} // namespace transformations
} // namespace sdfg
