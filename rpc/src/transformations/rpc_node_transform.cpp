#include "sdfg/transformations/rpc_node_transform.h"

#include <curl/curl.h>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>
#include <string>
#include <variant>

#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/codegen/utils.h"
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
    bool enable_fusion,
    bool normalize,
    bool dump_steps
)
    : node_(node), target_(target), category_(category), rpc_context_(rpc_context), dump_steps_(dump_steps),
      enable_fusion_(enable_fusion), normalize_(normalize) {}

std::string RPCNodeTransform::name() const { return "RPCNodeTransform"; }

std::string RPCNodeTransform::get_node_id_str() const { return std::to_string(this->node_.element_id()); }

bool RPCNodeTransform::
    can_be_applied(sdfg::builder::StructuredSDFGBuilder& builder, sdfg::analysis::AnalysisManager& analysis_manager) {
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();

    bool applicable = false;
    auto outermost_loops = loop_analysis.outermost_loops();

    for (auto outermost : outermost_loops) {
        auto loop_info = loop_analysis.loop_info(outermost);

        if (!loop_info.has_side_effects) {
            applicable = true;
            break;
        }
    }

    if (!applicable) {
        if (report_) {
            report_->transform_impossible(this->name(), "No applicable loop (" + get_node_id_str() + ")");
        }
        DEBUG_PRINTLN(
            "[RPC] Skipping node " << get_node_id_str()
                                   << ": no applicable loop (all outermost loops have side effects), no request sent"
        );
        return false;
    }

    DEBUG_PRINTLN(
        "[RPC] can_be_applied for node " << get_node_id_str() << ": querying " << rpc_context_.get_remote_address()
    );

    // Open a session once per SDFG.
    // if (!this->session_id_.has_value()) {
    //     this->session_id_ = rpc_context_.start_session();
    // }
    if (this->session_id_.has_value()) {
        builder.subject().add_metadata("transfer_tuning_session_id", this->session_id_.value());
    }

    auto opt_resp = query_rpc_server(
        {.sdfg = builder.subject(),
         .category = this->category_,
         .target = this->target_,
         .enable_fusion = this->enable_fusion_,
	 .normalize = this->normalize_,
         .session_id = this->session_id_},
        rpc_context_
    );

    // In case query was successful, store response
    if (std::holds_alternative<std::unique_ptr<passes::rpc::RpcOptResponse>>(opt_resp)) {
        this->opt_resp_ = std::move(std::get<std::unique_ptr<passes::rpc::RpcOptResponse>>(opt_resp));
    }

    bool can_apply = this->opt_resp_ != nullptr &&
                     (this->opt_resp_->sdfg_result.has_value() || this->opt_resp_->local_replay.has_value());

    if (!can_apply) {
        DEBUG_PRINTLN(
            "[RPC] Skipping node " << get_node_id_str() << ": server returned no applicable optimization from "
                                   << rpc_context_.get_remote_address()
        );
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
    nlohmann::json payload = {
        {"sdfg", sdfg_json},
        {"category", request.category},
        {"target", request.target},
        {"enable_fusion", request.enable_fusion},
        {"normalize", request.normalize}
    };
    if (request.session_id.has_value()) {
        payload["session_id"] = request.session_id.value();
    }
    std::string payload_str = payload.dump();

    // Log where the request is going and what it carries. Header values (which may contain auth
    // tokens) are intentionally omitted; only header keys are printed.
    const std::string remote_address = context.get_remote_address();
    DEBUG_PRINTLN(
        "[RPC] Sending optimization request to " << remote_address << " (target=" << request.target << ", category="
                                                 << request.category << ", payload=" << payload_str.size() << " bytes)"
    );
    for (const auto& [key, value] : context_headers) {
        DEBUG_PRINTLN("[RPC]   header: " << key);
    }

    // Send query
    HttpResult res = post_json(curl_handle, remote_address, payload_str, headers);

    DEBUG_PRINTLN(
        "[RPC] Received response from " << remote_address << " (http_status=" << res.http_status
                                        << ", curl_code=" << res.curl_code << ", body=" << res.body.size() << " bytes)"
    );

    auto rpc_response = parse_rpc_response(res);

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl_handle);

    return std::move(rpc_response);
}

std::variant<std::unique_ptr<passes::rpc::RpcOptResponse>, std::string> RPCNodeTransform::parse_rpc_response(HttpResult
                                                                                                                 result
) {
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

        auto json_session_id = parsed.find("session_id");
        if (json_session_id != parsed.end() && json_session_id->is_string()) {
            rpc_response->session_id = json_session_id->get<std::string>();
        }

        auto parse_metadata = [](const nlohmann::json& json_metadata) {
            passes::rpc::RpcOptimizationMetadata meta;
            auto json_region_id = json_metadata.find("region_id");
            if (json_region_id != json_metadata.end() && !json_region_id->is_null()) {
                meta.region_id = json_region_id->get<std::string>();
            }
            auto json_speedup = json_metadata.find("speedup");
            if (json_speedup != json_metadata.end() && !json_speedup->is_null()) {
                meta.speedup = json_speedup->get<double>();
            }
            auto json_vector_distance = json_metadata.find("vector_distance");
            if (json_vector_distance != json_metadata.end() && !json_vector_distance->is_null()) {
                meta.vector_distance = json_vector_distance->get<double>();
            }
            return meta;
        };

        auto parse_local_replay = [](const nlohmann::json& json_replay) {
            passes::rpc::RpcLocalReplayRecipe recipe;
            recipe.sequence = json_replay.at("sequence");
            return recipe;
        };

        // The multi-cutout endpoint returns a "results" array, each entry carrying its own
        // element_id, local replay recipe, and metadata.
        auto json_results = parsed.find("results");
        if (json_results != parsed.end() && json_results->is_array()) {
            for (const auto& json_result : *json_results) {
                passes::rpc::RpcRegionResult region_result;

                auto json_element_id = json_result.find("element_id");
                if (json_element_id != json_result.end() && !json_element_id->is_null()) {
                    region_result.element_id = json_element_id->get<int64_t>();
                }

                auto json_result_replay = json_result.find("local_replay");
                if (json_result_replay != json_result.end() && !json_result_replay->is_null()) {
                    region_result.local_replay = parse_local_replay(*json_result_replay);
                }

                auto json_metadata = json_result.find("metadata");
                if (json_metadata != json_result.end()) {
                    region_result.metadata = parse_metadata(*json_metadata);
                }

                rpc_response->results.push_back(std::move(region_result));
            }
        } else {
            // Single-region responses expose metadata (and replay) at the top level.
            auto json_metadata = parsed.find("metadata");
            if (json_metadata != parsed.end()) {
                passes::rpc::RpcRegionResult region_result;
                region_result.metadata = parse_metadata(*json_metadata);
                region_result.local_replay = rpc_response->local_replay;
                rpc_response->results.push_back(std::move(region_result));
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

    // Record the transfer-tuning session on the SDFG. Prefer the client-owned id from start_session
    // (authoritative, independent of whether the server echoes it), falling back to the response.
    const auto& session_id = this->session_id_.has_value() ? this->session_id_ : opt.session_id;
    if (session_id.has_value()) {
        builder.subject().add_metadata("transfer_tuning_session_id", session_id.value());
    }

    if (opt.sdfg_result.has_value()) {
        auto& sdfg_response = opt.sdfg_result->sdfg;

        // this consumes the SDFG result
        if (this->node_.get_parent() == nullptr) {
            // Whole-SDFG case: node_ is the root sequence. Replace its body in place with the
            // optimized SDFG's body.
            auto& root = static_cast<structured_control_flow::Sequence&>(this->node_);
            builder.remove_children(root);
            builder.move_children(sdfg_response->root(), root);
        } else {
            // Nested-loop case: splice the optimized children into the parent in place of the loop.

            auto parent_scope = static_cast<structured_control_flow::Sequence*>(this->node_.get_parent());
            size_t index = parent_scope->index(this->node_);

            auto num_children = sdfg_response->root().size();
            builder.move_children(sdfg_response->root(), *parent_scope, index); // move all optimized children into
                                                                                // place
            builder.remove_child(*parent_scope, index + num_children); // remove old loop
        }

        if (opt.sdfg_result->sdfg->element_counter() > builder.subject().element_counter()) {
            builder.set_element_counter(opt.sdfg_result->sdfg->element_counter());
        }

        for (auto& container : sdfg_response->containers()) {
            if (builder.subject().exists(container)) {
                continue;
            }
            auto& type = sdfg_response->type(container);
            if (type.type_id() == sdfg::types::TypeID::Reference) {
                auto& reference_type = dynamic_cast<const sdfg::codegen::Reference&>(type).reference_type();
                builder.add_container(container, reference_type, false, false);
            } else {
                builder.add_container(container, type, false, false);
            }
        }

        opt.sdfg_result->sdfg.reset();
    } else if (opt.local_replay.has_value()) {
        try {
            Replayer replayer;
            replayer.replay(builder, analysis_manager, opt.local_replay.value().sequence, false);
        } catch (const std::exception& e) {
            std::cerr << "[ERROR] Failed to replay rpc optimization: " << e.what() << std::endl;
            return;
        }
    }

    if (opt.local_replay.has_value()) {
        auto recipe = opt.local_replay.value();
        for (const auto& region_result : opt.results) {
            DEBUG_PRINTLN(
                "[RPC] Applied RPC optimization sequence with speedup "
                << region_result.metadata.speedup << " and vector distance " << region_result.metadata.vector_distance
                << " to loopnest " << element_id
            );
        }
    } else {
        for (const auto& region_result : opt.results) {
            DEBUG_PRINTLN(
                "[RPC] Applied plain SDFG with speedup " << region_result.metadata.speedup << " and vector distance "
                                                         << region_result.metadata.vector_distance
            );
        }
    }
}

void RPCNodeTransform::to_json(nlohmann::json& j) const {
    j["transformation_type"] = name();
    nlohmann::json params = {{"target", target_}, {"category", category_}};
    nlohmann::json results_array = nlohmann::json::array();
    for (const auto& region_result : opt_resp_->results) {
        nlohmann::json entry;
        if (region_result.element_id.has_value()) {
            entry["element_id"] = region_result.element_id.value();
        }
        nlohmann::json metadata = {
            {"speedup", region_result.metadata.speedup}, {"vector_distance", region_result.metadata.vector_distance}
        };
        if (region_result.metadata.region_id.has_value()) {
            metadata["region_id"] = region_result.metadata.region_id.value();
        }
        entry["metadata"] = metadata;
        if (region_result.local_replay.has_value()) {
            entry["local_replay"] = {{"sequence", region_result.local_replay->sequence}};
        }
        results_array.push_back(entry);
    }
    params["results"] = results_array;
    j["parameters"] = params;
}

} // namespace transformations
} // namespace sdfg
