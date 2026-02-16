#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <ranges>
#include "sdfg/passes/rpc/daisytuner_rpc_context.h"

namespace sdfg::passes::rpc {

std::shared_ptr<SimpleRpcContext> SimpleRpcContextBuilder::build(bool print) const {
    if (server.empty()) {
        throw std::runtime_error("No server configured");
    }
    if (print) {
        std::cerr << "[INFO] Using RPC target " << server << "/" << endpoint << ", headers: [";
        for (const auto& key : headers | std::views::keys) {
            std::cerr << key << ", ";
        }
        std::cerr << "]" << std::endl;
    }
    return std::make_shared<SimpleRpcContext>(server, endpoint, headers);
}

SimpleRpcContextBuilder& SimpleRpcContextBuilder::initialize_local_default() {
    this->server = "http://localhost:8080/docc";
    this->endpoint = "transfertune";

    return *this;
}

SimpleRpcContextBuilder& SimpleRpcContextBuilder::from_file(std::filesystem::path config_file) {
    std::ifstream in(config_file);

    if (!in) {
        throw std::runtime_error("Config file not readable: " + config_file.string());
    }

    nlohmann::json j;
    in >> j;

    auto serverJ = j.find("SERVER");
    if (serverJ != j.end() && serverJ->is_string()) {
        server = serverJ->get<std::string>();
    }
    auto endpointJ = j.find("ENDPOINT");
    if (endpointJ != j.end() && endpointJ->is_string()) {
        endpoint = endpointJ->get<std::string>();
    }

    auto headersJ = j.find("HEADERS");
    if (headersJ != j.end() && headersJ->is_object()) {
        for (auto& [key, value] : headersJ->items()) {
            if (value.is_string()) {
                headers[key] = value.get<std::string>();
            }
        }
    }

    return *this;
}

SimpleRpcContextBuilder& SimpleRpcContextBuilder::from_env(std::string env_var) {
    auto envVar = std::getenv(env_var.c_str());
    if (envVar && *envVar) {
        auto cfg_path = std::filesystem::path(envVar);
        from_file(envVar);
    }
    return *this;
}

SimpleRpcContextBuilder& SimpleRpcContextBuilder::from_header_env(std::string env_var) {
    auto headerOverrideVar = std::getenv(env_var.c_str());
    if (headerOverrideVar && *headerOverrideVar) {
        std::string headerOverride = headerOverrideVar;
        auto idx = headerOverride.find_first_of(':');
        if (idx != std::string::npos) {
            std::string key = headerOverride.substr(0, idx);
            std::string value = headerOverride.substr(idx + 1);
            headers[key] = value;
        } else {
            headers["RPC-Hint"] = headerOverride;
        }
    }
    return *this;
}

SimpleRpcContextBuilder& SimpleRpcContextBuilder::from_docc_config() {
    auto auth = DaisytunerRpcContext::find_docc_auth();
    if (auth) {
        const char* rpc_server = std::getenv("RPC_SERVER");
        server = (rpc_server && *rpc_server) ? rpc_server : DaisytunerRpcContext::DEFAULT_SERVER;
        endpoint = DaisytunerRpcContext::DEFAULT_ENDPOINT;
        add_header(
            std::string(DaisytunerRpcContext::DEFAULT_AUTH_HEADER),
            DaisytunerRpcContext::build_auth_header_content(auth.value())
        );
    }
    return *this;
}

SimpleRpcContextBuilder& SimpleRpcContextBuilder::add_header(std::string name, std::string value) {
    headers[name] = value;
    return *this;
}

} // namespace sdfg::passes::rpc
