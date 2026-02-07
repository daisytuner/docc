#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

namespace sdfg::passes::rpc {

class RpcContext {
public:
    virtual std::string get_remote_address() const = 0;
    virtual std::unordered_map<std::string, std::string> get_auth_headers() const = 0;

    virtual ~RpcContext() = default;
};

class SimpleRpcContext : public RpcContext {
private:
    std::string server_;
    std::string endpoint_;
    std::unordered_map<std::string, std::string> headers_;

public:
    SimpleRpcContext(std::string server, std::string endpoint, std::unordered_map<std::string, std::string> headers = {})
        : server_(std::move(server)), endpoint_(std::move(endpoint)), headers_(std::move(headers)) {}

    std::string get_remote_address() const override { return server_ + "/" + endpoint_; }

    std::unordered_map<std::string, std::string> get_auth_headers() const override { return headers_; }
};

struct SimpleRpcContextBuilder {
    std::string server;
    std::string endpoint;
    std::unordered_map<std::string, std::string> headers;

    SimpleRpcContextBuilder() {}

    SimpleRpcContextBuilder& initialize_local_default();
    SimpleRpcContextBuilder& from_file(std::filesystem::path config_file);
    SimpleRpcContextBuilder& from_header_env(std::string env_var = "RPC_HEADER");
    SimpleRpcContextBuilder& from_env(std::string env_var = "SDFG_RPC_CONFIG");

    SimpleRpcContextBuilder& add_header(std::string name, std::string value);

    SimpleRpcContextBuilder& from_docc_config();

    std::shared_ptr<SimpleRpcContext> build(bool print = true) const;
};


inline std::shared_ptr<RpcContext> build_rpc_context_local() {
    SimpleRpcContextBuilder b;
    return b.initialize_local_default().build();
}

inline std::shared_ptr<RpcContext> build_rpc_context_from_file(std::filesystem::path config_file) {
    SimpleRpcContextBuilder b;
    return b.from_file(std::move(config_file)).build();
}

inline std::shared_ptr<RpcContext> build_rpc_context_auto() {
    SimpleRpcContextBuilder b;
    return b.initialize_local_default().from_env().from_header_env().build();
}

} // namespace sdfg::passes::rpc
