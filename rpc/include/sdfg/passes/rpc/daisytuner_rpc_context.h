#pragma once

#include "rpc_context.h"

namespace sdfg::passes::rpc {

struct DaisytunerRpcContextBuilder : public sdfg::passes::rpc::SimpleRpcContextBuilder {

    SimpleRpcContextBuilder& initialize_local_default();
    SimpleRpcContextBuilder& from_file(std::filesystem::path config_file);
    SimpleRpcContextBuilder& from_header_env(std::string env_var = "RPC_HEADER");
    SimpleRpcContextBuilder& from_env(std::string env_var = "SDFG_RPC_CONFIG");

    SimpleRpcContextBuilder& add_header(std::string name, std::string value);

    SimpleRpcContextBuilder& from_docc_config();
};

class DaisytunerTransfertuningRpcContext : public SimpleRpcContext {
public:
    inline static constexpr auto DEFAULT_ENDPOINT = "transfertune";
    inline static constexpr auto DEFAULT_AUTH_HEADER = "Authorization";
    inline static constexpr auto DEFAULT_SERVER = "https://docc-backend-1080482399950.europe-west1.run.app/docc";

    DaisytunerTransfertuningRpcContext(std::string license_token, bool job_specific_token = false);

    static std::unique_ptr<DaisytunerTransfertuningRpcContext> from_docc_config();

    static std::string build_auth_header_content(std::pair<std::string, bool> docc_auth);

    static std::optional<std::pair<std::string, bool>> find_docc_auth();
};

} // namespace sdfg::passes::rpc
