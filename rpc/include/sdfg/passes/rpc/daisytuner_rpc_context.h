#pragma once

#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "sdfg/passes/rpc/rpc_context.h"

namespace sdfg::passes::rpc {

class DaisytunerRpcContext : public SimpleRpcContext {
public:
    inline static constexpr auto DEFAULT_ENDPOINT = "transfertune";
    inline static constexpr auto DEFAULT_AUTH_HEADER = "Authorization";
    inline static constexpr auto DEFAULT_SERVER = "https://docc-backend-1080482399950.europe-west1.run.app/docc";

    DaisytunerRpcContext(std::string license_token, bool is_job_token = false);

    static std::shared_ptr<DaisytunerRpcContext> from_docc_config();

    static std::string build_auth_header_content(std::pair<std::string, bool> docc_auth);

    static std::optional<std::pair<std::string, bool>> find_docc_auth();
};

inline std::shared_ptr<DaisytunerRpcContext> build_rpc_context_from_docc_config() {
    return DaisytunerRpcContext::from_docc_config();
}

} // namespace sdfg::passes::rpc
