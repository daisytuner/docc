#include "sdfg/passes/rpc/daisytuner_rpc_context.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace sdfg::passes::rpc {

DaisytunerRpcContext::DaisytunerRpcContext(std::string license_token, bool is_job_token)
    : SimpleRpcContext(
          DEFAULT_SERVER,
          DEFAULT_ENDPOINT,
          {{DEFAULT_AUTH_HEADER, build_auth_header_content({license_token, is_job_token})}}
      ) {
    // Check for RPC_SERVER environment variable to override default server
    const char* rpc_server = std::getenv("RPC_SERVER");
    if (rpc_server && *rpc_server) {
        set_server(rpc_server);
    }
}

std::string DaisytunerRpcContext::build_auth_header_content(std::pair<std::string, bool> docc_auth) {
    return static_cast<std::string>(docc_auth.second ? "Job" : "Token") + " " + docc_auth.first;
}

std::optional<std::pair<std::string, bool>> DaisytunerRpcContext::find_docc_auth() {
    // Check $DOCC_JOB_TOKEN
    const char* job_token = std::getenv("DOCC_JOB_TOKEN");
    if (job_token && *job_token) {
        return std::make_pair(std::string(job_token), true);
    }

    // Check $DOCC_ACCESS_TOKEN
    const char* env_token = std::getenv("DOCC_ACCESS_TOKEN");
    if (env_token && *env_token) {
        return std::make_pair(std::string(env_token), false);
    }

    // Check $HOME/.config/docc/token
    const char* home = std::getenv("HOME");
    if (home && *home) {
        std::filesystem::path config_dir = std::filesystem::path(home) / ".config" / "docc";
        std::ifstream infile((config_dir / "token").string());
        if (infile) {
            std::ostringstream ss;
            ss << infile.rdbuf();
            std::string token = ss.str();
            token.erase(token.find_last_not_of(" \n\r\t") + 1);
            return std::make_pair(token, false);
        }
    }

    // Check /var/lib/daisytuner/session
    std::ifstream session_file("/var/lib/daisytuner/session");
    if (session_file) {
        std::ostringstream ss;
        ss << session_file.rdbuf();
        std::string token = ss.str();
        token.erase(token.find_last_not_of(" \n\r\t") + 1);
        return std::make_pair(token, false);
    }

    return std::nullopt;
}

std::shared_ptr<DaisytunerRpcContext> DaisytunerRpcContext::from_docc_config() {
    auto auth = find_docc_auth();
    if (!auth.has_value()) {
        throw std::runtime_error(
            "DOCC access token not found. Please set DOCC_JOB_TOKEN or DOCC_ACCESS_TOKEN, "
            "or place a token in $HOME/.config/docc/token"
        );
    }

    auto context = std::make_shared<DaisytunerRpcContext>(auth->first, auth->second);

    // Check for RPC_SERVER environment variable to override default server
    const char* rpc_server = std::getenv("RPC_SERVER");
    if (rpc_server && *rpc_server) {
        context->set_server(rpc_server);
    }

    return context;
}

} // namespace sdfg::passes::rpc
