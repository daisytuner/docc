#include <gtest/gtest.h>

#include "sdfg/codegen/dispatchers/node_dispatcher_registry.h"
#include "sdfg/passes/rpc/rpc_context.h"
#include "sdfg/passes/rpc/rpc_scheduler.h"
#include "sdfg/serializer/json_serializer.h"

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    sdfg::codegen::register_default_dispatchers();
    sdfg::serializer::register_default_serializers();

    sdfg::passes::rpc::SimpleRpcContextBuilder ctxBuilder;
    auto ctx_ = ctxBuilder
                    .initialize_local_default() // localhost:8080/docc
                    .from_env() // $SDFG_RPC_CONFIG can override
                    .from_header_env() // $RPC_HEADER can override/add headers
                    .build();

    sdfg::passes::rpc::register_rpc_loop_opt(ctx_, "sequential", "server", true);

    return RUN_ALL_TESTS();
}
