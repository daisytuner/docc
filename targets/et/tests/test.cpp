#include <gtest/gtest.h>

#include "sdfg/codegen/dispatchers/node_dispatcher_registry.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/targets/cuda/plugin.h"
#include "sdfg/targets/omp/plugin.h"

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    sdfg::codegen::register_default_dispatchers();
    sdfg::serializer::register_default_serializers();
    return RUN_ALL_TESTS();
}
