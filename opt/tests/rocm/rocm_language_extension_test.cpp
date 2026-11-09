#include "sdfg/targets/rocm/codegen/rocm_language_extension.h"

#include <gtest/gtest.h>

#include "sdfg/builder/sdfg_builder.h"

namespace sdfg::rocm {

TEST(ROCMLanguageExtensionTest, ComputeType_BFloatInFloat) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    ROCMLanguageExtension generator(builder.subject());

    EXPECT_EQ(generator.compute_type(types::PrimitiveType::BFloat), types::PrimitiveType::Float);
    EXPECT_EQ(generator.compute_type(types::PrimitiveType::Half), types::PrimitiveType::Half);
    EXPECT_EQ(generator.compute_type(types::PrimitiveType::Float), types::PrimitiveType::Float);
}

} // namespace sdfg::rocm
