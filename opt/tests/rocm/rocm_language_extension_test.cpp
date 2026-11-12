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

TEST(ROCMLanguageExtensionTest, StreamingCopy_NativeWidthsOnly) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    ROCMLanguageExtension generator(builder.subject());

    for (size_t bytes : {4, 8, 16}) {
        EXPECT_FALSE(generator.streaming_copy("dst", "src", bytes).empty()) << bytes;
    }
    for (size_t bytes : {1, 2, 12, 32}) {
        EXPECT_TRUE(generator.streaming_copy("dst", "src", bytes).empty()) << bytes;
    }
}

} // namespace sdfg::rocm
