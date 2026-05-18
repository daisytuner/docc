#include <gtest/gtest.h>

#include "docc/passes/dumps/global_cfg_dump_pass.h"


namespace docc::passes {

TEST(GlobalCFGDumpPassTest, EscapeDot) {
    std::string input = R"(func<name>{test}"example")";
    std::string expected = R"(func\<name\>\{test\}\"example\")";
    std::string output = GlobalCFGPrinterPass::escapeDot(input);
    EXPECT_EQ(expected, output);
}

} // namespace docc::passes
