#include <gtest/gtest.h>

#include "docc/docc_paths.h"

using namespace docc;

TEST(DoccPathsTest, ResolveCmakeRoot) {
    std::string root = "CMake:/home/daisy/docc-llvm/cmake-build-debug";

    auto p = utils::DoccPaths::from_root(root);

    EXPECT_EQ(p.docc_root_path, "/home/daisy/docc-llvm/cmake-build-debug");
    EXPECT_EQ(p.root_mode, utils::DoccRootMode::CMake);
    std::vector<std::filesystem::path> expected_inc_paths = {
        "/home/daisy/docc-llvm/cmake-build-debug/../docc/rtl/include",
        "/home/daisy/docc-llvm/cmake-build-debug/../docc/arg-capture-io/include",
    };
    EXPECT_EQ(p.target_inc_paths(), expected_inc_paths);
}

TEST(DoccPathsTest, ResolveDistRoot) {
    std::string root = "Dist:/usr/lib/docc-0.1.6";

    auto p = utils::DoccPaths::from_root(root);

    EXPECT_EQ(p.docc_root_path, "/usr/lib/docc-0.1.6");
    EXPECT_EQ(p.root_mode, utils::DoccRootMode::Dist);
    std::vector<std::filesystem::path> expected_inc_paths = {
        "/usr/lib/docc-0.1.6/include",
    };
    EXPECT_EQ(p.target_inc_paths(), expected_inc_paths);
}

TEST(DoccPathsTest, ResolveNoneRoot) {
    std::string root = "None";

    auto p = utils::DoccPaths::from_root(root);

    EXPECT_EQ(p.root_mode, utils::DoccRootMode::None);
    std::vector<std::filesystem::path> expected_inc_paths = {};
    EXPECT_EQ(p.target_inc_paths(), expected_inc_paths);

    root = "";
    p = utils::DoccPaths::from_root(root);

    EXPECT_EQ(p.root_mode, utils::DoccRootMode::None);
    EXPECT_EQ(p.target_inc_paths(), expected_inc_paths);
}
