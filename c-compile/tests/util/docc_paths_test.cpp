#include "docc/util/docc_paths.h"

#include <gtest/gtest.h>

using namespace docc;

void EXPECT_EQ_PATHS(std::string path1, std::string path2) {
    if (path1.back() == '/') {
        path1.pop_back();
    }
    if (path2.back() == '/') {
        path2.pop_back();
    }
    EXPECT_EQ(path1, path2);
}

void EXPECT_EQ_PATHS(std::vector<std::filesystem::path> paths1, std::vector<std::filesystem::path> paths2) {
    ASSERT_EQ(paths1.size(), paths2.size());
    for (size_t i = 0; i < paths1.size(); i++) {
        EXPECT_EQ_PATHS(paths1.at(i), paths2.at(i));
    }
}

TEST(DoccPathsTest, ResolveCmakeSuffixRoot) {
    std::string root = "CMake:/home/daisy/docc-llvm/cmake-build-debug/docc:/home/daisy/docc-llvm/docc";

    auto p = util::DefaultDoccPaths::from_root(root);

    EXPECT_EQ_PATHS(p->bin_root(), "/home/daisy/docc-llvm/cmake-build-debug/docc");
    EXPECT_EQ_PATHS(p->src_root(), "/home/daisy/docc-llvm/docc");
    EXPECT_EQ(p->root_mode(), util::DefaultDoccPaths::DoccRootMode::CMake);
    std::vector<std::filesystem::path> expected_inc_paths = {
        "/home/daisy/docc-llvm/docc/rtl/include",
        "/home/daisy/docc-llvm/docc/arg-capture-io/include",
    };
    EXPECT_EQ_PATHS(p->get_default_include_paths(), expected_inc_paths);
}

TEST(DoccPathsTest, FromLibraryLocation) {
    auto p = util::DefaultDoccPaths::from_lib_location(util::find_lib_location());

    std::filesystem::path ref_bin_root = std::filesystem::path(DOCC_BIN_ROOT).lexically_normal();
    std::filesystem::path ref_src_root = std::filesystem::path(DOCC_SRC_ROOT).lexically_normal();

    EXPECT_EQ_PATHS(p->bin_root().lexically_normal(), ref_bin_root);
    EXPECT_EQ_PATHS(p->src_root().lexically_normal(), ref_src_root);
    EXPECT_EQ(p->root_mode(), util::DefaultDoccPaths::DoccRootMode::CMake);
    std::vector<std::filesystem::path> expected_inc_paths = {
        ref_src_root / "rtl" / "include",
        ref_src_root / "arg-capture-io" / "include",
    };
    auto inc_paths = p->get_default_include_paths();
    for (int i = 0; i < expected_inc_paths.size(); i++) {
        EXPECT_EQ_PATHS(inc_paths.at(i).lexically_normal(), expected_inc_paths.at(i).lexically_normal());
    }
}

TEST(DoccPathsTest, ResolveCmakeDirectRoot) {
    std::string root = "CMake:/home/daisy/docc/cmake-build-debug:/home/daisy/docc";

    auto p = util::DefaultDoccPaths::from_root(root);

    EXPECT_EQ_PATHS(p->bin_root(), "/home/daisy/docc/cmake-build-debug");
    EXPECT_EQ_PATHS(p->src_root(), "/home/daisy/docc");
    EXPECT_EQ(p->root_mode(), util::DefaultDoccPaths::DoccRootMode::CMake);
    std::vector<std::filesystem::path> expected_inc_paths = {
        "/home/daisy/docc/rtl/include",
        "/home/daisy/docc/arg-capture-io/include",
    };
    EXPECT_EQ_PATHS(p->get_default_include_paths(), expected_inc_paths);
}

TEST(DoccPathsTest, ResolveDistRoot) {
    std::string root = "Dist:/usr/lib/docc-0.1.6";

    auto p = util::DefaultDoccPaths::from_root(root);

    EXPECT_EQ_PATHS(p->bin_root(), "/usr/lib/docc-0.1.6/bin");
    EXPECT_EQ_PATHS(p->src_root(), "/usr/lib/docc-0.1.6");
    EXPECT_EQ(p->root_mode(), util::DefaultDoccPaths::DoccRootMode::Dist);
    std::vector<std::filesystem::path> expected_inc_paths = {
        "/usr/lib/docc-0.1.6/include",
    };
    EXPECT_EQ_PATHS(p->get_default_include_paths(), expected_inc_paths);
}

TEST(DoccPathsTest, ResolveNoneRoot) {
    std::string root = "None";

    auto p = util::DefaultDoccPaths::from_root(root);

    EXPECT_EQ(p->root_mode(), util::DefaultDoccPaths::DoccRootMode::None);
    std::vector<std::filesystem::path> expected_inc_paths = {};
    EXPECT_EQ_PATHS(p->get_default_include_paths(), expected_inc_paths);

    root = "";
    p = util::DefaultDoccPaths::from_root(root);

    EXPECT_EQ(p->root_mode(), util::DefaultDoccPaths::DoccRootMode::None);
    EXPECT_EQ_PATHS(p->get_default_include_paths(), expected_inc_paths);
}
