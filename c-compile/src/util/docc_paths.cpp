#include "docc/util/docc_paths.h"

#include <dlfcn.h>
#include <regex>
#include <string>
#include <utility>

namespace docc::util {

namespace {
void _anchor() {}
} // namespace


std::optional<std::filesystem::path> find_lib_location() {
    Dl_info info;
    if (dladdr((void*) &_anchor, &info)) {
        return std::filesystem::canonical(info.dli_fname);
    } else {
        return std::nullopt;
    }
}

DefaultDoccPaths::DefaultDoccPaths(std::filesystem::path bin_root, std::filesystem::path src_root, DoccRootMode root_mode)
    : bin_root_(std::move(bin_root)), src_root_(std::move(src_root)), root_mode_(root_mode) {}

std::unique_ptr<DefaultDoccPaths> DefaultDoccPaths::from_lib_location(std::optional<std::filesystem::path> lib_location
) {
    if (lib_location) {
        auto str = lib_location->string();
        std::regex pip_matcher(".*/python3\\.[0-9]+/site-packages/.*");
        if (std::regex_match(str, pip_matcher)) {
            auto parent = lib_location->parent_path();
            while (!parent.empty() && parent.has_parent_path() && parent.stem().string() != "docc") {
                parent = parent.parent_path();
            }
            return std::make_unique<DefaultDoccPaths>(parent, "", DoccRootMode::Pip);
        }
        std::regex cmake_build_matcher(".*/.*build.*/.*");
        if (std::regex_match(str, cmake_build_matcher)) {
            std::filesystem::path parent = *lib_location;
            std::regex build_dir_matcher(".*build.*");
            bool is_build_dir;
            do {
                parent = parent.parent_path();
                is_build_dir = std::regex_match(parent.stem().string(), build_dir_matcher);
                if (is_build_dir) {
                    auto rel = std::filesystem::relative(*lib_location, parent);
                    if (!rel.empty()) {
                        auto outermost = *rel.begin();
                        std::string suffix = "";
                        std::optional<std::filesystem::path> test_path;
                        if (outermost.string() == "docc") {
                            test_path = parent / ".." / "docc" / "sdfg";
                            suffix = "docc";
                        } else {
                            test_path = parent / ".." / "sdfg";
                        }

                        if (test_path && std::filesystem::exists(*test_path) &&
                            std::filesystem::is_directory(*test_path)) {
                            if (suffix.empty()) {
                                return std::make_unique<DefaultDoccPaths>(parent, parent / "..", DoccRootMode::CMake);
                            } else {
                                return std::make_unique<
                                    DefaultDoccPaths>(parent / suffix, parent / ".." / suffix, DoccRootMode::CMake);
                            }
                        }
                    }
                }
            } while (!is_build_dir && !parent.empty());
        }
    }

    return std::make_unique<DefaultDoccPaths>("", "", DoccRootMode::None);
}

std::vector<std::filesystem::path> DefaultDoccPaths::get_default_include_paths() {
    switch (root_mode_) {
        case DoccRootMode::Pip:
            return {bin_root_ / "include"};
        case DoccRootMode::CMake: // src root must point to the root of docc.git
            return {src_root_ / "rtl" / "include", src_root_ / "arg-capture-io" / "include"};
        case DoccRootMode::None:
        case DoccRootMode::GlobalDist:
        default:
            return {};
    }
}

std::vector<std::filesystem::path> DefaultDoccPaths::get_default_library_paths() {
    switch (root_mode_) {
        case DoccRootMode::Pip:
            return {bin_root_ / "lib"};
        case DoccRootMode::CMake: // bin root must point to the cmake build dir of docc
            return {bin_root_ / "rtl", bin_root_ / "arg-capture-io"};
        case DoccRootMode::None:
        case DoccRootMode::GlobalDist:
        default:
            return {};
    }
}

} // namespace docc::util
