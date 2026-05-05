#pragma once

#include <filesystem>
#include <optional>
#include <vector>

#include "sdfg/plugins/plugins.h"

namespace docc::util {

std::optional<std::filesystem::path> find_lib_location();

class DoccPaths {
public:
    virtual ~DoccPaths() = default;

    [[nodiscard]] virtual std::vector<std::filesystem::path> get_default_include_paths() const = 0;
    [[nodiscard]] virtual std::vector<std::filesystem::path> get_default_library_paths() const = 0;
};

class DefaultDoccPaths : public DoccPaths {
public:
    enum class DoccRootMode {
        None, // dont know
        CMake, // from source in git folder structure
        Pip, // installed as a python package
        GlobalDist // expect all relevant parts to be installed in globally available and searched paths
    };

private:
    std::filesystem::path bin_root_;
    std::filesystem::path src_root_;
    DoccRootMode root_mode_;

public:
    DefaultDoccPaths(std::filesystem::path bin_root, std::filesystem::path src_root, DoccRootMode root_mode);

    const std::filesystem::path& bin_root() const { return bin_root_; }
    const std::filesystem::path& src_root() const { return src_root_; }
    DoccRootMode root_mode() const { return root_mode_; }

    static std::unique_ptr<DefaultDoccPaths> from_lib_location(std::optional<std::filesystem::path> lib_location);

    std::vector<std::filesystem::path> get_default_include_paths() const override;
    std::vector<std::filesystem::path> get_default_library_paths() const override;

    std::optional<std::filesystem::path> get_builtin_target_plugin_src_path(const std::string& plugin) const;
    std::optional<std::filesystem::path> get_builtin_target_plugin_rt_lib_path(const std::string& plugin) const;
    std::optional<std::filesystem::path> get_builtin_target_plugin_rt_include_path(const std::string& plugin) const;
    std::optional<std::filesystem::path>
    get_builtin_target_plugin_rt_libexec_src_path(const std::string& plugin, const std::string& rt_name) const;
};

} // namespace docc::util
