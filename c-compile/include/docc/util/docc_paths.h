#pragma once

#include <filesystem>
#include <optional>
#include <vector>

namespace docc::util {

std::optional<std::filesystem::path> find_lib_location();

class DoccPaths {
public:
    virtual ~DoccPaths() = default;

    [[nodiscard]] virtual std::vector<std::filesystem::path> get_default_include_paths() = 0;
    [[nodiscard]] virtual std::vector<std::filesystem::path> get_default_library_paths() = 0;
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

    static std::unique_ptr<DefaultDoccPaths> from_lib_location(std::optional<std::filesystem::path> lib_location);

    std::vector<std::filesystem::path> get_default_include_paths() override;
    std::vector<std::filesystem::path> get_default_library_paths() override;
};


} // namespace docc::util
