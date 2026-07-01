#pragma once
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "utils.h"

namespace sdfg {
namespace codegen {

inline std::string CODE_SNIPPET_INIT_ONCE = "init_once";
inline std::string CODE_SNIPPET_DEINIT_ONCE = "deinit_once";

class CodeSnippet {
protected:
    PrettyPrinter stream_;
    std::string extension_;
    bool as_file_;
    std::string name_;

public:
    CodeSnippet(const std::string& name, const std::string& extension, bool as_file)
        : extension_(extension), as_file_(as_file), name_(name) {};

    PrettyPrinter& stream() { return stream_; }

    const PrettyPrinter& stream() const { return stream_; }

    const std::string& extension() const { return extension_; }

    bool is_as_file() const { return as_file_; }

    const std::string& name() const { return name_; }
};

class LibDependency {
public:
    virtual ~LibDependency() = default;
    virtual std::string_view name() const = 0;
    virtual void enumerate_includes(std::vector<std::string>& out_list) const = 0;
    virtual std::vector<std::string_view>& globally_unique_ids() const = 0;
};

struct DependencyState {
    bool used = false;
    bool runtime_available = false;
};

class CodeSnippetFactory {
protected:
    std::unordered_map<std::string, CodeSnippet> snippets_;
    const std::filesystem::path output_path_;
    const std::filesystem::path header_path_;

    std::unordered_set<std::string> setup_snippets_;
    std::unordered_set<std::string> teardown_snippets_;
    std::unordered_set<std::string> globals_snippets_;

    std::unordered_map<std::string, const LibDependency*> conflicting_ids_;
    std::unordered_map<const LibDependency*, DependencyState> dependencies_;

public:
    CodeSnippetFactory(const std::pair<std::filesystem::path, std::filesystem::path>* config = nullptr);

    virtual ~CodeSnippetFactory() = default;

    CodeSnippet& require(const std::string& name, const std::string& extension, bool as_file = true);

    std::unordered_map<std::string, CodeSnippet>::iterator find(const std::string& name);

    const std::unordered_map<std::string, CodeSnippet>& snippets() const;

    const std::filesystem::path& output_path() const { return output_path_; }
    const std::filesystem::path& header_path() const { return header_path_; }

    void add_setup(const std::string& snippet);
    void add_teardown(const std::string& snippet);
    void add_global(const std::string& snippet);
    const std::unordered_set<std::string>& setup_snippets() const;
    const std::unordered_set<std::string>& teardown_snippets() const;

    std::vector<const LibDependency*> get_used_lib_dependencies() const;

    const std::unordered_set<std::string>& globals_snippets() const;

    /**
     * For the generation process to declare discovered runtime dependencies before the codegen starts.
     * Reasons could be: user code requires that it exists, it has been found in the filesystem or the config says it is
     * always available For example tenstorrent debug support, only available if compiled this way locally, then the
     * headers of it may be relied upon. Otherwise it would crash
     */
    void add_available_dependency(const LibDependency* dependency);

    /**
     * The dependency is either already required or the system knows it is available
     */
    bool is_available(const LibDependency* dependency) const;

    /**
     * Checks if this dependency would conflict with existing ones.
     * @return conflict if any: {conflicting ID, source of that ID}
     */
    std::optional<std::pair<const std::string&, const LibDependency*>> check_for_conflicts(const LibDependency*
                                                                                               dependency) const;

    /**
     * Adds conflicts of this dependency to the list, throws if conflicts exist
     * Warning: not transactional for now!
     */
    void require_no_conflicts(const LibDependency* dependency);

    /**
     * Ensure that the given dependency is available or throw
     * @return true if the dependency is now newly used
     */
    bool require_dependency(const LibDependency* dependency);
};


} // namespace codegen
} // namespace sdfg
