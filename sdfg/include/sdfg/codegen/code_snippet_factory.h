#pragma once
#include <string>
#include <unordered_set>

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

class CodeSnippetFactory {
protected:
    std::unordered_map<std::string, CodeSnippet> snippets_;
    const std::filesystem::path output_path_;
    const std::filesystem::path header_path_;

    std::unordered_set<std::string> setup_snippets_;
    std::unordered_set<std::string> teardown_snippets_;
    std::unordered_set<std::string> globals_snippets_;

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
    const std::unordered_set<std::string>& globals_snippets() const;
};


} // namespace codegen
} // namespace sdfg
