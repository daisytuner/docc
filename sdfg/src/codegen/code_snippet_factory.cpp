#include "sdfg/codegen/code_snippet_factory.h"
#include <algorithm>
#include <unordered_set>

namespace sdfg::codegen {

CodeSnippetFactory::CodeSnippetFactory(const std::pair<std::filesystem::path, std::filesystem::path>* config)
    : output_path_(config ? config->first : "."), header_path_(config ? config->second : "") {}


CodeSnippet& CodeSnippetFactory::require(const std::string& name, const std::string& extension, bool as_file) {
    auto [snippet, newly_created] = snippets_.try_emplace(name, name, extension, as_file);

    if (!newly_created && extension != snippet->second.extension()) {
        throw std::runtime_error(
            "Code snippet " + name + " already exists with '." + snippet->second.extension() +
            "', but was required with '." + extension + "'"
        );
    }

    return snippet->second;
}

std::unordered_map<std::string, CodeSnippet>::iterator CodeSnippetFactory::find(const std::string& name) {
    return snippets_.find(name);
}
const std::unordered_map<std::string, CodeSnippet>& CodeSnippetFactory::snippets() const { return snippets_; }

void CodeSnippetFactory::add_setup(const std::string& snippet) { setup_snippets_.insert(snippet); }

void CodeSnippetFactory::add_teardown(const std::string& snippet) { teardown_snippets_.insert(snippet); }

void CodeSnippetFactory::add_global(const std::string& snippet) { globals_snippets_.insert(snippet); }

const std::unordered_set<std::string>& CodeSnippetFactory::setup_snippets() const { return setup_snippets_; }

const std::unordered_set<std::string>& CodeSnippetFactory::teardown_snippets() const { return teardown_snippets_; }

std::vector<const LibDependency*> CodeSnippetFactory::get_used_lib_dependencies() const {
    std::vector<const LibDependency*> dependencies;
    dependencies.reserve(dependencies_.size());

    for (const auto& [dependency, state] : dependencies_) {
        dependencies.push_back(dependency);
    }

    std::sort(dependencies.begin(), dependencies.end(), [](const LibDependency* lhs, const LibDependency* rhs) {
        return lhs->name() < rhs->name();
    });

    return dependencies;
}

const std::unordered_set<std::string>& CodeSnippetFactory::globals_snippets() const { return globals_snippets_; }

void CodeSnippetFactory::add_available_dependency(const LibDependency* dependency) {
    dependencies_.emplace(dependency, DependencyState{.used = false, .runtime_available = true});
}

bool CodeSnippetFactory::is_available(const LibDependency* dependency) const {
    auto it = dependencies_.find(dependency);
    return it != dependencies_.end();
}

std::optional<std::pair<const std::string&, const LibDependency*>> CodeSnippetFactory::
    check_for_conflicts(const LibDependency* dependency) const {
    auto& ids = dependency->globally_unique_ids();
    for (auto& id : ids) {
        std::string s(id);
        auto it = conflicting_ids_.find(s);
        if (it != conflicting_ids_.end()) {
            return {{s, it->second}};
        }
    }
    return std::nullopt;
}

void CodeSnippetFactory::require_no_conflicts(const LibDependency* dependency) {
    auto& ids = dependency->globally_unique_ids();
    for (auto& id : ids) {
        auto [it, newly_created] = conflicting_ids_.emplace(id, dependency);
        if (!newly_created) {
            throw std::runtime_error(
                "Cannot add '" + std::string(dependency->name()) + "', conflicting globally unique id '" +
                std::string(id) + "' with '" + std::string(it->second->name()) + "'"
            );
        }
    }
}

bool CodeSnippetFactory::require_dependency(const LibDependency* dependency) {
    auto [it, newly_created] = dependencies_.emplace(dependency, DependencyState{.used = true});
    if (newly_created) {
        require_no_conflicts(dependency);
        return true;
    } else {
        auto& state = it->second;
        if (!state.used) {
            require_no_conflicts(dependency);
            state.used = true;

            return true;
        } else {
            return false;
        }
    }
}

} // namespace sdfg::codegen
