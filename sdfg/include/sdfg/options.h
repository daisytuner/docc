#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>

namespace sdfg {

enum class OptionType { Bool, Int, Double, String };

using OptionValue = std::variant<bool, int64_t, double, std::string>;

template<class T>
constexpr OptionType option_type_of() {
    if constexpr (std::is_same_v<T, bool>) {
        return OptionType::Bool;
    } else if constexpr (std::is_same_v<T, int64_t>) {
        return OptionType::Int;
    } else if constexpr (std::is_same_v<T, double>) {
        return OptionType::Double;
    } else {
        static_assert(std::is_same_v<T, std::string>, "Unsupported option type");
        return OptionType::String;
    }
}

struct OptionSpec {
    std::string key;
    OptionType type;
    OptionValue default_value;
    std::string doc;
};

template<class T>
struct OptionKey {
    std::string_view key;
    constexpr explicit OptionKey(std::string_view k) : key(k) {
    }

    // Build the registry spec from this handle so key and type aren't restated.
    OptionSpec spec(T default_value, std::string doc) const {
        return {std::string(key), option_type_of<T>(), OptionValue{std::move(default_value)}, std::move(doc)};
    }
};

// A per-run bag of option values, keyed by option name. Shared by passes,
// analyses, and any helper that is handed one; not tied to passes.
class Options {
public:
    void set(std::string key, OptionValue value) {
        values_.insert_or_assign(std::move(key), std::move(value));
    }

    bool has(std::string_view key) const {
        return values_.find(std::string(key)) != values_.end();
    }

    template<class T>
    T get(const OptionKey<T>& key, T fallback = T{}) const {
        auto it = values_.find(std::string(key.key));
        if (it == values_.end()) {
            return fallback;
        }
        if (auto* v = std::get_if<T>(&it->second)) {
            return *v;
        }
        return fallback;
    }

    // Shared empty bag so consumers run with defaults when nothing is forwarded.
    static const Options& empty() {
        static const Options e;
        return e;
    }

private:
    std::unordered_map<std::string, OptionValue> values_;
};

// Registry of option specs for discovery/validation. Options are registered
// independently of who consumes them (passes, analyses, helpers).
class OptionRegistry {
public:
    void register_option(const OptionSpec& spec) {
        if (!options_.emplace(spec.key, spec).second) {
            throw std::runtime_error("Duplicate option key: " + spec.key);
        }
    }

    const std::unordered_map<std::string, OptionSpec>& options() const {
        return options_;
    }

    const OptionSpec* find_option(std::string_view key) const {
        auto it = options_.find(std::string(key));
        return it == options_.end() ? nullptr : &it->second;
    }

private:
    std::unordered_map<std::string, OptionSpec> options_;
};

} // namespace sdfg
