#pragma once

#include "sdfg/passes/statistics.h"
#include "sdfg/structured_sdfg.h"

namespace sdfg {
namespace analysis {

class AnalysisManager;

class Analysis {
    friend class AnalysisManager;

protected:
    StructuredSDFG& sdfg_;
    symbolic::Assumptions additional_assumptions_;

    virtual void run(analysis::AnalysisManager& analysis_manager) = 0;

public:
    Analysis(StructuredSDFG& sdfg);

    virtual ~Analysis() = default;

    virtual std::string name() const = 0;

    Analysis(const Analysis& a) = delete;
    Analysis& operator=(const Analysis&) = delete;
};

class AnalysisManager {
private:
    StructuredSDFG& sdfg_;
    symbolic::Assumptions additional_assumptions_;

    std::unordered_map<std::type_index, std::unique_ptr<Analysis>> cache_;

    static constexpr const char* STATS_SCOPE = "AnalysisMgr";

public:
    AnalysisManager(StructuredSDFG& sdfg);
    AnalysisManager(StructuredSDFG& sdfg, const symbolic::Assumptions& additional_assumptions);

    AnalysisManager(const AnalysisManager& am) = delete;
    AnalysisManager& operator=(const AnalysisManager&) = delete;

    template<class T>
    T& get() {
        std::type_index type = std::type_index(typeid(T));

        // Check cache
        auto it = cache_.find(type);
        if (it != cache_.end()) {
            return *static_cast<T*>(it->second.get());
        }

        // Run a new analysis
        cache_[type] = std::make_unique<T>(this->sdfg_);
        cache_[type]->additional_assumptions_ = this->additional_assumptions_;

        passes::CompileStatistics* stats = nullptr;
        if (passes::CompileStatistics::enabled()) {
            stats = &passes::CompileStatistics::instance();
            stats->enter_scope(cache_[type]->name(), STATS_SCOPE);
        }

        cache_[type]->run(*this);

        if (stats) {
            stats->exit_scope();
        }

        return *static_cast<T*>(cache_[type].get());
    }

    template<class T>
    void invalidate() {
        std::type_index type = std::type_index(typeid(T));
        if (cache_.find(type) != cache_.end()) {
            cache_.erase(type);
        }
    }

    // Preserve only the listed analyses and invalidate all others.
    // Analyses not present in the cache are unaffected.
    template<class... Ts>
    void preserve() {
        std::unordered_map<std::type_index, std::unique_ptr<Analysis>> kept;
        auto try_keep = [&](std::type_index type) {
            auto it = cache_.find(type);
            if (it != cache_.end()) {
                kept.emplace(type, std::move(it->second));
            }
        };
        (try_keep(std::type_index(typeid(Ts))), ...);
        cache_.clear();
        for (auto& [type, analysis] : kept) {
            cache_.emplace(type, std::move(analysis));
        }
    }

    void invalidate_all();
};

} // namespace analysis
} // namespace sdfg
