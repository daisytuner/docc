#pragma once

#include <chrono>

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
        std::chrono::high_resolution_clock::time_point start;
        if (passes::AnalysisStatistics::instance().enabled()) {
            start = std::chrono::high_resolution_clock::now();
        }

        cache_[type] = std::make_unique<T>(this->sdfg_);
        cache_[type]->additional_assumptions_ = this->additional_assumptions_;
        cache_[type]->run(*this);

        if (passes::AnalysisStatistics::instance().enabled()) {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            passes::AnalysisStatistics::instance().add_analysis(cache_[type]->name(), duration);
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

    void invalidate_all();
};

} // namespace analysis
} // namespace sdfg
