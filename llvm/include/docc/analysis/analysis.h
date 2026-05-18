#pragma once

#include <memory>
#include <mutex>
#include <typeindex>
#include <unordered_map>

namespace docc {
namespace analysis {

class SDFGRegistry;
class AnalysisManager;

class Analysis {
    friend class AnalysisManager;
    virtual void anchor();

protected:
    virtual void run(AnalysisManager& AM) = 0;

public:
    virtual ~Analysis() = default;
};

class AnalysisManager {
    friend class SDFGRegistry;

private:
    std::recursive_mutex lock_;
    std::unordered_map<std::type_index, std::unique_ptr<Analysis>> cache_;

public:
    AnalysisManager() = default;

    AnalysisManager(const AnalysisManager&) = delete;
    AnalysisManager& operator=(const AnalysisManager&) = delete;

    template<class T>
    static bool available(AnalysisManager& AM) {
        return T::available(AM);
    }

    template<class T>
    T& get() {
        std::lock_guard<std::recursive_mutex> lock(lock_);

        std::type_index Key = std::type_index(typeid(T));
        auto It = cache_.find(Key);
        if (It != cache_.end()) return *static_cast<T*>(It->second.get());

        cache_[Key] = std::make_unique<T>();
        cache_[Key]->run(*this);

        return *static_cast<T*>(cache_[Key].get());
    }
};

} // namespace analysis
} // namespace docc
