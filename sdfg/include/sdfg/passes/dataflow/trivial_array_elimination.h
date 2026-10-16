#pragma once

#include <string>
#include <unordered_set>

#include "sdfg/passes/pass.h"

namespace sdfg {
namespace passes {

class TrivialArrayElimination : public Pass {
    std::unordered_set<std::string> containers_;

public:
    TrivialArrayElimination();

    /// Only eliminates the trivial dimensions of @p containers.
    explicit TrivialArrayElimination(std::unordered_set<std::string> containers);

    virtual std::string name() override;

    virtual bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;
};

} // namespace passes
} // namespace sdfg
