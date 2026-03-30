#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/data_flow/library_nodes/stdlib/malloc.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_sdfg.h"

namespace sdfg {
namespace analysis {

class EscapeAnalysis : public Analysis {
private:
    std::unordered_map<std::string, bool> escapes_;
    std::unordered_map<std::string, User*> last_uses_;
    std::unordered_set<std::string> malloc_containers_;
    std::unordered_map<std::string, structured_control_flow::Block*> malloc_blocks_;

protected:
    void run(analysis::AnalysisManager& analysis_manager) override;

public:
    EscapeAnalysis(StructuredSDFG& sdfg);

    std::string name() const override { return "EscapeAnalysis"; }

    bool is_malloc_allocation(const std::string& container) const;

    bool escapes(const std::string& container) const;

    User* last_use(const std::string& container) const;

    structured_control_flow::Block* malloc_block(const std::string& container) const;

    std::unordered_set<std::string> non_escaping_allocations() const;
};

} // namespace analysis
} // namespace sdfg
