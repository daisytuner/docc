#pragma once

#include <string>

#include "sdfg/passes/pass.h"

namespace sdfg {
namespace passes {

/**
 * @brief Writes a result straight into the array a contiguous copy of it fills.
 *
 * Frontends return a view of an intermediate, such as a reshape of a sum, by copying the intermediate into the
 * result. When a block of the root sequence only copies a transient T, element for element in the same order, into a
 * pointer argument or external R that nothing before it touches, and after it T is only freed, everything before the
 * copy writes and reads R instead of T: the copy, T's allocation and its free go away.
 */
class ViewCopyElimination : public Pass {
public:
    std::string name() override;

    bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;
};

} // namespace passes
} // namespace sdfg
