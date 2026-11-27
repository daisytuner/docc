#pragma once

#include <set>
#include <string>

#include "sdfg/passes/pass.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace passes {

/**
 * @brief Widens the float types a target does not store to Float, which holds all their values exactly.
 *
 * Every container, memlet and constant of one of the given types, alone or as the element of a pointer, array or
 * tensor, takes Float instead, and so do the types library nodes store themselves (cast targets, quantizations,
 * contraction and cmath types); tasklets then compute in Float. It runs after the contraction precision, which picks
 * the precision of each contraction by its operands' own types. Arguments change type too: their original types are
 * recorded in the metadata ARGUMENT_TYPES as a JSON object of argument names and type names, from which the runtime
 * converts what the caller passes and returns, so that callers keep their types.
 */
class FloatWidening : public Pass {
public:
    /// Metadata key of the original types of the widened arguments.
    static constexpr const char* ARGUMENT_TYPES = "widened_argument_types";

    explicit FloatWidening(std::set<types::PrimitiveType> widened);

    virtual std::string name() override;

    virtual bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

private:
    std::set<types::PrimitiveType> widened_;
};

} // namespace passes
} // namespace sdfg
