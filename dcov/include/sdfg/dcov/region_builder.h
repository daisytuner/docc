#pragma once

#include "sdfg/dcov/region.h"
#include "sdfg/structured_sdfg.h"

namespace sdfg {
namespace dcov {

/**
 * @brief Partitions a StructuredSDFG into a region @ref Module.
 *
 * The builder walks the structured control-flow tree and produces:
 * - one synthetic "function" root region (parent of top-level atoms),
 * - one region per structured loop (for/map/reduce/while),
 * - one region per library node,
 * while attaching tasklet atoms to their nearest enclosing region.
 *
 * Region keys are semantic fingerprints (see @ref region.h) and are independent
 * of element ids, schedule, and target.
 */
class RegionBuilder {
public:
    RegionBuilder() = default;

    /**
     * @brief Build the region model for a deserialized SDFG.
     * @param sdfg The (typically py4.norm) structured SDFG. Non-const because
     *             loop metadata is sourced from LoopAnalysis.
     * @param build_config Optional (key,value) labels (target, frontend, ...).
     */
    Module build(StructuredSDFG& sdfg, const std::vector<std::pair<std::string, std::string>>& build_config = {});
};

} // namespace dcov
} // namespace sdfg
