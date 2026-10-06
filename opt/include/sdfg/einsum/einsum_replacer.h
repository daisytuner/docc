#pragma once

#include <memory>
#include <string>
#include <vector>

#include <sdfg/builder/structured_sdfg_builder.h>
#include <sdfg/data_flow/access_node.h>
#include <sdfg/options.h>
#include <sdfg/passes/expansion/lib_node_expander.h>
#include <sdfg/structured_control_flow/control_flow_node.h>
#include <sdfg/structured_control_flow/sequence.h>

#include "sdfg/einsum/einsum_detection.h"

namespace sdfg::einsum {

typedef passes::LibNodeExpander::ExpandOutcome ReplaceOutcome;

/**
 * @brief ExpandContext for replacing an EinsumCluster with a new implementation.
 *
 * Mirrors the library-node expansion infrastructure (LibNodeExpansionContext) but is driven by
 * an analysis-only EinsumCluster instead of a materialized library node. It reuses the
 * AccessNodeExpand interface so replacers construct their replacement exactly like a library-node
 * expander would.
 *
 * The "replacement site" is the outermost loop consumed into the cluster (or, if none were
 * consumed, the block holding the reduction core). The replacement is inserted right after the
 * site and the original computation is removed on cleanup().
 */
class EinsumReplacementContext {
public:
    virtual ~EinsumReplacementContext() = default;

    virtual std::unique_ptr<passes::LibNodeExpander::AccessNodeExpand> replacement_requires_access_nodes(
        const std::vector<passes::LibNodeExpander::InputUse>& access_dirs, bool leave_unconsumed_indices
    ) = 0;

    virtual ReplaceOutcome successfully_modified_node_only() = 0;
    virtual ReplaceOutcome unable() = 0;
    virtual ReplaceOutcome unapplicable() = 0;
};


/**
 * @brief Interface for replacing an EinsumCluster with a different implementation.
 *
 * A replacer inspects a (read-only) EinsumCluster and, if applicable, uses the given context to
 * build a replacement via the AccessNodeExpand API.
 */
class EinsumReplacer {
public:
    virtual ~EinsumReplacer() = default;

    virtual std::string name() const = 0;

    virtual ReplaceOutcome replace(EinsumReplacementContext& context, const EinsumCluster& cluster) const = 0;
};

/**
 * @brief Drive a single replacement.
 *
 * Builds an EinsumReplacementContext, runs the replacer and, on success, removes the original
 * computation. Returns the replacer's outcome.
 */
ReplaceOutcome replace_einsum_cluster(
    builder::StructuredSDFGBuilder& builder, const EinsumCluster& cluster, const EinsumReplacer& replacer
);

} // namespace sdfg::einsum
