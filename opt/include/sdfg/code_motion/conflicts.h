#pragma once

#include <map>
#include <optional>
#include <string>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/memory_layout_analysis.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/symbolic/assumptions.h"

namespace sdfg {
namespace code_motion {

/**
 * Whether @p node may run at another point of the program: it transfers no control (return, break, continue) and
 * calls no library node with side effects other than its accesses: a library node with side effects may move if it
 * has memlets and describes every pointer it takes (LibraryNode::pointer_use_creates_side_effects), as allocations,
 * frees and tile copies do.
 */
bool movable(StructuredSDFG& sdfg, structured_control_flow::ControlFlowNode& node);

/// A box of elements of a container: per dimension of its index space, the first and last index.
struct Box {
    analysis::MemoryLayout layout;
    data_flow::Subset lower;
    data_flow::Subset upper;
};

/// How a subtree uses a container.
struct ContainerUse {
    bool reads = false;
    bool writes = false;
    /// The boxes around the elements it may access; none if any.
    std::optional<std::vector<Box>> boxes;
};

/// How a subtree uses each container it uses: computed once, compared with several others.
using Accesses = std::map<std::string, ContainerUse>;

/**
 * How @p node uses each container it uses. The elements a subtree accesses are the tiles MemoryLayoutAnalysis bounds
 * them by: a block's per memlet, another node's per tile group. A container is accessed anywhere if the subtree uses
 * it as a symbol, views, moves, captures or invalidates its pointer, or touches it through a memlet no tile covers.
 * Whether it reads or writes a container follows the users and the library nodes' pointer metadata.
 */
Accesses accesses_of(
    StructuredSDFG& sdfg, analysis::AnalysisManager& analysis_manager, structured_control_flow::ControlFlowNode& node
);

/// The kinds of dependences between two subtrees @ref may_conflict looks for.
struct Dependences {
    bool flow = true; ///< the first writes what the second reads
    bool anti = true; ///< the first reads what the second writes
    bool output = true; ///< both write the same
};

/**
 * Whether subtrees accessing @p first and @p second may depend on one another by one of @p kinds: unless, for every
 * container the kinds relate, the elements they access are disjoint. With all kinds, the two may compute differently
 * in the other order.
 *
 * @param scope Assumptions on the symbols both use. Symbols constant in it take the same value in both subtrees;
 *        the others, e.g. the counter of a loop around both marked non-constant, take any values in each.
 */
bool may_conflict(
    const Accesses& first, const Accesses& second, const symbolic::Assumptions& scope, Dependences kinds = {}
);

/// @ref may_conflict of the accesses of @p first and @p second.
bool may_conflict(
    StructuredSDFG& sdfg,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::ControlFlowNode& first,
    structured_control_flow::ControlFlowNode& second,
    const symbolic::Assumptions& scope,
    Dependences kinds = {}
);

/// Whether a subtree with @p accesses reads a container it writes: running it again may then compute differently.
bool reads_what_it_writes(const Accesses& accesses);

} // namespace code_motion
} // namespace sdfg
