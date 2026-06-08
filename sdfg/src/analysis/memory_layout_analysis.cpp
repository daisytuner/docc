#include "sdfg/analysis/memory_layout_analysis.h"

#include <algorithm>
#include <optional>
#include <set>
#include <unordered_set>

#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/structured_control_flow/while.h"
#include "sdfg/symbolic/delinearization.h"
#include "sdfg/symbolic/extreme_values.h"
#include "sdfg/symbolic/polynomials.h"

namespace sdfg {
namespace analysis {

namespace {

// Sentinel symbol stored in shape[0] of a MemoryLayout when the leading dimension's
// extent is unknown (raw pointer accesses). The symbol never escapes the analysis:
// any expression that mentions it must be reported to the caller as `SymEngine::null`
// from the public size accessors (see `MemoryTile::extents()` etc.).
constexpr const char* kUnboundedName = "__unbounded__";

bool is_unbounded_dim(const symbolic::Expression& e) {
    if (e.is_null()) return false;
    if (!SymEngine::is_a<SymEngine::Symbol>(*e)) return false;
    return SymEngine::down_cast<const SymEngine::Symbol&>(*e).get_name() == kUnboundedName;
}

bool depends_on_unbounded(const symbolic::Expression& e) {
    if (e.is_null()) return false;
    for (const auto& a : symbolic::atoms(e)) {
        if (is_unbounded_dim(a)) return true;
    }
    return false;
}

bool layout_has_unbounded_first_dim(const MemoryLayout& layout) {
    const auto& shape = layout.shape();
    return !shape.empty() && is_unbounded_dim(shape[0]);
}

// Collect immediate child scopes (Sequence/IfElse/While/StructuredLoop) of a given
// scope that carry their own MemoryTile entries. Blocks are excluded because their
// per-memlet info is held in `accesses_`, not in `tiles_`/`tile_groups_`.
void collect_direct_child_scopes(
    structured_control_flow::ControlFlowNode& scope, std::set<const structured_control_flow::ControlFlowNode*>& result
) {
    if (auto* loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&scope)) {
        result.insert(&loop->root());
    } else if (auto* w = dynamic_cast<structured_control_flow::While*>(&scope)) {
        result.insert(&w->root());
    } else if (auto* seq = dynamic_cast<structured_control_flow::Sequence*>(&scope)) {
        for (size_t i = 0; i < seq->size(); i++) {
            auto& child = seq->at(i).first;
            if (!dynamic_cast<structured_control_flow::Block*>(&child)) {
                result.insert(&child);
            }
        }
    } else if (auto* ife = dynamic_cast<structured_control_flow::IfElse*>(&scope)) {
        for (size_t i = 0; i < ife->size(); i++) {
            result.insert(&ife->at(i).first);
        }
    }
}
} // namespace

MemoryLayoutAnalysis::MemoryLayoutAnalysis(StructuredSDFG& sdfg) : Analysis(sdfg) {}

void MemoryLayoutAnalysis::run(analysis::AnalysisManager& analysis_manager) {
    accesses_.clear();
    tiles_.clear();
    tile_groups_.clear();
    traverse(sdfg_.root(), analysis_manager);
}

void MemoryLayoutAnalysis::
    traverse(structured_control_flow::ControlFlowNode& node, analysis::AnalysisManager& analysis_manager) {
    // Snapshot current memlets and tile keys before recursing into the scope's children
    std::vector<const data_flow::Memlet*> memlets_before;
    memlets_before.reserve(accesses_.size());
    for (const auto& entry : accesses_) {
        memlets_before.push_back(entry.first);
    }
    std::set<std::pair<const structured_control_flow::ControlFlowNode*, std::string>> tiles_before;
    for (const auto& entry : tiles_) {
        tiles_before.insert(entry.first);
    }

    if (auto block = dynamic_cast<structured_control_flow::Block*>(&node)) {
        process_block(*block, analysis_manager);
    } else if (auto sequence = dynamic_cast<structured_control_flow::Sequence*>(&node)) {
        for (size_t i = 0; i < sequence->size(); i++) {
            traverse(sequence->at(i).first, analysis_manager);
        }
    } else if (auto if_else = dynamic_cast<structured_control_flow::IfElse*>(&node)) {
        for (size_t i = 0; i < if_else->size(); i++) {
            traverse(if_else->at(i).first, analysis_manager);
        }
    } else if (auto while_stmt = dynamic_cast<structured_control_flow::While*>(&node)) {
        traverse(while_stmt->root(), analysis_manager);
    } else if (auto loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&node)) {
        traverse(loop->root(), analysis_manager);
    } else {
        // Break, Continue, Return nodes don't contain blocks
        return;
    }

    // Merge tiles for containers accessed within this scope
    merge_scope_layouts(node, memlets_before, tiles_before, analysis_manager);
}

void MemoryLayoutAnalysis::
    process_block(structured_control_flow::Block& block, analysis::AnalysisManager& analysis_manager) {
    auto& assumptions_analysis = analysis_manager.get<AssumptionsAnalysis>();
    // Use trivial bounds (type-derived, e.g. unsigned >= 0) so delinearization
    // can soundly discharge non-negativity proof obligations on parameters.
    auto& assumptions = assumptions_analysis.get(block, /*include_trivial_bounds=*/true);

    auto& dfg = block.dataflow();
    for (auto& memlet : dfg.edges()) {
        const auto& subset = memlet.subset();
        if (subset.empty()) {
            continue;
        }

        // Get container name from the AccessNode (either src or dst)
        std::string container_name;
        if (auto* access = dynamic_cast<const data_flow::AccessNode*>(&memlet.src())) {
            container_name = access->data();
        } else if (auto* access = dynamic_cast<const data_flow::AccessNode*>(&memlet.dst())) {
            container_name = access->data();
        } else {
            continue; // Skip memlets without AccessNode
        }

        auto& base_type = memlet.base_type();
        switch (base_type.type_id()) {
            case types::TypeID::Scalar:
            case types::TypeID::Structure:
                continue; // Skip scalars and structures
            case types::TypeID::Tensor: {
                // Tensor types already contain layout information, so we can directly store it without delinearization
                auto& tensor_type = dynamic_cast<const types::Tensor&>(memlet.base_type());

                MemoryLayout layout(tensor_type.shape(), tensor_type.strides(), tensor_type.offset());
                MemoryAccess layout_info{container_name, subset, layout, true};
                this->accesses_.emplace(&memlet, layout_info);
                continue;
            }
            case types::TypeID::Array: {
                // Arrays are c-like stack array, so we can infer a simple row-major layout without needing
                // delinearization
                auto* array_type = dynamic_cast<const types::Array*>(&memlet.base_type());
                symbolic::MultiExpression shape = {array_type->num_elements()};
                while (array_type->element_type().type_id() == types::TypeID::Array) {
                    array_type = dynamic_cast<const types::Array*>(&array_type->element_type());
                }
                if (array_type->element_type().type_id() != types::TypeID::Scalar) {
                    continue; // Skip non-scalar arrays
                }

                MemoryLayout layout(shape);
                MemoryAccess layout_info{container_name, subset, layout, true};
                this->accesses_.emplace(&memlet, layout_info);
                continue;
            }
            case types::TypeID::Pointer: {
                // For pointers, we attempt to delinearize the access pattern to infer the layout based
                // on assumptions from loop bounds
                auto* pointer_type = dynamic_cast<const types::Pointer*>(&memlet.base_type());

                // Typed pointer to a (possibly multi-dim) fixed array of scalar,
                // e.g. `float (*A)[M]`. The pointer adds one unbounded leading
                // dimension; remaining dimensions come from the array shape. The
                // subset is expected to be one index per dimension — no
                // delinearization needed.
                if (pointer_type->pointee_type().type_id() == types::TypeID::Array) {
                    auto* array_type = dynamic_cast<const types::Array*>(&pointer_type->pointee_type());
                    symbolic::MultiExpression array_shape = {array_type->num_elements()};
                    while (array_type->element_type().type_id() == types::TypeID::Array) {
                        array_type = dynamic_cast<const types::Array*>(&array_type->element_type());
                        array_shape.push_back(array_type->num_elements());
                    }
                    if (array_type->element_type().type_id() != types::TypeID::Scalar) {
                        continue; // Skip non-scalar leaf
                    }
                    if (subset.size() != array_shape.size() + 1) {
                        continue; // Require one index per dimension (leading pointer + array dims)
                    }

                    symbolic::MultiExpression shape;
                    shape.push_back(symbolic::symbol("__unbounded__"));
                    for (const auto& dim : array_shape) {
                        shape.push_back(dim);
                    }

                    MemoryLayout layout(shape);
                    MemoryAccess layout_info{container_name, subset, layout, false};
                    this->accesses_.emplace(&memlet, layout_info);
                    continue;
                }

                if (pointer_type->pointee_type().type_id() != types::TypeID::Scalar) {
                    continue; // Skip non-scalar pointers
                }

                if (subset.size() != 1) {
                    continue; // Require full linearization
                }
                auto& linearized_expr = subset.at(0);

                auto result = symbolic::delinearize(linearized_expr, assumptions);
                if (!result.success) {
                    continue; // Delinearization failed, skip
                }

                // Delinearization returns N indices but only N-1 dimensions (from stride division)
                // The first dimension is unbounded - insert a placeholder that will be filled in by merge
                // Using a special symbol as placeholder for the first dimension
                symbolic::MultiExpression shape;
                shape.push_back(symbolic::symbol("__unbounded__"));
                for (const auto& dim : result.dimensions) {
                    shape.push_back(dim);
                }

                // Store symbolic indices and dimensions with unbounded first dimension
                // The merge phase will attempt to bound the first dimension using loop assumptions
                MemoryLayout layout(shape);
                MemoryAccess layout_info{container_name, result.indices, layout, false};
                this->accesses_.emplace(&memlet, layout_info);
                continue;
            }
            default:
                continue; // Skip unsupported types
        }
    }
}

const MemoryAccess* MemoryLayoutAnalysis::access(const data_flow::Memlet& memlet) const {
    auto layout_it = accesses_.find(&memlet);
    if (layout_it == accesses_.end()) {
        return nullptr;
    }
    return &layout_it->second;
}

void MemoryLayoutAnalysis::merge_scope_layouts(
    structured_control_flow::ControlFlowNode& scope,
    const std::vector<const data_flow::Memlet*>& memlets_before,
    const std::set<std::pair<const structured_control_flow::ControlFlowNode*, std::string>>& tiles_before,
    analysis::AnalysisManager& analysis_manager
) {
    // Convert memlets_before to a set for O(1) lookup
    std::unordered_set<const data_flow::Memlet*> before_set(memlets_before.begin(), memlets_before.end());

    // Group all new accesses by container
    std::unordered_map<std::string, std::vector<const data_flow::Memlet*>> all_container_groups;
    for (auto& [memlet_ptr, acc] : accesses_) {
        if (before_set.find(memlet_ptr) != before_set.end()) {
            continue;
        }
        all_container_groups[acc.container].push_back(memlet_ptr);
    }

    // Sort memlets within each container group by element_id for deterministic processing order
    for (auto& [container, memlets] : all_container_groups) {
        std::sort(memlets.begin(), memlets.end(), [](const data_flow::Memlet* a, const data_flow::Memlet* b) {
            return a->element_id() < b->element_id();
        });
    }

    auto* loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&scope);

    auto& assumptions_analysis = analysis_manager.get<AssumptionsAnalysis>();
    // For loops, query at the loop body so the induction variable's bounds are visible.
    auto& assumption_node = loop ? static_cast<structured_control_flow::ControlFlowNode&>(loop->root()) : scope;
    // Trivial-bounds view: includes type-derived defaults (e.g. Int32 ∈ [INT_MIN, INT_MAX]).
    // Used as the assumption set passed to symbolic::minimum/maximum so that the
    // resolver has sign information for parameters.
    auto& assumptions = assumptions_analysis.get(assumption_node, /*include_trivial_bounds=*/true);
    // Narrowing-only view: excludes type-derived defaults. A symbol that only
    // appears here (or in neither) has at most its type's intrinsic range — any
    // min/max resolution would collapse to INT_MIN/INT_MAX-style numerics, which
    // is not a sound tile bound. We use this to decide whether to emit a tile.
    auto& narrowing_assumptions = assumptions_analysis.get(assumption_node, /*include_trivial_bounds=*/false);
    // Parameters of a scope can only be constant symbols (invariant within the
    // scope). SDFG-level read-only arguments are constant by construction; for
    // each scope-local entry, the constant() flag tells us whether the symbol
    // can be treated opaquely by the min/max resolver.
    symbolic::SymbolSet parameters = assumptions_analysis.parameters();
    for (auto& entry : assumptions) {
        if (loop && symbolic::eq(entry.first, loop->indvar())) {
            continue; // The induction variable is not a parameter of its own loop scope
        }
        if (entry.second.constant()) {
            parameters.insert(entry.first);
        }
    }

    // Soundness check: every free (non-parameter) symbol in an index expression
    // must have a narrowing assumption at this scope. Otherwise symbolic::minimum/
    // maximum would fall back to the symbol's type-default range and produce
    // bogus tile bounds (e.g. INT_MAX) that the rest of the pipeline would
    // silently consume as truth.
    auto has_narrowing = [&](const symbolic::Symbol& sym) -> bool {
        auto it = narrowing_assumptions.find(sym);
        if (it == narrowing_assumptions.end()) return false;
        return !it->second.lower_bounds().empty() || !it->second.upper_bounds().empty();
    };
    auto bounds_are_sound = [&](const symbolic::Expression& expr) -> bool {
        for (const auto& sym : symbolic::atoms(expr)) {
            if (parameters.contains(sym)) continue;
            if (loop && symbolic::eq(sym, loop->indvar())) continue;
            if (!has_narrowing(sym)) return false;
        }
        return true;
    };

    // Find direct child scopes that may carry tiles for this scope
    std::set<const structured_control_flow::ControlFlowNode*> direct_child_scopes;
    collect_direct_child_scopes(scope, direct_child_scopes);

    for (auto& [container, memlets] : all_container_groups) {
        if (memlets.empty()) continue;

        // Find inner tiles from direct child scopes only
        std::vector<const MemoryTile*> inner_tiles;
        for (auto& [key, tile] : tiles_) {
            if (tiles_before.count(key) > 0) continue;
            if (key.second != container) continue;
            if (direct_child_scopes.count(key.first) == 0) continue;
            inner_tiles.push_back(&tile);
        }

        size_t ndims = 0;
        MemoryLayout reference_layout({symbolic::one()});
        // Separate min/max index lists to avoid unnecessary symbolic min/max
        std::vector<std::vector<symbolic::Expression>> min_indices;
        std::vector<std::vector<symbolic::Expression>> max_indices;

        if (!inner_tiles.empty()) {
            // Use inner tile min/max as representative values
            // Inner tiles have already resolved inner loop variables to their bounds
            ndims = inner_tiles[0]->min_subset.size();
            reference_layout = inner_tiles[0]->layout;
            min_indices.resize(ndims);
            max_indices.resize(ndims);

            for (const auto* tile : inner_tiles) {
                if (tile->min_subset.size() != ndims) continue;
                for (size_t d = 0; d < ndims; ++d) {
                    min_indices[d].push_back(tile->min_subset[d]);
                    max_indices[d].push_back(tile->max_subset[d]);
                }
            }

            // Propagate tile groups from child scopes upward using the same
            // base-partitioning logic: group inner groups by their min_subset
            // base at this scope level, then merge each partition.
            std::vector<const MemoryTileGroup*> inner_groups;
            for (auto& [key, groups] : tile_groups_) {
                if (tiles_before.count({key.first, key.second}) > 0) continue;
                if (key.second != container) continue;
                if (direct_child_scopes.count(key.first) == 0) continue;
                for (const auto& g : groups) {
                    inner_groups.push_back(&g);
                }
            }

            if (!inner_groups.empty()) {
                // Group inner groups by their base at this level
                struct OuterGroupEntry {
                    data_flow::Subset base;
                    std::vector<const MemoryTileGroup*> constituents;
                };
                std::vector<OuterGroupEntry> outer_partitions;

                for (const auto* ig : inner_groups) {
                    if (ig->tile.min_subset.size() != ndims) continue;

                    // Compute base: minimum of the inner group's min_subset per dim
                    data_flow::Subset base;
                    bool base_ok = true;
                    for (size_t d = 0; d < ndims; ++d) {
                        auto lb = symbolic::minimum(ig->tile.min_subset[d], parameters, assumptions, true);
                        if (lb.is_null()) {
                            lb = symbolic::minimum(ig->tile.min_subset[d], parameters, assumptions, false);
                        }
                        if (lb.is_null()) {
                            base_ok = false;
                            break;
                        }
                        base.push_back(symbolic::simplify(lb));
                    }
                    if (!base_ok) continue;

                    // Find matching partition (same base OR constant-offset base)
                    bool found = false;
                    for (auto& op : outer_partitions) {
                        bool const_diff = true;
                        for (size_t d = 0; d < ndims; ++d) {
                            auto diff = symbolic::simplify(symbolic::sub(base[d], op.base[d]));
                            if (!SymEngine::is_a<SymEngine::Integer>(*diff)) {
                                const_diff = false;
                                break;
                            }
                        }
                        if (const_diff) {
                            op.constituents.push_back(ig);
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        outer_partitions.push_back({base, {ig}});
                    }
                }

                // For each partition, merge constituent tile bounds and collect memlets
                std::vector<MemoryTileGroup> result_groups;
                for (auto& op : outer_partitions) {
                    data_flow::Subset grp_min, grp_max;
                    bool grp_bounded = true;

                    for (size_t d = 0; d < ndims; ++d) {
                        symbolic::Expression d_min = SymEngine::null;
                        symbolic::Expression d_max = SymEngine::null;

                        for (const auto* c : op.constituents) {
                            // min from min_subset
                            auto lb = symbolic::minimum(c->tile.min_subset[d], parameters, assumptions, true);
                            if (lb.is_null())
                                lb = symbolic::minimum(c->tile.min_subset[d], parameters, assumptions, false);
                            if (lb.is_null()) {
                                grp_bounded = false;
                                break;
                            }
                            d_min = d_min.is_null() ? lb : symbolic::min(d_min, lb);

                            // max from max_subset
                            auto ub = symbolic::maximum(c->tile.max_subset[d], parameters, assumptions, true);
                            if (ub.is_null())
                                ub = symbolic::maximum(c->tile.max_subset[d], parameters, assumptions, false);
                            if (ub.is_null()) {
                                grp_bounded = false;
                                break;
                            }
                            d_max = d_max.is_null() ? ub : symbolic::max(d_max, ub);
                        }
                        if (!grp_bounded) break;
                        grp_min.push_back(symbolic::simplify(d_min));
                        grp_max.push_back(symbolic::simplify(d_max));
                    }
                    if (!grp_bounded) continue;

                    // Collect all memlets from constituent groups
                    std::vector<const data_flow::Memlet*> grp_memlets;
                    for (const auto* c : op.constituents) {
                        grp_memlets.insert(grp_memlets.end(), c->memlets.begin(), c->memlets.end());
                    }

                    MemoryTile grp_tile{
                        container, grp_min, grp_max, reference_layout, !layout_has_unbounded_first_dim(reference_layout)
                    };
                    result_groups.push_back({grp_tile, std::move(grp_memlets)});
                }

                if (!result_groups.empty()) {
                    tile_groups_.insert({{&scope, container}, std::move(result_groups)});
                }
            }
        } else {
            // Use raw access indices (no inner tiles available)
            auto& first_access = accesses_.at(memlets[0]);
            auto& reference_shape = first_access.layout.shape();
            ndims = reference_shape.size();
            reference_layout = first_access.layout;
            min_indices.resize(ndims);
            max_indices.resize(ndims);

            bool consistent = true;
            for (const auto* memlet_ptr : memlets) {
                auto& acc = accesses_.at(memlet_ptr);
                auto& shape = acc.layout.shape();

                if (shape.size() != ndims) {
                    consistent = false;
                    break;
                }
                // Check inner dimensions match (all except first which may be unbounded)
                for (size_t d = 1; d < ndims; ++d) {
                    if (!symbolic::eq(shape[d], reference_shape[d])) {
                        consistent = false;
                        break;
                    }
                }
                if (!consistent) break;

                // Collect indices for each dimension
                if (acc.subset.size() != ndims) {
                    consistent = false;
                    break;
                }
                for (size_t d = 0; d < ndims; ++d) {
                    min_indices[d].push_back(acc.subset[d]);
                    max_indices[d].push_back(acc.subset[d]);
                }
            }

            if (!consistent) continue;

            // Compute tile groups for raw memlets
            compute_tile_groups(scope, container, memlets, reference_layout, ndims, parameters, assumptions);
        }

        if (ndims == 0) continue;

        // Compute min/max bounds for each dimension
        data_flow::Subset min_subset;
        data_flow::Subset max_subset;
        bool all_bounded = true;

        for (size_t d = 0; d < ndims; ++d) {
            symbolic::Expression dim_min = SymEngine::null;
            symbolic::Expression dim_max = SymEngine::null;

            // Compute dim_min from min_indices
            for (const auto& idx : min_indices[d]) {
                if (!bounds_are_sound(idx)) {
                    all_bounded = false;
                    break;
                }
                auto lb = symbolic::minimum(idx, parameters, assumptions, true);
                if (lb.is_null()) {
                    lb = symbolic::minimum(idx, parameters, assumptions, false);
                }
                if (lb.is_null()) {
                    all_bounded = false;
                    break;
                }
                if (dim_min.is_null()) {
                    dim_min = lb;
                } else {
                    dim_min = symbolic::min(dim_min, lb);
                }
            }
            if (!all_bounded) break;

            // Compute dim_max from max_indices
            for (const auto& idx : max_indices[d]) {
                if (!bounds_are_sound(idx)) {
                    all_bounded = false;
                    break;
                }
                auto ub = symbolic::maximum(idx, parameters, assumptions, true);
                if (ub.is_null()) {
                    ub = symbolic::maximum(idx, parameters, assumptions, false);
                }
                if (ub.is_null()) {
                    all_bounded = false;
                    break;
                }
                if (dim_max.is_null()) {
                    dim_max = ub;
                } else {
                    dim_max = symbolic::max(dim_max, ub);
                }
            }
            if (!all_bounded) break;

            min_subset.push_back(symbolic::simplify(dim_min));
            max_subset.push_back(symbolic::simplify(dim_max));
        }

        if (!all_bounded) continue;

        // Store this scope's tile with the original memory layout. `first_dim_bounded`
        // mirrors the underlying layout: false whenever shape[0] is the unbounded sentinel.
        MemoryTile merged_tile{
            container, min_subset, max_subset, reference_layout, !layout_has_unbounded_first_dim(reference_layout)
        };
        tiles_.insert({{&scope, container}, merged_tile});
    }
}

const MemoryTile* MemoryLayoutAnalysis::
    tile(const structured_control_flow::ControlFlowNode& scope, const std::string& container) const {
    auto key = std::make_pair(&scope, container);
    auto it = tiles_.find(key);
    if (it == tiles_.end()) {
        return nullptr;
    }
    return &it->second;
}

void MemoryLayoutAnalysis::compute_tile_groups(
    structured_control_flow::ControlFlowNode& scope,
    const std::string& container,
    const std::vector<const data_flow::Memlet*>& memlets,
    const MemoryLayout& reference_layout,
    size_t ndims,
    const symbolic::SymbolSet& parameters,
    const symbolic::Assumptions& assumptions
) {
    // For each memlet, compute per-dimension base (minimum of index expression)
    // Group memlets whose bases are symbolically equal in all dimensions
    struct GroupEntry {
        data_flow::Subset base; // per-dim minimum
        std::vector<const data_flow::Memlet*> group_memlets;
    };

    std::vector<GroupEntry> groups;

    for (const auto* memlet_ptr : memlets) {
        auto& acc = accesses_.at(memlet_ptr);
        if (acc.subset.size() != ndims) continue;

        // Compute per-dimension base (minimum)
        data_flow::Subset base;
        bool base_ok = true;
        for (size_t d = 0; d < ndims; ++d) {
            auto lb = symbolic::minimum(acc.subset[d], parameters, assumptions, true);
            if (lb.is_null()) {
                lb = symbolic::minimum(acc.subset[d], parameters, assumptions, false);
            }
            if (lb.is_null()) {
                base_ok = false;
                break;
            }
            base.push_back(symbolic::simplify(lb));
        }
        if (!base_ok) continue;

        // Find existing group with same base
        bool found = false;
        for (auto& group : groups) {
            if (group.base.size() != ndims) continue;
            bool match = true;
            for (size_t d = 0; d < ndims; ++d) {
                if (!symbolic::eq(group.base[d], base[d])) {
                    match = false;
                    break;
                }
            }
            if (match) {
                group.group_memlets.push_back(memlet_ptr);
                found = true;
                break;
            }
        }
        if (!found) {
            groups.push_back({base, {memlet_ptr}});
        }
    }

    if (groups.empty()) return;

    // Merge groups whose bases differ only by integer constants.
    // E.g. stencil bases [i-1, j], [i, j], [i+1, j] should merge (constant offsets in dim0).
    // But SYR2K bases [i, 0] vs [j, 0] should NOT merge (symbolic difference).
    std::vector<GroupEntry> merged_groups;
    for (auto& group : groups) {
        bool merged = false;
        for (auto& existing : merged_groups) {
            bool const_diff = true;
            for (size_t d = 0; d < ndims; ++d) {
                auto diff = symbolic::simplify(symbolic::sub(group.base[d], existing.base[d]));
                if (!SymEngine::is_a<SymEngine::Integer>(*diff)) {
                    const_diff = false;
                    break;
                }
            }
            if (const_diff) {
                existing.group_memlets
                    .insert(existing.group_memlets.end(), group.group_memlets.begin(), group.group_memlets.end());
                merged = true;
                break;
            }
        }
        if (!merged) {
            merged_groups.push_back(std::move(group));
        }
    }

    // Compute tile for each merged group
    std::vector<MemoryTileGroup> result_groups;
    for (auto& group : merged_groups) {
        std::vector<std::vector<symbolic::Expression>> min_indices(ndims);
        std::vector<std::vector<symbolic::Expression>> max_indices(ndims);

        for (const auto* memlet_ptr : group.group_memlets) {
            auto& acc = accesses_.at(memlet_ptr);
            for (size_t d = 0; d < ndims; ++d) {
                min_indices[d].push_back(acc.subset[d]);
                max_indices[d].push_back(acc.subset[d]);
            }
        }

        data_flow::Subset min_subset;
        data_flow::Subset max_subset;
        bool all_bounded = true;

        for (size_t d = 0; d < ndims; ++d) {
            symbolic::Expression dim_min = SymEngine::null;
            symbolic::Expression dim_max = SymEngine::null;

            for (const auto& idx : min_indices[d]) {
                auto lb = symbolic::minimum(idx, parameters, assumptions, true);
                if (lb.is_null()) {
                    lb = symbolic::minimum(idx, parameters, assumptions, false);
                }
                if (lb.is_null()) {
                    all_bounded = false;
                    break;
                }
                if (dim_min.is_null()) {
                    dim_min = lb;
                } else {
                    dim_min = symbolic::min(dim_min, lb);
                }
            }
            if (!all_bounded) break;

            for (const auto& idx : max_indices[d]) {
                auto ub = symbolic::maximum(idx, parameters, assumptions, true);
                if (ub.is_null()) {
                    ub = symbolic::maximum(idx, parameters, assumptions, false);
                }
                if (ub.is_null()) {
                    all_bounded = false;
                    break;
                }
                if (dim_max.is_null()) {
                    dim_max = ub;
                } else {
                    dim_max = symbolic::max(dim_max, ub);
                }
            }
            if (!all_bounded) break;

            min_subset.push_back(symbolic::simplify(dim_min));
            max_subset.push_back(symbolic::simplify(dim_max));
        }

        if (!all_bounded) continue;

        MemoryTile tile{
            container, min_subset, max_subset, reference_layout, !layout_has_unbounded_first_dim(reference_layout)
        };
        result_groups.push_back({tile, group.group_memlets});
    }

    if (!result_groups.empty()) {
        tile_groups_.insert({{&scope, container}, std::move(result_groups)});
    }
}

const std::vector<MemoryTileGroup>* MemoryLayoutAnalysis::
    tile_groups(const structured_control_flow::ControlFlowNode& scope, const std::string& container) const {
    auto key = std::make_pair(&scope, container);
    auto it = tile_groups_.find(key);
    if (it == tile_groups_.end()) {
        return nullptr;
    }
    return &it->second;
}

const MemoryTileGroup* MemoryLayoutAnalysis::
    tile_group_for(const structured_control_flow::ControlFlowNode& scope, const data_flow::Memlet& memlet) const {
    // Find which container this memlet accesses
    auto acc_it = accesses_.find(&memlet);
    if (acc_it == accesses_.end()) {
        return nullptr;
    }
    auto& container = acc_it->second.container;

    auto key = std::make_pair(&scope, container);
    auto groups_it = tile_groups_.find(key);
    if (groups_it == tile_groups_.end()) {
        return nullptr;
    }

    for (const auto& group : groups_it->second) {
        for (const auto* m : group.memlets) {
            if (m == &memlet) {
                return &group;
            }
        }
    }
    return nullptr;
}

symbolic::MultiExpression MemoryTile::extents() const {
    symbolic::MultiExpression result;
    for (size_t d = 0; d < min_subset.size(); ++d) {
        auto ext =
            symbolic::simplify(symbolic::expand(symbolic::add(symbolic::sub(max_subset[d], min_subset[d]), symbolic::one())
            ));
        // Defensive: subset values are always proven-bounded, so this should never trigger
        // for row-major layouts. Guards future custom layouts whose subsets could pick up
        // the unbounded sentinel.
        if (depends_on_unbounded(ext)) {
            result.push_back(SymEngine::null);
        } else {
            result.push_back(ext);
        }
    }
    return result;
}

symbolic::MultiExpression MemoryTile::extents_approx() const {
    symbolic::MultiExpression result;
    for (size_t d = 0; d < min_subset.size(); ++d) {
        auto ext = symbolic::simplify(symbolic::expand(
            symbolic::overapproximate(symbolic::add(symbolic::sub(max_subset[d], min_subset[d]), symbolic::one()))
        ));
        if (depends_on_unbounded(ext)) {
            result.push_back(SymEngine::null);
        } else {
            result.push_back(ext);
        }
    }
    return result;
}

std::pair<symbolic::Expression, symbolic::Expression> MemoryTile::contiguous_range() const {
    auto& strides = layout.strides();
    auto first = layout.offset();
    auto last = layout.offset();
    for (size_t d = 0; d < min_subset.size(); ++d) {
        first = symbolic::add(first, symbolic::mul(strides[d], min_subset[d]));
        last = symbolic::add(last, symbolic::mul(strides[d], max_subset[d]));
    }
    first = symbolic::simplify(symbolic::expand(first));
    last = symbolic::simplify(symbolic::expand(last));
    // If either endpoint references the unbounded sentinel, the linear range is undefined
    // (e.g. a non-row-major layout whose stride references shape[0]). Report as unknown
    // rather than leaking the sentinel symbol to callers.
    if (depends_on_unbounded(first) || depends_on_unbounded(last)) {
        return {SymEngine::null, SymEngine::null};
    }
    return {first, last};
}

} // namespace analysis
} // namespace sdfg
