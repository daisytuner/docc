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
// Collect StructuredLoop nodes that are direct children of the given node,
// stopping at loop boundaries (does not recurse into nested loops).
void collect_direct_child_loops(
    structured_control_flow::ControlFlowNode& node, std::set<const structured_control_flow::StructuredLoop*>& result
) {
    if (auto* loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&node)) {
        result.insert(loop);
        return;
    }
    if (auto* seq = dynamic_cast<structured_control_flow::Sequence*>(&node)) {
        for (size_t i = 0; i < seq->size(); i++) {
            collect_direct_child_loops(seq->at(i).first, result);
        }
    } else if (auto* ife = dynamic_cast<structured_control_flow::IfElse*>(&node)) {
        for (size_t i = 0; i < ife->size(); i++) {
            collect_direct_child_loops(ife->at(i).first, result);
        }
    } else if (auto* w = dynamic_cast<structured_control_flow::While*>(&node)) {
        collect_direct_child_loops(w->root(), result);
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
        // Snapshot current memlets before traversing loop body
        std::vector<const data_flow::Memlet*> memlets_before;
        memlets_before.reserve(accesses_.size());
        for (const auto& entry : accesses_) {
            memlets_before.push_back(entry.first);
        }

        // Snapshot tile keys before traversal
        std::set<std::pair<const structured_control_flow::StructuredLoop*, std::string>> tiles_before;
        for (const auto& entry : tiles_) {
            tiles_before.insert(entry.first);
        }

        traverse(loop->root(), analysis_manager);

        // Merge layouts for containers accessed within this loop
        merge_loop_layouts(*loop, memlets_before, tiles_before, analysis_manager);
    }
    // Break, Continue, Return nodes don't contain blocks
}

void MemoryLayoutAnalysis::
    process_block(structured_control_flow::Block& block, analysis::AnalysisManager& analysis_manager) {
    auto& assumptions_analysis = analysis_manager.get<AssumptionsAnalysis>();
    auto& assumptions = assumptions_analysis.get(block);

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

void MemoryLayoutAnalysis::merge_loop_layouts(
    structured_control_flow::StructuredLoop& loop,
    const std::vector<const data_flow::Memlet*>& memlets_before,
    const std::set<std::pair<const structured_control_flow::StructuredLoop*, std::string>>& tiles_before,
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

    auto& assumptions_analysis = analysis_manager.get<AssumptionsAnalysis>();
    auto& assumptions = assumptions_analysis.get(loop.root());
    // Start with SDFG-level parameters (read-only arguments like N, M)
    // then add any additional constant symbols from loop assumptions
    symbolic::SymbolSet parameters = assumptions_analysis.parameters();
    for (auto& entry : assumptions) {
        if (symbolic::eq(entry.first, loop.indvar())) {
            continue; // Skip induction variable itself
        }

        if (entry.second.constant()) {
            parameters.insert(entry.first);
        }
    }

    // Find direct child loops of this loop (not grandchildren)
    std::set<const structured_control_flow::StructuredLoop*> direct_child_loops;
    collect_direct_child_loops(loop.root(), direct_child_loops);

    for (auto& [container, memlets] : all_container_groups) {
        if (memlets.empty()) continue;

        // Find inner tiles from direct child loops only
        std::vector<const MemoryTile*> inner_tiles;
        for (auto& [key, tile] : tiles_) {
            if (tiles_before.count(key) > 0) continue;
            if (key.second != container) continue;
            if (direct_child_loops.count(key.first) == 0) continue;
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

            // Propagate tile groups from child loops upward using the same
            // base-partitioning logic: group inner groups by their min_subset
            // base at this loop level, then merge each partition.
            std::vector<const MemoryTileGroup*> inner_groups;
            for (auto& [key, groups] : tile_groups_) {
                if (tiles_before.count({key.first, key.second}) > 0) continue;
                if (key.second != container) continue;
                if (direct_child_loops.count(key.first) == 0) continue;
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

                    MemoryTile grp_tile{container, grp_min, grp_max, reference_layout, true};
                    result_groups.push_back({grp_tile, std::move(grp_memlets)});
                }

                if (!result_groups.empty()) {
                    tile_groups_.insert({{&loop, container}, std::move(result_groups)});
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
            compute_tile_groups(loop, container, memlets, reference_layout, ndims, parameters, assumptions);
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

        // Store this loop's tile with the original memory layout
        MemoryTile merged_tile{container, min_subset, max_subset, reference_layout, true};
        tiles_.insert({{&loop, container}, merged_tile});
    }
}

const MemoryTile* MemoryLayoutAnalysis::
    tile(const structured_control_flow::StructuredLoop& loop, const std::string& container) const {
    auto key = std::make_pair(&loop, container);
    auto it = tiles_.find(key);
    if (it == tiles_.end()) {
        return nullptr;
    }
    return &it->second;
}

void MemoryLayoutAnalysis::compute_tile_groups(
    structured_control_flow::StructuredLoop& loop,
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

        MemoryTile tile{container, min_subset, max_subset, reference_layout, true};
        result_groups.push_back({tile, group.group_memlets});
    }

    if (!result_groups.empty()) {
        tile_groups_.insert({{&loop, container}, std::move(result_groups)});
    }
}

const std::vector<MemoryTileGroup>* MemoryLayoutAnalysis::
    tile_groups(const structured_control_flow::StructuredLoop& loop, const std::string& container) const {
    auto key = std::make_pair(&loop, container);
    auto it = tile_groups_.find(key);
    if (it == tile_groups_.end()) {
        return nullptr;
    }
    return &it->second;
}

const MemoryTileGroup* MemoryLayoutAnalysis::
    tile_group_for(const structured_control_flow::StructuredLoop& loop, const data_flow::Memlet& memlet) const {
    // Find which container this memlet accesses
    auto acc_it = accesses_.find(&memlet);
    if (acc_it == accesses_.end()) {
        return nullptr;
    }
    auto& container = acc_it->second.container;

    auto key = std::make_pair(&loop, container);
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
        result.push_back(symbolic::simplify(
            symbolic::expand(symbolic::add(symbolic::sub(max_subset[d], min_subset[d]), symbolic::one()))
        ));
    }
    return result;
}

symbolic::MultiExpression MemoryTile::extents_approx() const {
    symbolic::MultiExpression result;
    for (size_t d = 0; d < min_subset.size(); ++d) {
        result.push_back(symbolic::simplify(symbolic::expand(
            symbolic::overapproximate(symbolic::add(symbolic::sub(max_subset[d], min_subset[d]), symbolic::one()))
        )));
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
    return {symbolic::simplify(symbolic::expand(first)), symbolic::simplify(symbolic::expand(last))};
}

} // namespace analysis
} // namespace sdfg
