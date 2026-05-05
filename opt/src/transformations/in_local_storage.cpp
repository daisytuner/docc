#include "sdfg/transformations/in_local_storage.h"

#include <cstddef>
#include <functional>
#include <string>

#include "sdfg/analysis/memory_layout_analysis.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_nodes/barrier_local_node.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/passes/structured_control_flow/dead_cfg_elimination.h"
#include "sdfg/passes/structured_control_flow/sequence_fusion.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/gpu/gpu_schedule_type.h"
#include "sdfg/types/array.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"

namespace sdfg {
namespace transformations {

InLocalStorage::InLocalStorage(
    structured_control_flow::StructuredLoop& loop,
    const data_flow::AccessNode& access_node,
    const types::StorageType& storage_type
)
    : loop_(loop), access_node_(access_node), container_(access_node.data()), storage_type_(storage_type) {}

std::string InLocalStorage::name() const { return "InLocalStorage"; }

bool InLocalStorage::can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& sdfg = builder.subject();
    auto& body = this->loop_.root();

    tile_info_ = TileInfo{};

    // Criterion: Container must exist
    if (!sdfg.exists(this->container_)) {
        return false;
    }

    auto& type = sdfg.type(this->container_);

    // Criterion: Container must be Array or Pointer (not Scalar)
    if (type.type_id() != types::TypeID::Pointer && type.type_id() != types::TypeID::Array) {
        return false;
    }

    // Criterion: Container must be used in the loop body
    auto& users = analysis_manager.get<analysis::Users>();
    analysis::UsersView body_users(users, body);
    if (body_users.uses(this->container_).empty()) {
        return false;
    }

    // Criterion: Container must be read-only within the loop (no writes)
    if (!body_users.writes(this->container_).empty()) {
        return false;
    }

    // Use MemoryLayoutAnalysis tile group API
    auto& mla = analysis_manager.get<analysis::MemoryLayoutAnalysis>();

    // Find a representative memlet from the access node to identify its group.
    // An access node may have multiple out-edges belonging to different tile
    // groups (e.g., A[i,k] and A[j,k]).  We iterate all out-edges and select
    // the first one whose tile group has provably integer extents (i.e., can
    // actually be applied).  This avoids picking a group with symbolic extents
    // when another group on the same node would succeed.
    // For GPU shared memory, extents may be symbolic until GPU block size
    // substitution, so we accept the first valid group unconditionally.
    const analysis::MemoryTileGroup* group = nullptr;
    auto& dfg = access_node_.get_parent();
    for (auto& memlet : dfg.out_edges(access_node_)) {
        auto* candidate = mla.tile_group_for(loop_, memlet);
        if (!candidate) continue;

        auto extents = candidate->tile.extents_approx();
        if (extents.empty()) continue;

        if (storage_type_.is_nv_shared()) {
            // GPU path: accept first valid group (substitution happens later)
            group = candidate;
            break;
        }

        // CPU path: require provably integer extents
        bool all_integer = true;
        for (auto& ext : extents) {
            if (!SymEngine::is_a<SymEngine::Integer>(*ext)) {
                all_integer = false;
                break;
            }
        }
        if (all_integer) {
            group = candidate;
            break;
        }
    }
    if (!group) {
        return false;
    }

    auto& tile = group->tile;

    // Store group memlets for use in apply()
    group_memlets_.clear();
    group_memlets_.insert(group->memlets.begin(), group->memlets.end());

    // Get overapproximated extents (integer upper bounds)
    auto extents = tile.extents_approx();
    if (extents.empty()) {
        return false;
    }

    // Store tile info (before substitution, bases/strides stay symbolic)
    tile_info_.dimensions = extents;
    tile_info_.bases = tile.min_subset;
    tile_info_.strides = std::vector<symbolic::Expression>(tile.layout.strides().begin(), tile.layout.strides().end());
    tile_info_.offset = tile.layout.offset();

    // GPU shared memory: resolve symbolic extents using GPU block sizes and
    // require at least one cooperative dimension
    if (storage_type_.is_nv_shared()) {
        auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
        auto ancestors = scope_analysis.ancestor_scopes(&loop_);

        // Build substitution map: symbolic GPU map bounds → integer block sizes
        // E.g., Map condition "i < N" with block_size=32 → N=32
        for (auto* node : ancestors) {
            if (auto* ancestor_map = dynamic_cast<structured_control_flow::Map*>(node)) {
                if (!gpu::is_gpu_schedule(ancestor_map->schedule_type())) {
                    continue;
                }
                auto block_size = gpu::gpu_block_size(ancestor_map->schedule_type());
                // Extract symbolic bound from condition: Lt(indvar, BOUND)
                auto condition = ancestor_map->condition();
                if (SymEngine::is_a<SymEngine::StrictLessThan>(*condition)) {
                    auto stl = SymEngine::rcp_static_cast<const SymEngine::StrictLessThan>(condition);
                    auto rhs = stl->get_args()[1];
                    auto iter_count = symbolic::sub(rhs, ancestor_map->init());
                    if (!SymEngine::is_a<SymEngine::Integer>(*iter_count)) {
                        // Symbolic bound — substitute with block size in extents and bases
                        for (auto& ext : tile_info_.dimensions) {
                            ext = symbolic::simplify(symbolic::subs(ext, iter_count, block_size));
                        }
                        for (auto& base : tile_info_.bases) {
                            base = symbolic::simplify(symbolic::subs(base, iter_count, block_size));
                        }
                    }
                }
            }
        }

        // Also resolve the loop's own bound if symbolic and matches a block size
        // E.g., For k = 0..K where K is a parameter — check if K can be resolved
        // from any GPU ancestor map
        // (Already handled above: if K appears as a GPU map bound, it's substituted)

        // Criterion: All extents must now be provably integer
        for (auto& ext : tile_info_.dimensions) {
            if (!SymEngine::is_a<SymEngine::Integer>(*ext)) {
                return false;
            }
        }

        // Criterion: At least one cooperative dimension
        bool has_cooperative_dim = false;
        for (auto* node : ancestors) {
            if (auto* ancestor_map = dynamic_cast<structured_control_flow::Map*>(node)) {
                if (!gpu::is_gpu_schedule(ancestor_map->schedule_type())) {
                    continue;
                }
                // A GPU dim is cooperative if its indvar does NOT appear in any tile base
                bool appears_in_bases = false;
                for (auto& base : tile_info_.bases) {
                    if (symbolic::uses(base, ancestor_map->indvar())) {
                        appears_in_bases = true;
                        break;
                    }
                }
                if (!appears_in_bases) {
                    has_cooperative_dim = true;
                    break;
                }
            }
        }
        if (!has_cooperative_dim) {
            return false;
        }
    } else {
        // CPU path: All extents must be provably integer
        for (auto& ext : tile_info_.dimensions) {
            if (!SymEngine::is_a<SymEngine::Integer>(*ext)) {
                return false;
            }
        }
    }

    return true;
}

void InLocalStorage::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& sdfg = builder.subject();
    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();

    auto parent_node = scope_analysis.parent_scope(&loop_);
    auto parent = dynamic_cast<structured_control_flow::Sequence*>(parent_node);
    if (!parent) {
        throw InvalidSDFGException("InLocalStorage: Parent of loop must be a Sequence!");
    }

    // Get type information
    auto& type = sdfg.type(this->container_);
    types::Scalar scalar_type(type.primitive_type());

    // Create local buffer name
    local_name_ = builder.find_new_name("__daisy_in_local_storage_" + this->container_);

    // Collect varying dimensions (extent > 1) and compute buffer layout
    std::vector<size_t> varying_dims;
    std::vector<symbolic::Expression> dim_sizes;
    for (size_t d = 0; d < tile_info_.dimensions.size(); d++) {
        auto& dim_size = tile_info_.dimensions.at(d);
        if (!symbolic::eq(dim_size, symbolic::integer(1))) {
            varying_dims.push_back(d);
            dim_sizes.push_back(dim_size);
        }
    }

    // Compute total buffer size
    symbolic::Expression total_size = symbolic::integer(1);
    for (auto& ds : dim_sizes) {
        total_size = symbolic::mul(total_size, ds);
    }

    // Helper: build linearized local index from per-dimension symbolic expressions
    auto linearize_exprs = [&](const std::vector<symbolic::Expression>& indices) -> symbolic::Expression {
        symbolic::Expression linear_idx = symbolic::integer(0);
        symbolic::Expression stride = symbolic::integer(1);
        for (int i = indices.size() - 1; i >= 0; i--) {
            linear_idx = symbolic::add(linear_idx, symbolic::mul(indices[i], stride));
            stride = symbolic::mul(stride, dim_sizes[i]);
        }
        return linear_idx;
    };

    // Helper: build linearized local index from per-dimension indvars (symbols)
    auto linearize = [&](const std::vector<symbolic::Symbol>& indvars) -> symbolic::Expression {
        std::vector<symbolic::Expression> exprs(indvars.begin(), indvars.end());
        return linearize_exprs(exprs);
    };

    // Helper: build source subset (base[d] + copy_indvar[d]) for original container
    bool is_pointer = (type.type_id() == types::TypeID::Pointer);
    auto build_original_subset = [&](const std::vector<symbolic::Expression>& copy_indices) -> data_flow::Subset {
        std::vector<symbolic::Expression> full_indices;
        size_t var_idx = 0;
        for (size_t d = 0; d < tile_info_.dimensions.size(); d++) {
            if (!symbolic::eq(tile_info_.dimensions.at(d), symbolic::integer(1))) {
                full_indices.push_back(symbolic::add(tile_info_.bases.at(d), copy_indices.at(var_idx++)));
            } else {
                full_indices.push_back(tile_info_.bases.at(d));
            }
        }

        if (is_pointer) {
            symbolic::Expression linear = tile_info_.offset;
            for (size_t d = 0; d < full_indices.size(); d++) {
                linear = symbolic::add(linear, symbolic::mul(tile_info_.strides.at(d), full_indices.at(d)));
            }
            return {linear};
        } else {
            return data_flow::Subset(full_indices.begin(), full_indices.end());
        }
    };

    // ==================================================================
    // Branch: GPU cooperative path vs CPU sequential path
    // ==================================================================
    if (storage_type_.is_nv_shared()) {
        // ============================================================
        // GPU COOPERATIVE PATH
        // ============================================================
        auto ancestors = scope_analysis.ancestor_scopes(&loop_);

        // Collect cooperative GPU dimensions (indvar not in tile bases)
        struct CoopDim {
            symbolic::Symbol indvar;
            symbolic::Integer block_size;
            gpu::GPUDimension dimension;
        };
        std::vector<CoopDim> coop_dims;

        for (auto* node : ancestors) {
            if (auto* ancestor_map = dynamic_cast<structured_control_flow::Map*>(node)) {
                if (!gpu::is_gpu_schedule(ancestor_map->schedule_type())) {
                    continue;
                }
                bool appears_in_bases = false;
                for (auto& base : tile_info_.bases) {
                    if (symbolic::uses(base, ancestor_map->indvar())) {
                        appears_in_bases = true;
                        break;
                    }
                }
                if (!appears_in_bases) {
                    coop_dims.push_back(
                        {ancestor_map->indvar(),
                         gpu::gpu_block_size(ancestor_map->schedule_type()),
                         gpu::gpu_dimension(ancestor_map->schedule_type())}
                    );
                }
            }
        }

        // Compute total cooperative thread count
        symbolic::Expression total_coop_threads = symbolic::integer(1);
        for (auto& cd : coop_dims) {
            total_coop_threads = symbolic::mul(total_coop_threads, cd.block_size);
        }

        // Create the local buffer with NV_Shared storage
        types::Array buffer_type(storage_type_, 0, {}, scalar_type, total_size);
        builder.add_container(local_name_, buffer_type);

        // Emit: barrier → guarded cooperative copy → barrier → loop
        // 1. Barrier before copy
        auto& barrier_block1 = builder.add_block_before(*parent, loop_, {}, loop_.debug_info());
        builder.add_library_node<data_flow::BarrierLocalNode>(barrier_block1, {});

        // 2. Cooperative copy with if_else guard
        // Flatten cooperative thread index: coop_flat = sum(indvar[i] * product(block_size[j] for j>i))
        symbolic::Expression coop_flat = symbolic::integer(0);
        symbolic::Expression coop_stride = symbolic::integer(1);
        for (int i = coop_dims.size() - 1; i >= 0; i--) {
            coop_flat = symbolic::add(coop_flat, symbolic::mul(coop_dims[i].indvar, coop_stride));
            coop_stride = symbolic::mul(coop_stride, coop_dims[i].block_size);
        }

        // Each thread loads elements strided by total_coop_threads
        // Thread t loads elements: t, t + total_threads, t + 2*total_threads, ...
        // We emit a loop: for (idx = coop_flat; idx < total_size; idx += total_coop_threads)
        auto idx_name = builder.find_new_name("__daisy_ils_coop_" + this->container_);
        types::Scalar idx_type(types::PrimitiveType::UInt64);
        builder.add_container(idx_name, idx_type);
        auto idx_var = symbolic::symbol(idx_name);

        auto copy_init = coop_flat;
        auto copy_condition = symbolic::Lt(idx_var, total_size);
        auto copy_update = symbolic::add(idx_var, total_coop_threads);

        auto& copy_loop = builder.add_map_before(
            *parent,
            loop_,
            idx_var,
            copy_condition,
            copy_init,
            copy_update,
            structured_control_flow::ScheduleType_Sequential::create(),
            {},
            loop_.debug_info()
        );

        // Decompose flat idx back into per-dimension indices for source subset
        // idx maps to varying_dims in row-major order
        auto& copy_scope = copy_loop.root();
        auto& copy_block = builder.add_block(copy_scope);
        auto& copy_src = builder.add_access(copy_block, this->container_);
        auto& copy_dst = builder.add_access(copy_block, local_name_);
        auto& copy_tasklet = builder.add_tasklet(copy_block, data_flow::TaskletCode::assign, "_out", {"_in"});

        // Decompose idx_var into per-dim indices
        std::vector<symbolic::Expression> copy_indices;
        symbolic::Expression remainder = idx_var;
        for (size_t i = 0; i < dim_sizes.size(); i++) {
            if (i < dim_sizes.size() - 1) {
                // integer division: idx / (product of remaining dims)
                symbolic::Expression divisor = symbolic::integer(1);
                for (size_t j = i + 1; j < dim_sizes.size(); j++) {
                    divisor = symbolic::mul(divisor, dim_sizes[j]);
                }
                auto quotient = symbolic::div(remainder, divisor);
                copy_indices.push_back(quotient);
                remainder = symbolic::mod(remainder, divisor);
            } else {
                copy_indices.push_back(remainder);
            }
        }

        auto copy_src_subset = build_original_subset(copy_indices);
        data_flow::Subset copy_dst_subset = {idx_var};

        builder.add_computational_memlet(copy_block, copy_src, copy_tasklet, "_in", copy_src_subset, type);
        builder.add_computational_memlet(copy_block, copy_tasklet, "_out", copy_dst, copy_dst_subset, buffer_type);

        // 3. Barrier after copy
        auto& barrier_block2 = builder.add_block_before(*parent, loop_, {}, loop_.debug_info());
        builder.add_library_node<data_flow::BarrierLocalNode>(barrier_block2, {});
    } else {
        // ============================================================
        // CPU SEQUENTIAL PATH
        // ============================================================
        // Create the local buffer with specified storage type
        types::Array buffer_type(storage_type_, 0, {}, scalar_type, total_size);
        builder.add_container(local_name_, buffer_type);

        std::vector<symbolic::Symbol> copy_indvars;
        structured_control_flow::Sequence* copy_scope = parent;
        bool first_copy_loop = true;

        for (size_t i = 0; i < varying_dims.size(); i++) {
            size_t d = varying_dims[i];
            auto indvar_name = builder.find_new_name("__daisy_ils_" + this->container_ + "_d" + std::to_string(d));
            types::Scalar indvar_type(types::PrimitiveType::UInt64);
            builder.add_container(indvar_name, indvar_type);
            auto indvar = symbolic::symbol(indvar_name);
            copy_indvars.push_back(indvar);

            auto init = symbolic::integer(0);
            auto condition = symbolic::Lt(indvar, dim_sizes[i]);
            auto update = symbolic::add(indvar, symbolic::integer(1));

            if (first_copy_loop) {
                auto& copy_loop = builder.add_map_before(
                    *copy_scope,
                    loop_,
                    indvar,
                    condition,
                    init,
                    update,
                    structured_control_flow::ScheduleType_Sequential::create(),
                    {},
                    loop_.debug_info()
                );
                copy_scope = &copy_loop.root();
                first_copy_loop = false;
            } else {
                auto& copy_loop = builder.add_map(
                    *copy_scope,
                    indvar,
                    condition,
                    init,
                    update,
                    structured_control_flow::ScheduleType_Sequential::create(),
                    {},
                    loop_.debug_info()
                );
                copy_scope = &copy_loop.root();
            }
        }

        // Create copy block
        auto& copy_block = builder.add_block(*copy_scope);
        auto& copy_src = builder.add_access(copy_block, this->container_);
        auto& copy_dst = builder.add_access(copy_block, local_name_);
        auto& copy_tasklet = builder.add_tasklet(copy_block, data_flow::TaskletCode::assign, "_out", {"_in"});

        std::vector<symbolic::Expression> copy_exprs(copy_indvars.begin(), copy_indvars.end());
        auto copy_src_subset = build_original_subset(copy_exprs);
        data_flow::Subset copy_dst_subset = {linearize(copy_indvars)};

        builder.add_computational_memlet(copy_block, copy_src, copy_tasklet, "_in", copy_src_subset, type);
        types::Array buffer_type_ref(storage_type_, 0, {}, scalar_type, total_size);
        builder.add_computational_memlet(copy_block, copy_tasklet, "_out", copy_dst, copy_dst_subset, buffer_type_ref);
    }

    // ==================================================================
    // Update accesses in the main loop to use the local buffer
    // ==================================================================
    types::Array buffer_type(storage_type_, 0, {}, scalar_type, total_size);
    auto& mla = analysis_manager.get<analysis::MemoryLayoutAnalysis>();

    // Recursive helper to traverse all blocks in the loop body
    std::function<void(structured_control_flow::ControlFlowNode&)> rewrite_accesses;
    rewrite_accesses = [&](structured_control_flow::ControlFlowNode& node) {
        if (auto* block = dynamic_cast<structured_control_flow::Block*>(&node)) {
            auto& dfg = block->dataflow();

            // Collect access nodes to process (avoid iterator invalidation)
            std::vector<data_flow::AccessNode*> access_nodes;
            for (auto* access_node : dfg.data_nodes()) {
                if (access_node->data() == this->container_) {
                    access_nodes.push_back(access_node);
                }
            }

            for (auto* access : access_nodes) {
                // Classify memlets: group vs non-group
                struct MemletRewrite {
                    data_flow::Memlet* memlet;
                    data_flow::Subset local_subset;
                    bool is_outgoing;
                };
                std::vector<MemletRewrite> group_rewrites;
                bool all_in_group = true;

                for (auto& memlet : dfg.out_edges(*access)) {
                    if (group_memlets_.count(&memlet) == 0) {
                        all_in_group = false;
                        continue;
                    }
                    auto* acc = mla.access(memlet);
                    if (acc && acc->subset.size() == tile_info_.dimensions.size()) {
                        std::vector<symbolic::Expression> local_indices;
                        for (size_t d = 0; d < tile_info_.dimensions.size(); d++) {
                            if (!symbolic::eq(tile_info_.dimensions.at(d), symbolic::integer(1))) {
                                local_indices.push_back(symbolic::sub(acc->subset.at(d), tile_info_.bases.at(d)));
                            }
                        }
                        symbolic::Expression linear_idx = linearize_exprs(local_indices);
                        group_rewrites.push_back({&memlet, {linear_idx}, true});
                    }
                }
                for (auto& memlet : dfg.in_edges(*access)) {
                    if (group_memlets_.count(&memlet) == 0) {
                        all_in_group = false;
                        continue;
                    }
                    auto* acc = mla.access(memlet);
                    if (acc && acc->subset.size() == tile_info_.dimensions.size()) {
                        std::vector<symbolic::Expression> local_indices;
                        for (size_t d = 0; d < tile_info_.dimensions.size(); d++) {
                            if (!symbolic::eq(tile_info_.dimensions.at(d), symbolic::integer(1))) {
                                local_indices.push_back(symbolic::sub(acc->subset.at(d), tile_info_.bases.at(d)));
                            }
                        }
                        symbolic::Expression linear_idx = linearize_exprs(local_indices);
                        group_rewrites.push_back({&memlet, {linear_idx}, false});
                    }
                }

                if (group_rewrites.empty()) continue;

                if (all_in_group) {
                    // Simple case: all memlets in group → rewrite in-place and rename
                    for (auto& rw : group_rewrites) {
                        rw.memlet->set_subset(rw.local_subset);
                        rw.memlet->set_base_type(buffer_type);
                    }
                    access->data(local_name_);
                } else {
                    // Mixed case: split — create new local access node, redirect group memlets
                    auto& local_access = builder.add_access(*block, local_name_);
                    for (auto& rw : group_rewrites) {
                        if (rw.is_outgoing) {
                            // outgoing: access→tasklet  →  local_access→tasklet
                            auto& dst_node = rw.memlet->dst();
                            auto dst_conn = rw.memlet->dst_conn();
                            builder.remove_memlet(*block, *rw.memlet);
                            builder.add_memlet(
                                *block, local_access, "void", dst_node, dst_conn, rw.local_subset, buffer_type, {}
                            );
                        } else {
                            // incoming: tasklet→access  →  tasklet→local_access
                            auto& src_node = rw.memlet->src();
                            auto src_conn = rw.memlet->src_conn();
                            builder.remove_memlet(*block, *rw.memlet);
                            builder.add_memlet(
                                *block, src_node, src_conn, local_access, "void", rw.local_subset, buffer_type, {}
                            );
                        }
                    }
                }
            }
        } else if (auto* seq = dynamic_cast<structured_control_flow::Sequence*>(&node)) {
            for (size_t i = 0; i < seq->size(); i++) {
                rewrite_accesses(seq->at(i).first);
            }
        } else if (auto* loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&node)) {
            rewrite_accesses(loop->root());
        } else if (auto* if_else = dynamic_cast<structured_control_flow::IfElse*>(&node)) {
            for (size_t i = 0; i < if_else->size(); i++) {
                rewrite_accesses(if_else->at(i).first);
            }
        }
    };
    rewrite_accesses(loop_.root());

    // Cleanup
    analysis_manager.invalidate_all();

    passes::SequenceFusion sf_pass;
    passes::DeadCFGElimination dce_pass;
    bool applies = false;
    do {
        applies = false;
        applies |= dce_pass.run(builder, analysis_manager);
        applies |= sf_pass.run(builder, analysis_manager);
    } while (applies);
}

void InLocalStorage::to_json(nlohmann::json& j) const {
    std::string loop_type;
    if (dynamic_cast<structured_control_flow::For*>(&loop_)) {
        loop_type = "for";
    } else if (dynamic_cast<structured_control_flow::Map*>(&loop_)) {
        loop_type = "map";
    } else {
        throw std::runtime_error("Unsupported loop type for serialization of loop: " + loop_.indvar()->get_name());
    }
    j["subgraph"] = {
        {"0", {{"element_id", this->loop_.element_id()}, {"type", loop_type}}},
        {"1", {{"element_id", this->access_node_.element_id()}, {"type", "access_node"}}}
    };
    j["transformation_type"] = this->name();
    j["container"] = container_;
}

InLocalStorage InLocalStorage::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc) {
    auto loop_id = desc["subgraph"]["0"]["element_id"].get<size_t>();
    auto element = builder.find_element_by_id(loop_id);
    if (!element) {
        throw InvalidTransformationDescriptionException("Element with ID " + std::to_string(loop_id) + " not found.");
    }
    auto loop = dynamic_cast<structured_control_flow::StructuredLoop*>(element);
    if (!loop) {
        throw InvalidTransformationDescriptionException(
            "Element with ID " + std::to_string(loop_id) + " is not a structured loop."
        );
    }

    auto access_node = dynamic_cast<
        data_flow::AccessNode*>(builder.find_element_by_id(desc.at("subgraph").at("1").at("element_id").get<size_t>()));
    if (!access_node) {
        throw InvalidTransformationDescriptionException(
            "Access node with ID " + std::to_string(desc.at("subgraph").at("1").at("element_id").get<size_t>()) +
            " not found."
        );
    }

    return InLocalStorage(*loop, *access_node);
}

} // namespace transformations
} // namespace sdfg
