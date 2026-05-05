#include "sdfg/transformations/out_local_storage.h"

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

OutLocalStorage::OutLocalStorage(
    structured_control_flow::StructuredLoop& loop,
    const data_flow::AccessNode& access_node,
    const types::StorageType& storage_type
)
    : loop_(loop), access_node_(access_node), container_(access_node.data()), storage_type_(storage_type) {};

std::string OutLocalStorage::name() const { return "OutLocalStorage"; };

bool OutLocalStorage::can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& sdfg = builder.subject();
    auto& body = this->loop_.root();

    tile_info_ = TileInfo{};

    // Criterion: Container must exist
    if (!sdfg.exists(this->container_)) {
        return false;
    }

    auto& type = sdfg.type(this->container_);

    // Criterion: Container must be used in the loop body
    auto& users = analysis_manager.get<analysis::Users>();
    analysis::UsersView body_users(users, body);
    if (body_users.uses(this->container_).empty()) {
        return false;
    }

    // Criterion: Container must have writes (this is OutLocalStorage, not InLocalStorage)
    if (body_users.writes(this->container_).empty()) {
        return false;
    }

    // Determine if container is also read (read-write vs write-only)
    tile_info_.has_read = !body_users.reads(this->container_).empty();

    // Handle scalar containers: no tile needed, dimensions stay empty
    if (type.type_id() == types::TypeID::Scalar) {
        return true;
    }

    // For Array/Pointer types: use MemoryLayoutAnalysis tile group API
    if (type.type_id() != types::TypeID::Pointer && type.type_id() != types::TypeID::Array) {
        return false;
    }

    auto& mla = analysis_manager.get<analysis::MemoryLayoutAnalysis>();

    // Find a representative memlet from the access node to identify its group.
    // An access node may have multiple edges belonging to different tile groups.
    // We iterate all edges and select the first one whose tile group is valid
    // at the target loop level.
    const analysis::MemoryTileGroup* group = nullptr;
    auto& dfg = access_node_.get_parent();
    for (auto& memlet : dfg.in_edges(access_node_)) {
        auto* candidate = mla.tile_group_for(loop_, memlet);
        if (candidate) {
            group = candidate;
            break;
        }
    }
    if (!group) {
        for (auto& memlet : dfg.out_edges(access_node_)) {
            auto* candidate = mla.tile_group_for(loop_, memlet);
            if (candidate) {
                group = candidate;
                break;
            }
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

void OutLocalStorage::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& sdfg = builder.subject();
    auto& users = analysis_manager.get<analysis::Users>();
    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();

    auto parent_node = scope_analysis.parent_scope(&loop_);
    auto parent = dynamic_cast<structured_control_flow::Sequence*>(parent_node);
    if (!parent) {
        throw InvalidSDFGException("OutLocalStorage: Parent of loop must be a Sequence!");
    }

    // Get type information
    auto& type = sdfg.type(this->container_);
    types::Scalar scalar_type(type.primitive_type());

    // Create local buffer name
    local_name_ = builder.find_new_name("__daisy_out_local_storage_" + this->container_);

    // ========================================================================
    // SCALAR PATH: tile_info_.dimensions is empty
    // ========================================================================
    if (tile_info_.dimensions.empty()) {
        // Create scalar local buffer
        builder.add_container(local_name_, scalar_type);

        // Get the access subset from the first user (all scalar, so empty subset)
        analysis::UsersView body_users(users, loop_.root());
        auto accesses = body_users.uses(this->container_);
        auto first_access = accesses.at(0);
        auto first_subset = first_access->subsets().at(0);

        // Init block (copy from container to local) - before loop
        if (tile_info_.has_read) {
            auto& init_block = builder.add_block_before(*parent, loop_, {}, loop_.debug_info());
            auto& init_src = builder.add_access(init_block, this->container_);
            auto& init_dst = builder.add_access(init_block, local_name_);
            auto& init_tasklet = builder.add_tasklet(init_block, data_flow::TaskletCode::assign, "_out", {"_in"});
            builder.add_computational_memlet(init_block, init_src, init_tasklet, "_in", first_subset, type);
            builder.add_computational_memlet(init_block, init_tasklet, "_out", init_dst, {}, scalar_type);
        }

        // Writeback block (copy from local to container) - after loop
        {
            auto& wb_block = builder.add_block_after(*parent, loop_, {}, loop_.debug_info());
            auto& wb_src = builder.add_access(wb_block, local_name_);
            auto& wb_dst = builder.add_access(wb_block, this->container_);
            auto& wb_tasklet = builder.add_tasklet(wb_block, data_flow::TaskletCode::assign, "_out", {"_in"});
            builder.add_computational_memlet(wb_block, wb_src, wb_tasklet, "_in", {}, scalar_type);
            builder.add_computational_memlet(wb_block, wb_tasklet, "_out", wb_dst, first_subset, type);
        }

        // Rewrite body accesses to use scalar local
        for (auto* user : body_users.uses(this->container_)) {
            auto element = user->element();
            if (auto access = dynamic_cast<data_flow::AccessNode*>(element)) {
                for (auto& iedge : access->get_parent().in_edges(*access)) {
                    auto memlet = &iedge;
                    memlet->set_subset({});
                    memlet->set_base_type(scalar_type);
                }
                for (auto& oedge : access->get_parent().out_edges(*access)) {
                    auto memlet = &oedge;
                    memlet->set_subset({});
                    memlet->set_base_type(scalar_type);
                }
            }
        }

        // Replace container name in the loop body
        loop_.replace(symbolic::symbol(this->container_), symbolic::symbol(local_name_));
    }
    // ========================================================================
    // ARRAY PATH: tile_info_.dimensions is non-empty
    // ========================================================================
    else {
        // Compute total buffer size
        symbolic::Expression total_size = symbolic::integer(1);
        for (auto& ds : tile_info_.dimensions) {
            total_size = symbolic::mul(total_size, ds);
        }

        // Create the local buffer with specified storage type
        types::Array buffer_type(storage_type_, 0, {}, scalar_type, total_size);
        builder.add_container(local_name_, buffer_type);

        // Helper: build linearized local index from per-dimension expressions
        auto linearize_exprs = [&](const std::vector<symbolic::Expression>& indices) -> symbolic::Expression {
            symbolic::Expression linear_idx = symbolic::integer(0);
            symbolic::Expression stride = symbolic::integer(1);
            for (int i = indices.size() - 1; i >= 0; i--) {
                linear_idx = symbolic::add(linear_idx, symbolic::mul(indices[i], stride));
                stride = symbolic::mul(stride, tile_info_.dimensions[i]);
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

        if (storage_type_.is_nv_shared()) {
            // ============================================================
            // GPU COOPERATIVE PATH
            // ============================================================
            auto ancestors = scope_analysis.ancestor_scopes(&loop_);

            // Collect cooperative GPU dimensions
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

            // Flatten cooperative thread index
            symbolic::Expression coop_flat = symbolic::integer(0);
            symbolic::Expression coop_stride = symbolic::integer(1);
            for (int i = coop_dims.size() - 1; i >= 0; i--) {
                coop_flat = symbolic::add(coop_flat, symbolic::mul(coop_dims[i].indvar, coop_stride));
                coop_stride = symbolic::mul(coop_stride, coop_dims[i].block_size);
            }

            // INIT: barrier → cooperative copy-in → barrier (if has_read)
            if (tile_info_.has_read) {
                // Barrier before init
                auto& barrier_block1 = builder.add_block_before(*parent, loop_, {}, loop_.debug_info());
                builder.add_library_node<data_flow::BarrierLocalNode>(barrier_block1, {});

                // Cooperative copy-in loop
                auto idx_name = builder.find_new_name("__daisy_ols_coop_init_" + this->container_);
                types::Scalar idx_type(types::PrimitiveType::UInt64);
                builder.add_container(idx_name, idx_type);
                auto idx_var = symbolic::symbol(idx_name);

                auto& init_loop = builder.add_map_before(
                    *parent,
                    loop_,
                    idx_var,
                    symbolic::Lt(idx_var, total_size),
                    coop_flat,
                    symbolic::add(idx_var, total_coop_threads),
                    structured_control_flow::ScheduleType_Sequential::create(),
                    {},
                    loop_.debug_info()
                );

                auto& init_block = builder.add_block(init_loop.root());
                auto& init_src = builder.add_access(init_block, this->container_);
                auto& init_dst = builder.add_access(init_block, local_name_);
                auto& init_tasklet = builder.add_tasklet(init_block, data_flow::TaskletCode::assign, "_out", {"_in"});

                // Decompose idx_var into per-dim indices
                std::vector<symbolic::Expression> init_indices;
                symbolic::Expression remainder = idx_var;
                for (size_t i = 0; i < tile_info_.dimensions.size(); i++) {
                    if (i < tile_info_.dimensions.size() - 1) {
                        symbolic::Expression divisor = symbolic::integer(1);
                        for (size_t j = i + 1; j < tile_info_.dimensions.size(); j++) {
                            divisor = symbolic::mul(divisor, tile_info_.dimensions[j]);
                        }
                        init_indices.push_back(symbolic::div(remainder, divisor));
                        remainder = symbolic::mod(remainder, divisor);
                    } else {
                        init_indices.push_back(remainder);
                    }
                }

                auto init_src_subset = build_original_subset(init_indices);
                builder.add_computational_memlet(init_block, init_src, init_tasklet, "_in", init_src_subset, type);
                builder.add_computational_memlet(init_block, init_tasklet, "_out", init_dst, {idx_var}, buffer_type);

                // Barrier after init
                auto& barrier_block2 = builder.add_block_before(*parent, loop_, {}, loop_.debug_info());
                builder.add_library_node<data_flow::BarrierLocalNode>(barrier_block2, {});
            }

            // WRITEBACK: barrier → cooperative copy-out → barrier
            {
                // Barrier before writeback
                auto& barrier_block3 = builder.add_block_after(*parent, loop_, {}, loop_.debug_info());
                builder.add_library_node<data_flow::BarrierLocalNode>(barrier_block3, {});

                // Cooperative writeback loop
                auto idx_name = builder.find_new_name("__daisy_ols_coop_wb_" + this->container_);
                types::Scalar idx_type(types::PrimitiveType::UInt64);
                builder.add_container(idx_name, idx_type);
                auto idx_var = symbolic::symbol(idx_name);

                auto& wb_loop = builder.add_map_after(
                    *parent,
                    loop_,
                    idx_var,
                    symbolic::Lt(idx_var, total_size),
                    coop_flat,
                    symbolic::add(idx_var, total_coop_threads),
                    structured_control_flow::ScheduleType_Sequential::create(),
                    {},
                    loop_.debug_info()
                );

                auto& wb_block = builder.add_block(wb_loop.root());
                auto& wb_src = builder.add_access(wb_block, local_name_);
                auto& wb_dst = builder.add_access(wb_block, this->container_);
                auto& wb_tasklet = builder.add_tasklet(wb_block, data_flow::TaskletCode::assign, "_out", {"_in"});

                // Decompose idx_var into per-dim indices
                std::vector<symbolic::Expression> wb_indices;
                symbolic::Expression remainder = idx_var;
                for (size_t i = 0; i < tile_info_.dimensions.size(); i++) {
                    if (i < tile_info_.dimensions.size() - 1) {
                        symbolic::Expression divisor = symbolic::integer(1);
                        for (size_t j = i + 1; j < tile_info_.dimensions.size(); j++) {
                            divisor = symbolic::mul(divisor, tile_info_.dimensions[j]);
                        }
                        wb_indices.push_back(symbolic::div(remainder, divisor));
                        remainder = symbolic::mod(remainder, divisor);
                    } else {
                        wb_indices.push_back(remainder);
                    }
                }

                auto wb_dst_subset = build_original_subset(wb_indices);
                builder.add_computational_memlet(wb_block, wb_src, wb_tasklet, "_in", {idx_var}, buffer_type);
                builder.add_computational_memlet(wb_block, wb_tasklet, "_out", wb_dst, wb_dst_subset, type);

                // Barrier after writeback
                auto& barrier_block4 = builder.add_block_after(*parent, loop_, {}, loop_.debug_info());
                builder.add_library_node<data_flow::BarrierLocalNode>(barrier_block4, {});
            }
        } else {
            // ============================================================
            // CPU SEQUENTIAL PATH
            // ============================================================
            if (tile_info_.has_read) {
                std::vector<symbolic::Symbol> init_indvars;
                structured_control_flow::Sequence* init_scope = parent;
                bool first_init_loop = true;

                for (size_t i = 0; i < tile_info_.dimensions.size(); i++) {
                    size_t d = i;
                    auto indvar_name =
                        builder.find_new_name("__daisy_ols_init_" + this->container_ + "_d" + std::to_string(d));
                    types::Scalar indvar_type(types::PrimitiveType::UInt64);
                    builder.add_container(indvar_name, indvar_type);
                    auto indvar = symbolic::symbol(indvar_name);
                    init_indvars.push_back(indvar);

                    auto init = symbolic::integer(0);
                    auto condition = symbolic::Lt(indvar, tile_info_.dimensions[i]);
                    auto update = symbolic::add(indvar, symbolic::integer(1));

                    if (first_init_loop) {
                        auto& init_loop = builder.add_map_before(
                            *init_scope,
                            loop_,
                            indvar,
                            condition,
                            init,
                            update,
                            structured_control_flow::ScheduleType_Sequential::create(),
                            {},
                            loop_.debug_info()
                        );
                        init_scope = &init_loop.root();
                        first_init_loop = false;
                    } else {
                        auto& init_loop = builder.add_map(
                            *init_scope,
                            indvar,
                            condition,
                            init,
                            update,
                            structured_control_flow::ScheduleType_Sequential::create(),
                            {},
                            loop_.debug_info()
                        );
                        init_scope = &init_loop.root();
                    }
                }

                // Create init copy block
                auto& init_block = builder.add_block(*init_scope);
                auto& init_src = builder.add_access(init_block, this->container_);
                auto& init_dst = builder.add_access(init_block, local_name_);
                auto& init_tasklet = builder.add_tasklet(init_block, data_flow::TaskletCode::assign, "_out", {"_in"});

                std::vector<symbolic::Expression> init_exprs(init_indvars.begin(), init_indvars.end());
                auto init_src_subset = build_original_subset(init_exprs);
                data_flow::Subset init_dst_subset = {linearize(init_indvars)};

                builder.add_computational_memlet(init_block, init_src, init_tasklet, "_in", init_src_subset, type);
                builder
                    .add_computational_memlet(init_block, init_tasklet, "_out", init_dst, init_dst_subset, buffer_type);
            }

            // Writeback Maps
            {
                std::vector<symbolic::Symbol> wb_indvars;
                structured_control_flow::Sequence* wb_scope = parent;
                bool first_wb_loop = true;

                for (size_t i = 0; i < tile_info_.dimensions.size(); i++) {
                    size_t d = i;
                    auto indvar_name =
                        builder.find_new_name("__daisy_ols_wb_" + this->container_ + "_d" + std::to_string(d));
                    types::Scalar indvar_type(types::PrimitiveType::UInt64);
                    builder.add_container(indvar_name, indvar_type);
                    auto indvar = symbolic::symbol(indvar_name);
                    wb_indvars.push_back(indvar);

                    auto init = symbolic::integer(0);
                    auto condition = symbolic::Lt(indvar, tile_info_.dimensions[i]);
                    auto update = symbolic::add(indvar, symbolic::integer(1));

                    if (first_wb_loop) {
                        auto& wb_loop = builder.add_map_after(
                            *wb_scope,
                            loop_,
                            indvar,
                            condition,
                            init,
                            update,
                            structured_control_flow::ScheduleType_Sequential::create(),
                            {},
                            loop_.debug_info()
                        );
                        wb_scope = &wb_loop.root();
                        first_wb_loop = false;
                    } else {
                        auto& wb_loop = builder.add_map(
                            *wb_scope,
                            indvar,
                            condition,
                            init,
                            update,
                            structured_control_flow::ScheduleType_Sequential::create(),
                            {},
                            loop_.debug_info()
                        );
                        wb_scope = &wb_loop.root();
                    }
                }

                // Create writeback copy block
                auto& wb_block = builder.add_block(*wb_scope);
                auto& wb_src = builder.add_access(wb_block, local_name_);
                auto& wb_dst = builder.add_access(wb_block, this->container_);
                auto& wb_tasklet = builder.add_tasklet(wb_block, data_flow::TaskletCode::assign, "_out", {"_in"});

                std::vector<symbolic::Expression> wb_exprs(wb_indvars.begin(), wb_indvars.end());
                data_flow::Subset wb_src_subset = {linearize(wb_indvars)};
                auto wb_dst_subset = build_original_subset(wb_exprs);

                builder.add_computational_memlet(wb_block, wb_src, wb_tasklet, "_in", wb_src_subset, buffer_type);
                builder.add_computational_memlet(wb_block, wb_tasklet, "_out", wb_dst, wb_dst_subset, type);
            }
        }

        // ==================================================================
        // Update accesses in the main loop to use the local buffer
        // ==================================================================
        auto& mla = analysis_manager.get<analysis::MemoryLayoutAnalysis>();

        // Recursive helper to traverse all blocks in the loop body
        std::function<void(structured_control_flow::ControlFlowNode&)> rewrite_accesses;
        rewrite_accesses = [&](structured_control_flow::ControlFlowNode& node) {
            if (auto* block = dynamic_cast<structured_control_flow::Block*>(&node)) {
                auto& dfg = block->dataflow();
                for (auto* access : dfg.data_nodes()) {
                    if (access->data() != this->container_) continue;
                    bool all_rewritten = true;
                    // Rewrite outgoing memlets (reads from this access node)
                    for (auto& memlet : dfg.out_edges(*access)) {
                        if (group_memlets_.count(&memlet) == 0) {
                            all_rewritten = false;
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
                            memlet.set_subset({linear_idx});
                            memlet.set_base_type(buffer_type);
                        }
                    }
                    // Rewrite incoming memlets (writes to this access node)
                    for (auto& memlet : dfg.in_edges(*access)) {
                        if (group_memlets_.count(&memlet) == 0) {
                            all_rewritten = false;
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
                            memlet.set_subset({linear_idx});
                            memlet.set_base_type(buffer_type);
                        }
                    }
                    // Rename the access node only if all its memlets belong to our group
                    if (all_rewritten) {
                        access->data(local_name_);
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
    }

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
};

void OutLocalStorage::to_json(nlohmann::json& j) const {
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
};

OutLocalStorage OutLocalStorage::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc) {
    auto loop_id = desc["subgraph"]["0"]["element_id"].get<size_t>();
    auto element = builder.find_element_by_id(loop_id);
    if (!element) {
        throw InvalidTransformationDescriptionException("Element with ID " + std::to_string(loop_id) + " not found.");
    }
    auto loop = dynamic_cast<structured_control_flow::StructuredLoop*>(element);

    auto access_node = dynamic_cast<
        data_flow::AccessNode*>(builder.find_element_by_id(desc.at("subgraph").at("1").at("element_id").get<size_t>()));
    if (!access_node) {
        throw InvalidTransformationDescriptionException(
            "Access node with ID " + std::to_string(desc.at("subgraph").at("1").at("element_id").get<size_t>()) +
            " not found."
        );
    }

    return OutLocalStorage(*loop, *access_node);
};

} // namespace transformations
} // namespace sdfg
