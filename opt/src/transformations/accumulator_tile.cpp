#include "sdfg/transformations/accumulator_tile.h"

#include <cassert>
#include <cstddef>
#include <string>

#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/analysis/type_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/passes/structured_control_flow/dead_cfg_elimination.h"
#include "sdfg/passes/structured_control_flow/sequence_fusion.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/transformations/utils.h"
#include "sdfg/types/array.h"
#include "sdfg/types/scalar.h"

namespace sdfg {
namespace transformations {

AccumulatorTile::AccumulatorTile(structured_control_flow::StructuredLoop& loop, std::string container)
    : loop_(loop), container_(container) {}

std::string AccumulatorTile::name() const { return "AccumulatorTile"; }

bool AccumulatorTile::can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& sdfg = builder.subject();
    auto& body = this->loop_.root();

    // Clear any previous analysis
    tile_info_ = TileInfo{};

    // Check container exists
    if (!sdfg.exists(this->container_)) {
        return false;
    }

    // Check container is an array/pointer type
    auto& type = sdfg.type(this->container_);
    if (type.type_id() != types::TypeID::Pointer && type.type_id() != types::TypeID::Array) {
        return false;
    }

    // Check container is used in the loop
    auto& users = analysis_manager.get<analysis::Users>();
    analysis::UsersView body_users(users, body);
    auto accesses = body_users.uses(this->container_);
    if (accesses.size() == 0) {
        return false;
    }

    // Check that container has BOTH reads and writes (accumulator pattern)
    // - has reads: uses exist that aren't writes
    // - has writes: body_users.writes() is non-empty
    bool has_write = !body_users.writes(this->container_).empty();
    bool has_read = accesses.size() > body_users.writes(this->container_).size();

    if (!has_read || !has_write) {
        return false;
    }

    // Get the first access's subset as representative
    auto first_access = accesses.at(0);
    if (first_access->subsets().empty()) {
        return false;
    }
    tile_info_.representative_subset = first_access->subsets().at(0);

    // Check all accesses have identical subsets
    for (auto* access : accesses) {
        if (access->subsets().size() != first_access->subsets().size()) {
            return false;
        }
        for (size_t i = 0; i < first_access->subsets().size(); i++) {
            auto& first_sub = tile_info_.representative_subset;
            auto& sub = access->subsets().at(i);
            if (first_sub.size() != sub.size()) {
                return false;
            }
            for (size_t j = 0; j < first_sub.size(); j++) {
                if (!symbolic::eq(first_sub.at(j), sub.at(j))) {
                    return false;
                }
            }
        }
    }

    // Get loop analysis to find nested loops
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();

    // Get all descendant For loops (NOT including target loop)
    auto descendants = loop_analysis.descendants(&loop_);

    // Collect indvars from nested For loops
    std::vector<symbolic::Symbol> inner_indvars;
    for (auto* desc : descendants) {
        if (auto* for_loop = dynamic_cast<structured_control_flow::For*>(desc)) {
            inner_indvars.push_back(for_loop->indvar());
            tile_info_.inner_loops.push_back(for_loop);
        }
    }

    if (inner_indvars.empty()) {
        // No inner loops - degenerate case, use OutLocalStorage instead
        return false;
    }

    // Analyze each dimension
    auto& subset = tile_info_.representative_subset;
    bool found_inner_loop_dim = false;

    for (size_t dim = 0; dim < subset.size(); dim++) {
        auto& index_expr = subset.at(dim);
        auto atoms = symbolic::atoms(index_expr);

        // Find which inner loop indvar, if any, appears in this dimension's index
        symbolic::Symbol found_indvar = SymEngine::null;
        structured_control_flow::For* found_loop = nullptr;

        for (size_t i = 0; i < inner_indvars.size(); i++) {
            for (auto& atom : atoms) {
                if (symbolic::eq(atom, inner_indvars[i])) {
                    found_indvar = inner_indvars[i];
                    found_loop = tile_info_.inner_loops[i];
                    break;
                }
            }
            if (found_indvar != SymEngine::null) break;
        }

        if (found_indvar != SymEngine::null && found_loop) {
            // This dimension varies with an inner loop
            // Get iteration count of that loop
            auto dim_size = found_loop->num_iterations();
            if (dim_size.is_null()) {
                return false;
            }

            // Base is index_expr with inner indvar replaced by loop init
            auto base = symbolic::subs(index_expr, found_indvar, found_loop->init());

            tile_info_.dimensions.push_back(dim_size);
            tile_info_.bases.push_back(base);
            found_inner_loop_dim = true;
        } else {
            // Constant dimension - size 1
            tile_info_.dimensions.push_back(symbolic::integer(1));
            tile_info_.bases.push_back(index_expr);
        }
    }

    // Must have at least one dimension that varies with inner loops
    if (!found_inner_loop_dim) {
        return false;
    }

    return true;
}

void AccumulatorTile::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& sdfg = builder.subject();
    auto& users = analysis_manager.get<analysis::Users>();
    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();

    auto parent_node = scope_analysis.parent_scope(&loop_);
    auto parent = dynamic_cast<structured_control_flow::Sequence*>(parent_node);
    if (!parent) {
        throw InvalidSDFGException("AccumulatorTile: Parent of loop must be a Sequence!");
    }

    // Get type information
    auto& type = sdfg.type(this->container_);
    types::Scalar scalar_type(type.primitive_type());

    // Create tile buffer name
    local_name_ = "__daisy_accumulator_tile_" + this->container_;

    // Compute total tile size
    symbolic::Expression total_size = symbolic::integer(1);
    for (auto& dim : tile_info_.dimensions) {
        if (!symbolic::eq(dim, symbolic::integer(1))) {
            total_size = symbolic::mul(total_size, dim);
        }
    }

    // Create the tile buffer array
    types::Array tile_type(scalar_type, total_size);
    builder.add_container(local_name_, tile_type);

    // Get representative subset
    auto& first_subset = tile_info_.representative_subset;

    // ========================================================================
    // Create INIT loops (copy from C to tile) - before target loop
    // ========================================================================
    std::vector<symbolic::Symbol> init_indvars;
    structured_control_flow::Sequence* init_scope = parent;
    bool first_init_loop = true;

    for (size_t d = 0; d < tile_info_.dimensions.size(); d++) {
        auto& dim_size = tile_info_.dimensions.at(d);
        if (symbolic::eq(dim_size, symbolic::integer(1))) {
            init_indvars.push_back(SymEngine::null);
            continue;
        }

        auto indvar_name = "__daisy_acc_init_" + this->container_ + "_d" + std::to_string(d);
        types::Scalar indvar_type(types::PrimitiveType::UInt64);
        builder.add_container(indvar_name, indvar_type);
        auto indvar = symbolic::symbol(indvar_name);
        init_indvars.push_back(indvar);

        auto init = symbolic::integer(0);
        auto condition = symbolic::Lt(indvar, dim_size);
        auto update = symbolic::add(indvar, symbolic::integer(1));

        if (first_init_loop) {
            // First init loop: insert before the target loop
            auto& init_loop =
                builder.add_for_before(*init_scope, loop_, indvar, condition, init, update, {}, loop_.debug_info());
            init_scope = &init_loop.root();
            first_init_loop = false;
        } else {
            // Nested init loops: add inside the current init scope
            auto& init_loop = builder.add_for(*init_scope, indvar, condition, init, update, {}, loop_.debug_info());
            init_scope = &init_loop.root();
        }
    }

    // Create init copy block
    auto& init_block = builder.add_block(*init_scope);
    auto& init_src = builder.add_access(init_block, this->container_);
    auto& init_dst = builder.add_access(init_block, local_name_);
    auto& init_tasklet = builder.add_tasklet(init_block, data_flow::TaskletCode::assign, "_out", {"_in"});

    // Build source subset (base + init_indvar) for original container
    data_flow::Subset init_src_subset;
    for (size_t d = 0; d < tile_info_.dimensions.size(); d++) {
        if (init_indvars[d] != SymEngine::null) {
            init_src_subset.push_back(symbolic::add(tile_info_.bases[d], init_indvars[d]));
        } else {
            init_src_subset.push_back(tile_info_.bases[d]);
        }
    }

    // Compute linearized index for tile buffer
    // Linear index = sum of (indvar[d] * stride[d]) for each varying dimension
    // stride[d] = product of dimensions[d+1..n]
    std::vector<symbolic::Expression> varying_dims;
    std::vector<symbolic::Symbol> varying_indvars;
    for (size_t d = 0; d < tile_info_.dimensions.size(); d++) {
        if (init_indvars[d] != SymEngine::null) {
            varying_dims.push_back(tile_info_.dimensions[d]);
            varying_indvars.push_back(init_indvars[d]);
        }
    }

    symbolic::Expression init_linear_idx = symbolic::integer(0);
    symbolic::Expression stride = symbolic::integer(1);
    for (int d = varying_indvars.size() - 1; d >= 0; d--) {
        init_linear_idx = symbolic::add(init_linear_idx, symbolic::mul(varying_indvars[d], stride));
        stride = symbolic::mul(stride, varying_dims[d]);
    }

    data_flow::Subset init_dst_subset = {init_linear_idx};

    builder.add_computational_memlet(init_block, init_src, init_tasklet, "_in", init_src_subset, type);
    builder.add_computational_memlet(init_block, init_tasklet, "_out", init_dst, init_dst_subset, tile_type);

    // ========================================================================
    // Create WRITEBACK loops (copy from tile to C) - after target loop
    // ========================================================================
    std::vector<symbolic::Symbol> wb_indvars;
    structured_control_flow::Sequence* wb_scope = parent;
    bool first_wb_loop = true;

    for (size_t d = 0; d < tile_info_.dimensions.size(); d++) {
        auto& dim_size = tile_info_.dimensions.at(d);
        if (symbolic::eq(dim_size, symbolic::integer(1))) {
            wb_indvars.push_back(SymEngine::null);
            continue;
        }

        auto indvar_name = "__daisy_acc_wb_" + this->container_ + "_d" + std::to_string(d);
        types::Scalar indvar_type(types::PrimitiveType::UInt64);
        builder.add_container(indvar_name, indvar_type);
        auto indvar = symbolic::symbol(indvar_name);
        wb_indvars.push_back(indvar);

        auto init = symbolic::integer(0);
        auto condition = symbolic::Lt(indvar, dim_size);
        auto update = symbolic::add(indvar, symbolic::integer(1));

        if (first_wb_loop) {
            // First writeback loop: insert after the target loop
            auto& wb_loop =
                builder.add_for_after(*wb_scope, loop_, indvar, condition, init, update, {}, loop_.debug_info());
            wb_scope = &wb_loop.root();
            first_wb_loop = false;
        } else {
            // Nested writeback loops: add inside the current writeback scope
            auto& wb_loop = builder.add_for(*wb_scope, indvar, condition, init, update, {}, loop_.debug_info());
            wb_scope = &wb_loop.root();
        }
    }

    // Create writeback copy block
    auto& wb_block = builder.add_block(*wb_scope);
    auto& wb_src = builder.add_access(wb_block, local_name_);
    auto& wb_dst = builder.add_access(wb_block, this->container_);
    auto& wb_tasklet = builder.add_tasklet(wb_block, data_flow::TaskletCode::assign, "_out", {"_in"});

    // Build destination subset for original container
    data_flow::Subset wb_dst_subset;
    for (size_t d = 0; d < tile_info_.dimensions.size(); d++) {
        if (wb_indvars[d] != SymEngine::null) {
            wb_dst_subset.push_back(symbolic::add(tile_info_.bases[d], wb_indvars[d]));
        } else {
            wb_dst_subset.push_back(tile_info_.bases[d]);
        }
    }

    // Compute linearized index for writeback source
    std::vector<symbolic::Expression> wb_varying_dims;
    std::vector<symbolic::Symbol> wb_varying_indvars;
    for (size_t d = 0; d < tile_info_.dimensions.size(); d++) {
        if (wb_indvars[d] != SymEngine::null) {
            wb_varying_dims.push_back(tile_info_.dimensions[d]);
            wb_varying_indvars.push_back(wb_indvars[d]);
        }
    }

    symbolic::Expression wb_linear_idx = symbolic::integer(0);
    symbolic::Expression wb_stride = symbolic::integer(1);
    for (int d = wb_varying_indvars.size() - 1; d >= 0; d--) {
        wb_linear_idx = symbolic::add(wb_linear_idx, symbolic::mul(wb_varying_indvars[d], wb_stride));
        wb_stride = symbolic::mul(wb_stride, wb_varying_dims[d]);
    }

    data_flow::Subset wb_src_subset = {wb_linear_idx};

    builder.add_computational_memlet(wb_block, wb_src, wb_tasklet, "_in", wb_src_subset, tile_type);
    builder.add_computational_memlet(wb_block, wb_tasklet, "_out", wb_dst, wb_dst_subset, type);

    // ========================================================================
    // Update accesses in the main loop to use tile
    // ========================================================================
    analysis::UsersView body_users(users, loop_.root());
    for (auto* user : body_users.uses(this->container_)) {
        auto element = user->element();
        if (auto memlet = dynamic_cast<data_flow::Memlet*>(element)) {
            auto& old_subset = memlet->subset();

            // Compute linearized index: sum of ((old_idx[d] - base[d]) * stride[d])
            // where stride includes products of dimensions after d
            symbolic::Expression linear_idx = symbolic::integer(0);
            symbolic::Expression current_stride = symbolic::integer(1);

            // Collect varying dimensions for stride calculation
            std::vector<int> varying_dim_indices;
            for (size_t d = 0; d < old_subset.size(); d++) {
                if (!symbolic::eq(tile_info_.dimensions[d], symbolic::integer(1))) {
                    varying_dim_indices.push_back(d);
                }
            }

            // Compute linearized index (row-major order)
            for (int i = varying_dim_indices.size() - 1; i >= 0; i--) {
                size_t d = varying_dim_indices[i];
                auto local_idx = symbolic::sub(old_subset.at(d), tile_info_.bases[d]);
                linear_idx = symbolic::add(linear_idx, symbolic::mul(local_idx, current_stride));
                current_stride = symbolic::mul(current_stride, tile_info_.dimensions[d]);
            }

            data_flow::Subset new_subset = {linear_idx};
            memlet->set_subset(new_subset);
            memlet->set_base_type(tile_type);
        }
    }

    // Replace container name in the loop body
    loop_.replace(symbolic::symbol(this->container_), symbolic::symbol(local_name_));

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

void AccumulatorTile::to_json(nlohmann::json& j) const {
    std::string loop_type;
    if (dynamic_cast<structured_control_flow::For*>(&loop_)) {
        loop_type = "for";
    } else if (dynamic_cast<structured_control_flow::Map*>(&loop_)) {
        loop_type = "map";
    } else {
        throw std::runtime_error("Unsupported loop type for serialization");
    }
    j["subgraph"] = {{"0", {{"element_id", this->loop_.element_id()}, {"type", loop_type}}}};
    j["transformation_type"] = this->name();
    j["container"] = container_;
}

AccumulatorTile AccumulatorTile::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc) {
    auto loop_id = desc["subgraph"]["0"]["element_id"].get<size_t>();
    std::string container = desc["container"].get<std::string>();
    auto element = builder.find_element_by_id(loop_id);
    if (!element) {
        throw InvalidTransformationDescriptionException("Element with ID " + std::to_string(loop_id) + " not found.");
    }
    auto loop = dynamic_cast<structured_control_flow::StructuredLoop*>(element);
    if (!loop) {
        throw InvalidTransformationDescriptionException("Element with ID " + std::to_string(loop_id) + " is not a loop.");
    }
    return AccumulatorTile(*loop, container);
}

} // namespace transformations
} // namespace sdfg
