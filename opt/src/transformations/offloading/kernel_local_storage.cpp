#include "sdfg/transformations/offloading/kernel_local_storage.h"

#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/analysis/type_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/library_nodes/barrier_local_node.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/exceptions.h"
#include "sdfg/passes/dataflow/trivial_array_elimination.h"
#include "sdfg/passes/structured_control_flow/dead_cfg_elimination.h"
#include "sdfg/passes/structured_control_flow/sequence_fusion.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/polynomials.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/gpu/gpu_schedule_type.h"
#include "sdfg/transformations/utils.h"
#include "sdfg/types/array.h"
#include "sdfg/types/structure.h"
#include "sdfg/types/type.h"
#include "sdfg/types/utils.h"
#include "symengine/symengine_rcp.h"

namespace sdfg {
namespace transformations {

KernelLocalStorage::KernelLocalStorage(
    structured_control_flow::StructuredLoop& loop, symbolic::Expression offset, const std::string& container
)
    : loop_(loop), offset_(offset), container_(container) {};

std::string KernelLocalStorage::name() const { return "KernelLocalStorage"; };

bool KernelLocalStorage::reads_container(std::string container, analysis::UsersView& body_users) {
    if (body_users.reads(container).size() == 1) {
        return true;
    }
    return false;
}

bool KernelLocalStorage::uses_inner_indvar(analysis::UsersView& body_users) {
    bool result = false;
    for (auto& user : body_users.reads(this->container_)) {
        auto& subsets = user->subsets();
        if (subsets.size() == 0) {
            continue;
        }
        if (subsets.size() == 1) { // TODO: Handle multiple subsets
            for (auto access : subsets.at(0)) {
                result |= symbolic::uses(access, loop_.indvar());
            }
        }
    }
    return result;
};

std::tuple<symbolic::Integer, symbolic::Integer, symbolic::Integer> KernelLocalStorage::
    dim_size(const std::vector<structured_control_flow::ControlFlowNode*> ancestors) {
    symbolic::Integer x_dim_size = symbolic::one();
    symbolic::Integer y_dim_size = symbolic::one();
    symbolic::Integer z_dim_size = symbolic::one();

    for (auto node : ancestors) {
        if (auto ancestor_map = dynamic_cast<structured_control_flow::Map*>(node)) {
            auto schedule_type = ancestor_map->schedule_type();
            if (!gpu::is_gpu_schedule(schedule_type)) {
                continue;
            }
            auto dim = gpu::gpu_dimension(schedule_type);
            if (dim == gpu::GPUDimension::X) {
                x_dim_size = gpu::gpu_block_size(schedule_type);
            } else if (dim == gpu::GPUDimension::Y) {
                y_dim_size = gpu::gpu_block_size(schedule_type);
            } else if (dim == gpu::GPUDimension::Z) {
                z_dim_size = gpu::gpu_block_size(schedule_type);
            } else {
                throw InvalidSDFGException("Unknown dimension in GPU Schedule type: " + std::to_string((int) dim));
            }
        }
    }

    return {x_dim_size, y_dim_size, z_dim_size};
};

std::tuple<symbolic::Symbol, symbolic::Symbol, symbolic::Symbol> KernelLocalStorage::
    dim_indvars(const std::vector<structured_control_flow::ControlFlowNode*> ancestors) {
    symbolic::Symbol x_dim_indvar = SymEngine::null;
    symbolic::Symbol y_dim_indvar = SymEngine::null;
    symbolic::Symbol z_dim_indvar = SymEngine::null;

    for (auto node : ancestors) {
        if (auto ancestor_map = dynamic_cast<structured_control_flow::Map*>(node)) {
            auto schedule_type = ancestor_map->schedule_type();
            if (!gpu::is_gpu_schedule(schedule_type)) {
                continue;
            }
            auto dim = gpu::gpu_dimension(schedule_type);
            if (dim == gpu::GPUDimension::X) {
                x_dim_indvar = ancestor_map->indvar();
            } else if (dim == gpu::GPUDimension::Y) {
                y_dim_indvar = ancestor_map->indvar();
            } else if (dim == gpu::GPUDimension::Z) {
                z_dim_indvar = ancestor_map->indvar();
            } else {
                throw InvalidSDFGException("Unknown dimension in GPU Schedule type: " + std::to_string((int) dim));
            }
        }
    }

    return {x_dim_indvar, y_dim_indvar, z_dim_indvar};
}

std::tuple<bool, bool, bool> KernelLocalStorage::
    available_dims(std::vector<symbolic::Expression> subsets, analysis::AnalysisManager& analysis_manager) {
    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto ancestors = scope_analysis.ancestor_scopes(&loop_);

    symbolic::Integer iteration_count = get_iteration_count(loop_);

    auto [x_dim_size, y_dim_size, z_dim_size] = dim_size(ancestors);
    auto [x_dim_indvar, y_dim_indvar, z_dim_indvar] = dim_indvars(ancestors);

    bool x_dim_available = (x_dim_indvar != SymEngine::null);
    bool y_dim_available = (y_dim_indvar != SymEngine::null);
    bool z_dim_available = (z_dim_indvar != SymEngine::null);

    if (x_dim_available) {
        bool x_used = false;
        for (auto subset : subsets) {
            for (auto atom : symbolic::atoms(subset)) {
                if (symbolic::eq(atom, x_dim_indvar)) {
                    x_used = true;
                }
            }
        }
        if (x_used) {
            x_dim_available = false;
        }
    }
    if (y_dim_available) {
        bool y_used = false;
        for (auto subset : subsets) {
            for (auto atom : symbolic::atoms(subset)) {
                if (symbolic::eq(atom, y_dim_indvar)) {
                    y_used = true;
                }
            }
        }
        if (y_used) {
            y_dim_available = false;
        }
    }
    if (z_dim_available) {
        bool z_used = false;
        for (auto subset : subsets) {
            for (auto atom : symbolic::atoms(subset)) {
                if (symbolic::eq(atom, z_dim_indvar)) {
                    z_used = true;
                }
            }
        }
        if (z_used) {
            z_dim_available = false;
        }
    }

    if (x_dim_available) {
        auto cond = symbolic::Ge(x_dim_size, iteration_count);
        if (symbolic::is_true(cond)) {
            x_dim_available = true;
        }
    }
    if (y_dim_available) {
        auto cond = symbolic::Ge(y_dim_size, iteration_count);
        if (symbolic::is_true(cond)) {
            y_dim_available = true;
        }
    }
    if (z_dim_available) {
        auto cond = symbolic::Ge(z_dim_size, iteration_count);
        if (symbolic::is_true(cond)) {
            z_dim_available = true;
        }
    }

    return {x_dim_available, y_dim_available, z_dim_available};
}

bool KernelLocalStorage::is_candidate(
    structured_control_flow::StructuredLoop& loop,
    const std::string& container,
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager
) {
    auto& sdfg = builder.subject();

    // Criterion: transformation cannot be applied twice on the same container
    std::set<std::string> containers(sdfg.containers().begin(), sdfg.containers().end());
    std::string shared_container_name = "__daisy_shared_" + container;
    if (containers.find(shared_container_name) != containers.end()) {
        return false;
    }

    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto ancestors = scope_analysis.ancestor_scopes(&loop);

    // Criterion: Must not be a GPU map itself
    if (auto loop_map = dynamic_cast<structured_control_flow::Map*>(&loop)) {
        if (gpu::is_gpu_schedule(loop_map->schedule_type())) {
            return false;
        }
    }

    // Criterion: Must be nested in a GPU schedule
    bool is_gpu_scope = false;
    for (auto ancestor : ancestors) {
        if (auto ancestor_map = dynamic_cast<structured_control_flow::Map*>(ancestor)) {
            if (gpu::is_gpu_schedule(ancestor_map->schedule_type())) {
                is_gpu_scope = true;
            } else if (ancestor_map->schedule_type().value() == ScheduleType_Sequential::value()) {
                continue;
            } else {
                return false;
            }
        }
    }
    if (!is_gpu_scope) {
        return false;
    }

    // Criterion: Container is contiguous (Maybe can be relaxed later)
    auto& type_analysis = analysis_manager.get<analysis::TypeAnalysis>();
    auto type = type_analysis.get_outer_type(container);
    auto& peeled_type = types::peel_to_innermost_element(*type);
    if (peeled_type.type_id() == types::TypeID::Pointer) {
        return false;
    }

    auto& inner_body = loop.root();
    auto& users = analysis_manager.get<analysis::Users>();
    analysis::UsersView inner_body_users(users, inner_body);

    // Criterion: Container is read-only
    if (!inner_body_users.writes(container).empty() || !inner_body_users.views(container).empty() ||
        !inner_body_users.moves(container).empty()) {
        return false;
    }
    if (inner_body_users.reads(container).empty()) {
        return false;
    }

    // Criterion: Memory accesses do not depend on moving symbols
    for (auto& user : inner_body_users.uses(container)) {
        auto& subsets = user->subsets();
        for (auto& subset : subsets) {
            for (auto& expr : subset) {
                for (auto& atom : symbolic::atoms(expr)) {
                    if (SymEngine::is_a<SymEngine::Symbol>(*atom)) {
                        auto symbol = SymEngine::rcp_static_cast<const SymEngine::Symbol>(atom);
                        if (!inner_body_users.moves(symbol->get_name()).empty()) {
                            return false;
                        }
                    }
                }
            }
        }
    }

    // Limitations: single memory access
    if (inner_body_users.reads(container).size() != 1) {
        return false;
    }
    auto read = inner_body_users.reads(container).at(0);
    if (read->subsets().size() != 1) {
        return false;
    }
    auto subsets = read->subsets().at(0);

    // Criterion: more than one GPU dimension is available
    symbolic::SymbolVec indvars;
    for (auto node : ancestors) {
        if (auto ancestor_map = dynamic_cast<structured_control_flow::Map*>(node)) {
            auto schedule_type = ancestor_map->schedule_type();
            if (!gpu::is_gpu_schedule(schedule_type)) {
                continue;
            }
            indvars.push_back(ancestor_map->indvar());
        }
    }
    if (indvars.size() <= 1) {
        return false;
    }

    indvars.push_back(loop.indvar());

    // Criterion: Memory access is polynomial of
    // c_0 * a + c_1 * b + c_2 * c + c_3 * k, where a, b, c are x-threads, y-threads, z-threads
    // and k is the inner loop index
    for (auto subset : subsets) {
        if (symbolic::polynomial(subset, indvars) == SymEngine::null) {
            return false;
        }
    }

    // Criterion: inner indvar is used in memory access
    bool uses_inner_indvar = false;
    for (auto subset : subsets) {
        for (auto atom : symbolic::atoms(subset)) {
            if (symbolic::eq(atom, loop.indvar())) {
                uses_inner_indvar = true;
            }
        }
    }
    if (!uses_inner_indvar) {
        return false;
    }

    // Criterion: Containers in subset expressions are not written to in the loop
    for (auto subset : subsets) {
        for (auto atom : symbolic::atoms(subset)) {
            if (SymEngine::is_a<SymEngine::Symbol>(*atom)) {
                auto symbol = SymEngine::rcp_static_cast<const SymEngine::Symbol>(atom);
                if (!inner_body_users.writes(symbol->get_name()).empty()) {
                    return false;
                }
            }
        }
    }

    return true;
}

bool KernelLocalStorage::
    can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    if (!is_candidate(loop_, container_, builder, analysis_manager)) {
        return false;
    }

    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto ancestors = scope_analysis.ancestor_scopes(&loop_);

    // Criterion: Iteration count is known and an Integer
    symbolic::Integer iteration_count = get_iteration_count(loop_);
    if (iteration_count == SymEngine::null) {
        return false;
    }

    // Criterion: All block dimensions are known and an Integer
    auto [x_dim_size, y_dim_size, z_dim_size] = dim_size(ancestors);
    if (x_dim_size == SymEngine::null || y_dim_size == SymEngine::null || z_dim_size == SymEngine::null) {
        return false;
    }

    // Criterion: Has a free dimension to map to and that dimension is big enough
    auto& users = analysis_manager.get<analysis::Users>();
    analysis::UsersView inner_body_users(users, loop_.root());
    auto read = inner_body_users.reads(container_).at(0);
    auto subsets = read->subsets().at(0);

    auto [x_dim_available, y_dim_available, z_dim_available] = available_dims(subsets, analysis_manager);
    if (!x_dim_available && !y_dim_available && !z_dim_available) {
        return false;
    }

    return true;
};

void KernelLocalStorage::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& sdfg = builder.subject();

    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto ancestors = scope_analysis.ancestor_scopes(&loop_);

    auto& users = analysis_manager.get<analysis::Users>();

    auto& inner_body = this->loop_.root();
    analysis::UsersView inner_body_users(users, inner_body);

    // Detect GPU backend from ancestor map schedule types
    bool is_rocm = false;
    for (auto node : ancestors) {
        if (auto ancestor_map = dynamic_cast<structured_control_flow::Map*>(node)) {
            if (ancestor_map->schedule_type().value() == "ROCM") {
                is_rocm = true;
                break;
            }
        }
    }

    std::string thread_prefix = is_rocm ? "__daisy_hip_thread_idx_" : "__daisy_cuda_thread_idx_";
    std::string x_name = thread_prefix + "x";
    std::string y_name = thread_prefix + "y";
    std::string z_name = thread_prefix + "z";
    symbolic::Symbol x_symbol = symbolic::symbol(x_name);
    symbolic::Symbol y_symbol = symbolic::symbol(y_name);
    symbolic::Symbol z_symbol = symbolic::symbol(z_name);

    auto index_type = types::Scalar(types::PrimitiveType::Int32);
    index_type.storage_type(types::StorageType::NV_Symbol());

    std::set<std::string> containers(sdfg.containers().begin(), sdfg.containers().end());
    if (containers.find(x_name) == containers.end()) {
        builder.add_container(x_name, index_type);
    }
    if (containers.find(y_name) == containers.end()) {
        builder.add_container(y_name, index_type);
    }
    if (containers.find(z_name) == containers.end()) {
        builder.add_container(z_name, index_type);
    }

    /**
        1. Add new shared memory container
        2. Add barrier before loop
        3. add copyin branch before loop
        4. Add barrier before loop
        5. replace container in loop
        6. replace subset expressions in loop
    */

    symbolic::Integer iteration_count = get_iteration_count(loop_);

    auto [x_dim_size, y_dim_size, z_dim_size] = dim_size(ancestors);
    auto [x_dim_indvar, y_dim_indvar, z_dim_indvar] = dim_indvars(ancestors);

    auto parent = scope_analysis.parent_scope(&loop_);
    auto parent_seq = static_cast<structured_control_flow::Sequence*>(parent);
    auto& seq = builder.add_sequence_before(*parent_seq, loop_, {}, loop_.debug_info());

    // 1. Add new shared memory container
    auto& type_analysis = analysis_manager.get<analysis::TypeAnalysis>();
    auto type = type_analysis.get_outer_type(container_);
    auto& peeled_type = types::peel_to_innermost_element(*type);
    auto read = inner_body_users.reads(this->container_).at(0);
    auto subsets = read->subsets().at(0);

    auto [x_dim_available, y_dim_available, z_dim_available] = available_dims(subsets, analysis_manager);

    // get free dim
    symbolic::Symbol target_dim;
    auto [dim_x, dim_y, dim_z] = available_dims(subsets, analysis_manager);

    if (dim_x) {
        target_dim = x_symbol;
    } else if (dim_y) {
        target_dim = y_symbol;
    } else if (dim_z) {
        target_dim = z_symbol;
    } else {
        throw InvalidSDFGException("No available GPU tiling dimension found!");
    }

    // std::unique_ptr<types::IType> element_type;

    // if (peeled_type.type_id() == types::TypeID::Structure) {
    //     auto struct_type = static_cast<const types::Structure&>(peeled_type);
    //     types::Structure new_struct_type(
    //         types::StorageType::NV_Shared(), 8, {}, struct_type.name()
    //     );
    //     element_type = new_struct_type.clone();
    // } else if (peeled_type.type_id() == types::TypeID::Scalar) {
    //     auto scalar_type = static_cast<const types::Scalar&>(peeled_type);
    //     types::Scalar new_scalar_type(
    //         types::StorageType::NV_Shared(), 8, {}, scalar_type.primitive_type()
    //     );
    //     element_type = new_scalar_type.clone();
    // } else {
    //     throw InvalidSDFGException(
    //         "Unsupported peeled type for KernelLocalStorage."
    //     );
    // }

    auto generic_storage = is_rocm ? types::StorageType("AMD_Generic") : types::StorageType::NV_Generic();

    types::Array tile_array_type(types::StorageType::NV_Shared(), 8, {}, peeled_type, iteration_count);
    types::Array z_array_type(generic_storage, 8, {}, tile_array_type, z_dim_size);
    types::Array* pred_y;
    if (symbolic::eq(target_dim, z_symbol)) {
        pred_y = &tile_array_type;
    } else {
        pred_y = &z_array_type;
    }
    types::Array y_array_type(generic_storage, 8, {}, *pred_y, y_dim_size);
    types::Array* pred_x;
    if (symbolic::eq(target_dim, y_symbol)) {
        pred_x = &z_array_type;
    } else {
        pred_x = &y_array_type;
    }
    types::Array x_array_type(generic_storage, 8, {}, *pred_x, x_dim_size);
    types::Array* final_type;
    if (symbolic::eq(target_dim, x_symbol)) {
        final_type = &y_array_type;
    } else {
        final_type = &x_array_type;
    }

    std::string shared_container_name = "__daisy_shared_" + container_;
    builder.add_container(shared_container_name, *final_type);

    // 2. Add barrier before loop
    auto& sync_block1 = builder.add_block(seq);

    builder.add_library_node<data_flow::BarrierLocalNode>(sync_block1, {});

    // 3. add copyin branch before loop
    auto& if_else = builder.add_if_else(seq);

    auto condition = symbolic::subs(loop_.condition(), loop_.indvar(), symbolic::add(target_dim, offset_));
    auto& branch = builder.add_case(if_else, condition);

    auto& copyin_block = builder.add_block(branch);

    auto& access_in = builder.add_access(copyin_block, container_);
    auto& access_out = builder.add_access(copyin_block, shared_container_name);

    auto& tasklet = builder.add_tasklet(copyin_block, data_flow::TaskletCode::assign, "out_", {"in_"});

    std::vector<symbolic::Expression> copyin_subsets;
    for (auto subset : subsets) {
        auto substituted = symbolic::subs(subset, loop_.indvar(), symbolic::add(target_dim, offset_));
        copyin_subsets.push_back(substituted);
    }

    builder.add_computational_memlet(copyin_block, access_in, tasklet, "in_", copyin_subsets, *type);

    std::vector<symbolic::Expression> shared_access_subsets = {x_symbol, y_symbol, z_symbol, target_dim};

    if (symbolic::eq(target_dim, x_symbol)) {
        shared_access_subsets.erase(shared_access_subsets.begin());
    } else if (symbolic::eq(target_dim, y_symbol)) {
        shared_access_subsets.erase(shared_access_subsets.begin() + 1);
    } else if (symbolic::eq(target_dim, z_symbol)) {
        shared_access_subsets.erase(shared_access_subsets.begin() + 2);
    }

    builder.add_computational_memlet(copyin_block, tasklet, "out_", access_out, shared_access_subsets);

    // 4. Add barrier before loop

    auto& sync_block2 = builder.add_block(seq);

    builder.add_library_node<data_flow::BarrierLocalNode>(sync_block2, {});

    // 5. replace container in loop
    loop_.replace(symbolic::symbol(container_), symbolic::symbol(shared_container_name));

    // 6. replace subset expressions in loop
    std::vector<symbolic::Expression> read_shared_access_subsets;
    symbolic::Expression substituted_dimension;
    for (auto& subset : shared_access_subsets) {
        auto substituted = symbolic::subs(subset, target_dim, symbolic::sub(loop_.indvar(), offset_));
        read_shared_access_subsets.push_back(substituted);
    }

    auto access_node = static_cast<data_flow::AccessNode*>(read->element());
    for (auto& oedge : access_node->get_parent().out_edges(*access_node)) {
        oedge.set_subset(read_shared_access_subsets);
        oedge.set_base_type(*final_type);
    }

    // End of transformation

    passes::SequenceFusion sf_pass;
    passes::DeadCFGElimination dce_pass;
    passes::TrivialArrayElimination tae_pass;
    bool applies = false;
    do {
        applies = false;
        applies |= dce_pass.run(builder, analysis_manager);
        applies |= sf_pass.run(builder, analysis_manager);
        applies |= tae_pass.run(builder, analysis_manager);
    } while (applies);
};

void KernelLocalStorage::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();

    std::string loop_type;
    if (dynamic_cast<structured_control_flow::For*>(&loop_)) {
        loop_type = "for";
    } else if (dynamic_cast<structured_control_flow::While*>(&loop_)) {
        loop_type = "while";
    } else if (dynamic_cast<structured_control_flow::Map*>(&loop_)) {
        loop_type = "map";
    } else {
        loop_type = "unknown";
    }

    j["subgraph"] = {{"0", {{"element_id", this->loop_.element_id()}, {"type", loop_type}}}};

    j["parameters"] = {{"offset", serializer::JSONSerializer::expression(offset_)}, {"container", this->container_}};

    // Legacy fields for backward compatibility
    j["loop_element_id"] = this->loop_.element_id();
    j["offset"] = serializer::JSONSerializer::expression(offset_);
    j["container"] = this->container_;
};

KernelLocalStorage KernelLocalStorage::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc) {
    size_t loop_id;
    if (desc.contains("subgraph")) {
        const auto& node_desc = desc.at("subgraph").at("0");
        loop_id = node_desc.at("element_id").get<size_t>();
    } else {
        loop_id = desc.at("loop_element_id").get<size_t>();
    }

    auto element = builder.find_element_by_id(loop_id);
    if (!element) {
        throw InvalidTransformationDescriptionException("Element with ID " + std::to_string(loop_id) + " not found.");
    }
    auto outer_loop = dynamic_cast<structured_control_flow::For*>(element);

    nlohmann::json offset_json;
    std::string container;
    if (desc.contains("parameters")) {
        const auto& params = desc.at("parameters");
        if (params.contains("offset")) {
            offset_json = params.at("offset");
        }
        if (params.contains("container")) {
            container = params.at("container").get<std::string>();
        }
    }
    if (offset_json.is_null() && desc.contains("offset")) {
        offset_json = desc.at("offset");
    }
    if (container.empty() && desc.contains("container")) {
        container = desc.at("container").get<std::string>();
    }

    auto offset = symbolic::parse(offset_json);

    return KernelLocalStorage(*outer_loop, offset, container);
};

} // namespace transformations
} // namespace sdfg
