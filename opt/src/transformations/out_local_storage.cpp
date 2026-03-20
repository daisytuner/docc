#include "sdfg/transformations/out_local_storage.h"

#include <cassert>
#include <cstddef>
#include <string>

#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/analysis/type_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/passes/structured_control_flow/dead_cfg_elimination.h"
#include "sdfg/passes/structured_control_flow/sequence_fusion.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/transformations/utils.h"
#include "sdfg/types/array.h"
#include "sdfg/types/scalar.h"

namespace sdfg {
namespace transformations {

OutLocalStorage::OutLocalStorage(structured_control_flow::StructuredLoop& loop, std::string container)
    : loop_(loop), container_(container) {};

std::string OutLocalStorage::name() const { return "OutLocalStorage"; };

bool OutLocalStorage::can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& body = this->loop_.root();
    this->requires_array_ = false;

    // Criterion: Check if container exists and is used in the loop
    auto& users = analysis_manager.get<analysis::Users>();
    analysis::UsersView body_users(users, body);
    if (body_users.uses(this->container_).size() == 0) {
        return false;
    }

    // Criterion: Check if all accesses to the container within the loop are identical
    auto accesses = body_users.uses(this->container_);
    auto first_access = accesses.at(0);
    auto first_subset = first_access->subsets().at(0);

    if (accesses.size() > 1) {
        for (auto access : accesses) {
            if (first_access->subsets().size() != access->subsets().size()) {
                return false;
            }
            for (size_t i = 0; i < first_access->subsets().size(); i++) {
                auto subset = access->subsets().at(i);
                if (first_subset.size() != subset.size()) {
                    return false;
                }
                for (size_t j = 0; j < first_subset.size(); j++) {
                    if (!symbolic::eq(first_subset.at(j), subset.at(j))) {
                        return false;
                    }
                }
            }
        }
    }

    // Criterion: Check if accesses do not depend on containers written in the loop
    auto writes = body_users.writes();
    symbolic::SymbolSet written_containers;
    for (auto write : writes) {
        written_containers.insert(symbolic::symbol(write->container()));
    }
    for (auto subset : first_access->subsets()) {
        for (auto access : subset) {
            for (auto atom : symbolic::atoms(access)) {
                if (written_containers.contains(atom)) {
                    return false;
                }
            }
        }
    }

    // Soft Criterion: Check if the accesses do not depend on the loop iteration
    // Decide if an array or scalar is required
    for (auto subset : first_access->subsets()) {
        for (auto access : subset) {
            for (auto atom : symbolic::atoms(access)) {
                if (symbolic::eq(atom, this->loop_.indvar())) {
                    this->requires_array_ = true;
                    break;
                }
            }
            if (this->requires_array_) {
                break;
            }
        }
        if (this->requires_array_) {
            break;
        }
    }

    // Criterion: Check if the loop iteration count is known and an Integer when an array is
    // required
    if (this->requires_array_) {
        auto iteration_count = get_iteration_count(this->loop_);
        if (iteration_count == SymEngine::null) {
            return false;
        }
    }

    return true;
};

void OutLocalStorage::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    if (requires_array_) {
        apply_array(builder, analysis_manager);
    } else {
        apply_scalar(builder, analysis_manager);
    }

    // End of transformation

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

void OutLocalStorage::apply_array(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& sdfg = builder.subject();
    auto& users = analysis_manager.get<analysis::Users>();
    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto parent_node = scope_analysis.parent_scope(&loop_);
    auto parent = dynamic_cast<structured_control_flow::Sequence*>(parent_node);
    if (!parent) {
        throw InvalidSDFGException("OutLocalStorage: Parent of loop must be a Sequence!");
    }

    analysis::TypeAnalysis type_analysis(sdfg, &loop_, analysis_manager);
    auto type = type_analysis.get_outer_type(container_);

    auto replacement_name = "__daisy_out_local_storage_" + this->container_;

    auto iteration_count = get_iteration_count(this->loop_);
    types::Scalar scalar_type(type->primitive_type());
    types::Array array_type(scalar_type, iteration_count);
    builder.add_container(replacement_name, array_type);

    auto indvar_name = "__daisy_out_local_storage_" + this->loop_.indvar()->get_name();
    types::Scalar indvar_type(sdfg.type(loop_.indvar()->get_name()).primitive_type());
    builder.add_container(indvar_name, indvar_type);
    auto indvar = symbolic::symbol(indvar_name);
    auto init = loop_.init();
    auto update = symbolic::subs(loop_.update(), loop_.indvar(), indvar);
    auto condition = symbolic::subs(loop_.condition(), loop_.indvar(), indvar);

    analysis::UsersView body_users(users, loop_.root());
    auto accesses = body_users.uses(this->container_);
    auto first_access = accesses.at(0);
    auto first_subset = first_access->subsets().at(0);
    auto& init_loop = builder.add_for_before(*parent, loop_, indvar, condition, init, update, {}, loop_.debug_info());
    auto& init_body = init_loop.root();
    auto& init_block = builder.add_block(init_body);
    auto& init_access_read = builder.add_access(init_block, this->container_);
    auto& init_access_write = builder.add_access(init_block, replacement_name);
    auto& init_tasklet = builder.add_tasklet(init_block, data_flow::TaskletCode::assign, "_out", {"_in"});
    auto& init_memlet_in =
        builder.add_computational_memlet(init_block, init_access_read, init_tasklet, "_in", first_subset, *type);
    init_memlet_in.replace(loop_.indvar(), indvar);
    builder.add_computational_memlet(init_block, init_tasklet, "_out", init_access_write, {indvar}, array_type);

    auto& reset_loop = builder.add_for_after(*parent, loop_, indvar, condition, init, update, {}, loop_.debug_info());
    auto& reset_body = reset_loop.root();
    auto& reset_block = builder.add_block(reset_body);
    auto& reset_access_read = builder.add_access(reset_block, replacement_name);
    auto& reset_access_write = builder.add_access(reset_block, this->container_);
    auto& reset_tasklet = builder.add_tasklet(reset_block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(reset_block, reset_access_read, reset_tasklet, "_in", {indvar}, array_type);
    auto& reset_memlet_out =
        builder.add_computational_memlet(reset_block, reset_tasklet, "_out", reset_access_write, first_subset, *type);
    reset_memlet_out.replace(loop_.indvar(), indvar);

    for (auto user : body_users.uses(this->container_)) {
        auto element = user->element();
        if (auto memlet = dynamic_cast<data_flow::Memlet*>(element)) {
            auto subset = memlet->subset();
            subset.clear();
            subset.push_back(this->loop_.indvar());
            memlet->set_subset(subset);
        }
    }

    for (auto user : body_users.uses(this->container_)) {
        auto element = user->element();
        if (auto access = dynamic_cast<data_flow::AccessNode*>(element)) {
            for (auto& iedge : access->get_parent().in_edges(*access)) {
                auto memlet = &iedge;
                auto subset = memlet->subset();
                subset.clear();
                subset.push_back(this->loop_.indvar());
                memlet->set_subset(subset);
                memlet->set_base_type(array_type);
            }
            for (auto& oedge : access->get_parent().out_edges(*access)) {
                auto memlet = &oedge;
                auto subset = memlet->subset();
                subset.clear();
                subset.push_back(this->loop_.indvar());
                memlet->set_subset(subset);
                memlet->set_base_type(array_type);
            }
        }
    }

    loop_.replace(symbolic::symbol(this->container_), symbolic::symbol(replacement_name));
};

void OutLocalStorage::apply_scalar(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& sdfg = builder.subject();
    auto& users = analysis_manager.get<analysis::Users>();
    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto parent_node = scope_analysis.parent_scope(&loop_);
    auto parent = dynamic_cast<structured_control_flow::Sequence*>(parent_node);
    if (!parent) {
        throw InvalidSDFGException("OutLocalStorage: Parent of loop must be a Sequence!");
    }

    analysis::TypeAnalysis type_analysis(sdfg, &loop_, analysis_manager);
    auto type = type_analysis.get_outer_type(container_);

    auto replacement_name = "__daisy_out_local_storage_" + this->container_;

    types::Scalar scalar_type(type->primitive_type());
    builder.add_container(replacement_name, scalar_type);

    analysis::UsersView body_users(users, loop_.root());
    auto accesses = body_users.uses(this->container_);
    auto first_access = accesses.at(0);
    auto first_subset = first_access->subsets().at(0);
    auto& init_block = builder.add_block_before(*parent, loop_, {}, loop_.debug_info());
    auto& init_access_read = builder.add_access(init_block, this->container_);
    auto& init_access_write = builder.add_access(init_block, replacement_name);
    auto& init_tasklet = builder.add_tasklet(init_block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(init_block, init_access_read, init_tasklet, "_in", first_subset, *type);
    builder.add_computational_memlet(init_block, init_tasklet, "_out", init_access_write, {}, scalar_type);

    auto& reset_block = builder.add_block_after(*parent, loop_, {}, loop_.debug_info());
    auto& reset_access_read = builder.add_access(reset_block, replacement_name);
    auto& reset_access_write = builder.add_access(reset_block, this->container_);
    auto& reset_tasklet = builder.add_tasklet(reset_block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(reset_block, reset_access_read, reset_tasklet, "_in", {}, scalar_type);
    builder.add_computational_memlet(reset_block, reset_tasklet, "_out", reset_access_write, first_subset, *type);

    for (auto user : body_users.uses(this->container_)) {
        auto element = user->element();
        if (auto access = dynamic_cast<data_flow::AccessNode*>(element)) {
            for (auto& iedge : access->get_parent().in_edges(*access)) {
                auto memlet = &iedge;
                auto subset = memlet->subset();
                subset.clear();
                memlet->set_subset(subset);
                memlet->set_base_type(scalar_type);
            }
            for (auto& oedge : access->get_parent().out_edges(*access)) {
                auto memlet = &oedge;
                auto subset = memlet->subset();
                subset.clear();
                memlet->set_subset(subset);
                memlet->set_base_type(scalar_type);
            }
        }
    }

    this->loop_.replace(symbolic::symbol(this->container_), symbolic::symbol(replacement_name));
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
    j["subgraph"] = {{"0", {{"element_id", this->loop_.element_id()}, {"type", loop_type}}}};
    j["transformation_type"] = this->name();
    j["container"] = container_;
};

OutLocalStorage OutLocalStorage::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc) {
    auto loop_id = desc["subgraph"]["0"]["element_id"].get<size_t>();
    std::string container = desc["container"].get<std::string>();
    auto element = builder.find_element_by_id(loop_id);
    if (!element) {
        throw InvalidTransformationDescriptionException("Element with ID " + std::to_string(loop_id) + " not found.");
    }
    auto loop = dynamic_cast<structured_control_flow::StructuredLoop*>(element);

    return OutLocalStorage(*loop, container);
};

} // namespace transformations
} // namespace sdfg
