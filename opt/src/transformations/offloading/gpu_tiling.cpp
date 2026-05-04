#include "sdfg/transformations/offloading/gpu_tiling.h"
#include <set>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/type_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/passes/offloading/sync_condition_propagation.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/structured_control_flow/while.h"
#include "sdfg/transformations/loop_tiling.h"
#include "sdfg/transformations/offloading/kernel_local_storage.h"
#include "sdfg/transformations/out_local_storage.h"


namespace sdfg {
namespace transformations {

GPUTiling::GPUTiling(structured_control_flow::StructuredLoop& loop, size_t size) : loop_(loop), size_(size) {}

std::string GPUTiling::name() const { return "GPUTiling"; }

bool GPUTiling::can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto sdfg = builder.subject().clone();
    builder::StructuredSDFGBuilder builder_local(sdfg);
    analysis::AnalysisManager analysis_manager_local(builder_local.subject());

    auto test_loop = builder_local.find_element_by_id(loop_.element_id());

    auto& loop_local = static_cast<structured_control_flow::StructuredLoop&>(*test_loop);

    LoopTiling tiling(loop_local, size_);

    if (!tiling.can_be_applied(builder_local, analysis_manager_local)) {
        return false;
    }

    tiling.apply(builder_local, analysis_manager_local);

    auto& users = analysis_manager_local.get<analysis::Users>();

    auto inner_loop = tiling.inner_loop();
    auto outer_loop = tiling.outer_loop();

    analysis::UsersView users_view(users, inner_loop->root());

    std::set<data_flow::AccessNode*> read_accesses;
    std::set<std::string> read_container_names;

    for (auto read : users_view.reads()) {
        if (!dynamic_cast<data_flow::AccessNode*>(read->element())) {
            continue;
        }
        if (read_container_names.find(read->container()) != read_container_names.end()) {
            continue;
        }
        auto access_node = dynamic_cast<data_flow::AccessNode*>(read->element());
        read_container_names.insert(read->container());
        read_accesses.insert(std::move(access_node));
    }

    if (read_accesses.empty()) {
        return false;
    }

    for (auto& access : read_accesses) {
        KernelLocalStorage kls(*inner_loop, outer_loop->indvar(), *access);
        if (kls.can_be_applied(builder_local, analysis_manager_local)) {
            target_containers_.insert(access->data());
        }
    }
    if (target_containers_.empty()) {
        return false;
    }

    return true;
}

void GPUTiling::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    LoopTiling tiling(loop_, size_);
    tiling.apply(builder, analysis_manager);
    auto inner_loop = tiling.inner_loop();
    auto outer_loop = tiling.outer_loop();

    analysis::UsersView users_view(analysis_manager.get<analysis::Users>(), inner_loop->root());

    for (const auto& container_name : target_containers_) {
        auto& users = analysis_manager.get<analysis::Users>();
        analysis::UsersView users_view(users, inner_loop->root());

        auto reads = users_view.reads(container_name);
        if (reads.empty()) continue;

        auto* access = dynamic_cast<data_flow::AccessNode*>(reads.front()->element());
        if (!access) continue;

        KernelLocalStorage kls(*inner_loop, outer_loop->indvar(), *access);
        if (kls.can_be_applied(builder, analysis_manager)) {
            kls.apply(builder, analysis_manager);
        }
    }

    passes::SyncConditionPropagation sync_condition_propagation;
    sync_condition_propagation.run_pass(builder, analysis_manager);

    applied_ = true;
    inner_loop_ = inner_loop;
    outer_loop_ = outer_loop;

    auto& users = analysis_manager.get<analysis::Users>();
    analysis::UsersView users_view_inner_loop(users, inner_loop->root());

    analysis::TypeAnalysis type_analysis(builder.subject(), inner_loop, analysis_manager);
    for (const auto& write : users_view_inner_loop.writes()) {
        if (!dynamic_cast<data_flow::AccessNode*>(write->element())) {
            continue;
        }
        if (type_analysis.get_outer_type(write->container())->type_id() == types::TypeID::Scalar ||
            type_analysis.get_outer_type(write->container())->type_id() == types::TypeID::Structure) {
            continue;
        }
        auto access_node = dynamic_cast<data_flow::AccessNode*>(write->element());
        OutLocalStorage ols(*inner_loop, *access_node);
        if (ols.can_be_applied(builder, analysis_manager)) {
            ols.apply(builder, analysis_manager);
        }
    }
}

void GPUTiling::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();

    // Determine loop type consistent with GNN feature extractor labelling.
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

    // Embedding-compatible description used by EmbeddingRecorder/EmbeddingReplayer.
    j["subgraph"] = {{"0", {{"element_id", this->loop_.element_id()}, {"type", loop_type}}}};
    j["parameters"] = {{"size", this->size_}};

    // Legacy fields for backward compatibility.
    j["loop_element_id"] = this->loop_.element_id();
    j["size"] = this->size_;
}

GPUTiling GPUTiling::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j) {
    // Prefer embedding-compatible representation, but support legacy layout.
    size_t loop_id;
    if (j.contains("subgraph")) {
        const auto& node_desc = j.at("subgraph").at("0");
        loop_id = node_desc.at("element_id").get<size_t>();
    } else {
        loop_id = j.at("loop_element_id").get<size_t>();
    }

    auto element = builder.find_element_by_id(loop_id);
    if (!element) {
        throw InvalidTransformationDescriptionException("Element with ID " + std::to_string(loop_id) + " not found.");
    }
    auto loop = dynamic_cast<structured_control_flow::StructuredLoop*>(element);

    size_t size;
    if (j.contains("parameters") && j.at("parameters").contains("size")) {
        size = j.at("parameters").at("size").get<size_t>();
    } else {
        size = j.at("size").get<size_t>();
    }

    return GPUTiling(*loop, size);
}

structured_control_flow::StructuredLoop* GPUTiling::inner_loop() {
    if (!applied_) {
        throw InvalidSDFGException("Accessing tiled loop before their creation.");
    }

    return inner_loop_;
}

structured_control_flow::StructuredLoop* GPUTiling::outer_loop() {
    if (!applied_) {
        throw InvalidSDFGException("Accessing tiled loop before their creation.");
    }

    return outer_loop_;
}

} // namespace transformations
} // namespace sdfg
