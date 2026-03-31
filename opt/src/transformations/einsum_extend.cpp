#include "sdfg/transformations/einsum_extend.h"

#include <cstddef>
#include <nlohmann/json_fwd.hpp>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/data_flow_node.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/einsum/einsum.h"
#include "sdfg/element.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/transformations/transformation.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace transformations {

EinsumExtend::EinsumExtend(einsum::EinsumNode& einsum_node) : einsum_node_(einsum_node) {}

std::string EinsumExtend::name() const { return "EinsumExtend"; }

bool EinsumExtend::can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    // Skip EinsumNodes with dimensions
    if (this->einsum_node_.dims().size() > 0) {
        return false;
    }

    size_t muls = 0;
    auto& dfg = this->einsum_node_.get_parent();
    for (auto& iedge : dfg.in_edges(this->einsum_node_)) {
        // Skip reduction container (connector "__einsum_out")
        if (iedge.dst_conn() == this->einsum_node_.inputs().back()) {
            continue;
        }

        // Skip constant nodes and access nodes without in edges
        auto& access_node = static_cast<data_flow::AccessNode&>(iedge.src());
        if (dynamic_cast<data_flow::ConstantNode*>(&access_node) || dfg.in_degree(access_node) == 0) {
            continue;
        }

        // Count the multiplication tasklets whose output access nodes are input access nodes of the EinsumNode
        for (auto& access_node_iedge : dfg.in_edges(access_node)) {
            auto* tasklet = dynamic_cast<data_flow::Tasklet*>(&access_node_iedge.src());
            if (tasklet && tasklet->code() == data_flow::TaskletCode::fp_mul) {
                muls++;
            }
        }
    }

    return muls > 0;
}

void EinsumExtend::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& dfg = this->einsum_node_.get_parent();
    auto* block = dynamic_cast<structured_control_flow::Block*>(dfg.get_parent());
    assert(block);

    // Construct inputs, in indices, and a map for new in edges
    std::vector<std::string> inputs;
    std::vector<data_flow::Subset> in_indices;
    std::unordered_map<std::string, std::tuple<data_flow::AccessNode&, const types::IType&, DebugInfo>> new_iedges_map;
    std::unordered_set<data_flow::Memlet*> memlets_for_removal;
    std::unordered_set<data_flow::DataFlowNode*> nodes_for_removal;
    DebugInfo new_deb_info(this->einsum_node_.debug_info());
    for (size_t i = 0; i < this->einsum_node_.inputs().size() - 1; i++) {
        auto& conn = this->einsum_node_.input(i);

        // Find corresponding in edge
        data_flow::Memlet* iedge = nullptr;
        for (auto& in_edge : dfg.in_edges(this->einsum_node_)) {
            if (in_edge.dst_conn() == conn) {
                iedge = &in_edge;
                break;
            }
        }
        assert(iedge);

        // Check if at the access node there is a multiplication tasklet
        auto& access_node = static_cast<data_flow::AccessNode&>(iedge->src());
        data_flow::Tasklet* tasklet = nullptr;
        data_flow::Memlet* tasklet_oedge = nullptr;
        if (!dynamic_cast<data_flow::ConstantNode*>(&access_node) && dfg.in_degree(access_node) > 0) {
            for (auto& access_node_iedge : dfg.in_edges(access_node)) {
                auto* tmp_taskelt = dynamic_cast<data_flow::Tasklet*>(&access_node_iedge.src());
                if (tmp_taskelt && tmp_taskelt->code() == data_flow::TaskletCode::fp_mul) {
                    tasklet = tmp_taskelt;
                    tasklet_oedge = &access_node_iedge;
                    break;
                }
            }
        }

        // Fill the data ...
        if (tasklet) {
            // ... with new access nodes and connectors
            for (auto& tasklet_iedge : dfg.in_edges(*tasklet)) {
                auto& tasklet_access_node = static_cast<data_flow::AccessNode&>(tasklet_iedge.src());
                std::string new_conn = iedge->dst_conn() + tasklet_iedge.dst_conn();
                inputs.push_back(new_conn);
                in_indices.push_back(tasklet_iedge.subset());
                new_iedges_map.insert(
                    {new_conn,
                     {tasklet_access_node,
                      tasklet_iedge.base_type(),
                      DebugInfo::merge(iedge->debug_info(), tasklet_iedge.debug_info())}}
                );
                memlets_for_removal.insert(&tasklet_iedge);
            }

            // Mark tasklet and its memlets for removal
            memlets_for_removal.insert(tasklet_oedge);
            new_deb_info = DebugInfo::merge(new_deb_info, tasklet->debug_info());
            nodes_for_removal.insert(tasklet);

            // Mark acess node for removal if not used elsewhere
            if (dfg.in_degree(access_node) == 1 && dfg.out_degree(access_node) == 1) {
                nodes_for_removal.insert(&access_node);
            }
        } else {
            // ... with the old stuff
            inputs.push_back(conn);
            in_indices.push_back(this->einsum_node_.in_indices(i));
            new_iedges_map.insert({conn, {access_node, iedge->base_type(), iedge->debug_info()}});
        }

        // Mark in edge for removal
        memlets_for_removal.insert(iedge);
    }

    // Special handling for the reduction input
    {
        auto& conn = this->einsum_node_.inputs().back();

        // Find corresponding in edge
        data_flow::Memlet* iedge = nullptr;
        for (auto& in_edge : dfg.in_edges(this->einsum_node_)) {
            if (in_edge.dst_conn() == conn) {
                iedge = &in_edge;
                break;
            }
        }
        assert(iedge);

        // Mapping and marking for removal
        auto& access_node = static_cast<data_flow::AccessNode&>(iedge->src());
        new_iedges_map.insert({conn, {access_node, iedge->base_type(), iedge->debug_info()}});
        memlets_for_removal.insert(iedge);
    }

    // Create new einsum node
    auto& new_libnode = builder.add_library_node<
        einsum::EinsumNode,
        const std::vector<std::string>&,
        const std::vector<einsum::EinsumDimension>&,
        const data_flow::Subset&,
        const std::vector<
            data_flow::Subset>&>(*block, new_deb_info, inputs, {}, this->einsum_node_.out_indices(), in_indices);
    auto& new_einsum_node = static_cast<einsum::EinsumNode&>(new_libnode);

    // Construct in edges
    for (auto& conn : new_einsum_node.inputs()) {
        auto [access_node, type, deb_info] = new_iedges_map.at(conn);
        builder.add_memlet(*block, access_node, "void", new_libnode, conn, {}, type, deb_info);
    }

    // Remove marked memlets & nodes
    for (auto* memlet : memlets_for_removal) {
        builder.remove_memlet(*block, *memlet);
    }
    for (auto* node : nodes_for_removal) {
        builder.remove_node(*block, *node);
    }

    // Redirect out edges
    while (dfg.out_edges(this->einsum_node_).begin() != dfg.out_edges(this->einsum_node_).end()) {
        auto& oedge = *dfg.out_edges(this->einsum_node_).begin();
        builder.add_memlet(
            *block,
            new_libnode,
            oedge.src_conn(),
            oedge.dst(),
            oedge.dst_conn(),
            oedge.subset(),
            oedge.base_type(),
            oedge.debug_info()
        );
        builder.remove_memlet(*block, oedge);
    }

    // Remove old einsum node
    builder.remove_node(*block, this->einsum_node_);

    analysis_manager.invalidate_all();
}

void EinsumExtend::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["einsum_node_element_id"] = this->einsum_node_.element_id();
}

EinsumExtend EinsumExtend::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j) {
    assert(j.contains("einsum_node_element_id"));
    assert(j["einsum_node_element_id"].is_number_unsigned());
    size_t einsum_node_id = j["einsum_node_element_id"].get<size_t>();
    auto* einsum_node_element = builder.find_element_by_id(einsum_node_id);
    if (!einsum_node_element) {
        throw InvalidTransformationDescriptionException(
            "Element with ID " + std::to_string(einsum_node_id) + " not found"
        );
    }
    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(einsum_node_element);
    if (!einsum_node) {
        throw InvalidTransformationDescriptionException(
            "Element with ID " + std::to_string(einsum_node_id) + " is not an EinsumNode"
        );
    }

    return EinsumExtend(*einsum_node);
}

} // namespace transformations
} // namespace sdfg
