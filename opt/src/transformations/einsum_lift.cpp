#include "sdfg/transformations/einsum_lift.h"

#include <cstddef>
#include <nlohmann/json_fwd.hpp>
#include <string>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/einsum/einsum.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/transformations/transformation.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace transformations {

bool EinsumLift::subsets_eq(const data_flow::Subset& subset1, const data_flow::Subset& subset2) {
    if (subset1.size() != subset2.size()) {
        return false;
    }
    for (size_t i = 0; i < subset1.size(); i++) {
        if (!symbolic::eq(subset1.at(i), subset2.at(i))) {
            return false;
        }
    }
    return true;
}

EinsumLift::EinsumLift(data_flow::Tasklet& tasklet) : tasklet_(tasklet) {}

std::string EinsumLift::name() const { return "EinsumLift"; }

bool EinsumLift::can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& dfg = this->tasklet_.get_parent();

    if (this->tasklet_.code() == data_flow::TaskletCode::fp_add) {
        // Get the output container
        auto& oedge = *dfg.out_edges(this->tasklet_).begin();
        auto& out = static_cast<data_flow::AccessNode&>(oedge.dst());
        auto& out_container = out.data();

        // Check that one of the input containers is the same as the output container
        for (auto& iedge : dfg.in_edges(this->tasklet_)) {
            if (dynamic_cast<data_flow::ConstantNode*>(&iedge.src())) {
                continue;
            }
            auto& in = static_cast<data_flow::AccessNode&>(iedge.src());
            if (in.data() == out_container && this->subsets_eq(oedge.subset(), iedge.subset())) {
                return true;
            }
        }
    } else if (this->tasklet_.code() == data_flow::TaskletCode::fp_fma) {
        // Get the output container
        auto& oedge = *dfg.out_edges(this->tasklet_).begin();
        auto& out = static_cast<data_flow::AccessNode&>(oedge.dst());
        auto& out_container = out.data();

        // Check that there is a summation on the output container
        auto& reduction_conn = this->tasklet_.inputs().back();
        for (auto& iedge : dfg.in_edges(this->tasklet_)) {
            if (dynamic_cast<data_flow::ConstantNode*>(&iedge.src()) || iedge.dst_conn() != reduction_conn) {
                continue;
            }
            auto& in = static_cast<data_flow::AccessNode&>(iedge.src());
            if (in.data() == out_container && this->subsets_eq(oedge.subset(), iedge.subset())) {
                return true;
            }
        }
    } else if (this->tasklet_.code() == data_flow::TaskletCode::fp_sub) {
        // Get the output container
        auto& oedge = *dfg.out_edges(this->tasklet_).begin();
        auto& out = static_cast<data_flow::AccessNode&>(oedge.dst());
        auto& out_container = out.data();

        // Check that there is a reduction on the first input
        auto& reduction_conn = this->tasklet_.inputs().front();
        for (auto& iedge : dfg.in_edges(this->tasklet_)) {
            if (dynamic_cast<data_flow::ConstantNode*>(&iedge.src()) || iedge.dst_conn() != reduction_conn) {
                continue;
            }
            auto& in = static_cast<data_flow::AccessNode&>(iedge.src());
            if (in.data() == out_container && this->subsets_eq(oedge.subset(), iedge.subset())) {
                return true;
            }
        }
    }

    return false;
}

void EinsumLift::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    std::string reduction_conn;
    data_flow::Subset out_indices;
    std::vector<std::string> inputs;
    std::vector<data_flow::Subset> in_indices;
    bool subtraction = false;
    auto& dfg = this->tasklet_.get_parent();
    auto* block = dynamic_cast<structured_control_flow::Block*>(dfg.get_parent());
    assert(block);

    // Collect information
    if (this->tasklet_.code() == data_flow::TaskletCode::fp_add) {
        auto& oedge = *dfg.out_edges(this->tasklet_).begin();
        auto& out = static_cast<data_flow::AccessNode&>(oedge.dst());
        auto& out_container = out.data();
        out_indices = oedge.subset();

        for (auto& iedge : dfg.in_edges(this->tasklet_)) {
            auto& in = static_cast<data_flow::AccessNode&>(iedge.src());
            if (in.data() == out_container) {
                reduction_conn = iedge.dst_conn();
            } else {
                inputs.push_back(iedge.dst_conn());
                in_indices.push_back(iedge.subset());
            }
        }
    } else if (this->tasklet_.code() == data_flow::TaskletCode::fp_fma) {
        auto& oedge = *dfg.out_edges(this->tasklet_).begin();
        out_indices = oedge.subset();

        reduction_conn = this->tasklet_.inputs().back();
        for (auto& iedge : dfg.in_edges(this->tasklet_)) {
            if (iedge.dst_conn() != reduction_conn) {
                inputs.push_back(iedge.dst_conn());
                in_indices.push_back(iedge.subset());
            }
        }
    } else if (this->tasklet_.code() == data_flow::TaskletCode::fp_sub) {
        auto& oedge = *dfg.out_edges(this->tasklet_).begin();
        out_indices = oedge.subset();

        reduction_conn = this->tasklet_.inputs().front();
        for (auto& iedge : dfg.in_edges(this->tasklet_)) {
            if (iedge.dst_conn() != reduction_conn) {
                inputs.push_back(iedge.dst_conn());
                in_indices.push_back(iedge.subset());
            }
        }
        inputs.push_back("__einsum_const");
        in_indices.push_back({});
        subtraction = true;
    }

    // Create EinsumNode
    auto& libnode = builder.add_library_node<
        einsum::EinsumNode,
        const std::vector<std::string>&,
        const std::vector<einsum::EinsumDimension>&,
        const data_flow::Subset&,
        const std::vector<data_flow::Subset>&>(*block, this->tasklet_.debug_info(), inputs, {}, out_indices, in_indices);

    // Redirect in edges
    while (dfg.in_edges(this->tasklet_).begin() != dfg.in_edges(this->tasklet_).end()) {
        auto& iedge = *dfg.in_edges(this->tasklet_).begin();
        if (iedge.dst_conn() == reduction_conn) {
            builder.add_memlet(
                *block, iedge.src(), iedge.src_conn(), libnode, "__einsum_out", {}, iedge.base_type(), iedge.debug_info()
            );
        } else {
            builder.add_memlet(
                *block,
                iedge.src(),
                iedge.src_conn(),
                libnode,
                iedge.dst_conn(),
                {},
                iedge.base_type(),
                iedge.debug_info()
            );
        }
        builder.remove_memlet(*block, iedge);
    }

    // Add multiplication with -1
    if (subtraction) {
        types::Scalar float_type(types::PrimitiveType::Float);
        auto& einsum_const = builder.add_constant(*block, "-1.0", float_type);
        builder.add_computational_memlet(*block, einsum_const, libnode, "__einsum_const", {}, float_type);
    }

    // Redirect out edges
    while (dfg.out_edges(this->tasklet_).begin() != dfg.out_edges(this->tasklet_).end()) {
        auto& oedge = *dfg.out_edges(this->tasklet_).begin();
        builder.add_memlet(
            *block, libnode, "__einsum_out", oedge.dst(), oedge.dst_conn(), {}, oedge.base_type(), oedge.debug_info()
        );
        builder.remove_memlet(*block, oedge);
    }

    // Delete tasklet
    builder.remove_node(*block, this->tasklet_);

    analysis_manager.invalidate_all();
}

void EinsumLift::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["tasklet_element_id"] = this->tasklet_.element_id();
}

EinsumLift EinsumLift::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j) {
    assert(j.contains("tasklet_element_id"));
    assert(j["tasklet_element_id"].is_number_unsigned());
    size_t tasklet_id = j["tasklet_element_id"].get<size_t>();
    auto* tasklet_element = builder.find_element_by_id(tasklet_id);
    if (!tasklet_element) {
        throw InvalidTransformationDescriptionException("Element with ID " + std::to_string(tasklet_id) + " not found");
    }
    auto* tasklet = dynamic_cast<data_flow::Tasklet*>(tasklet_element);
    if (!tasklet) {
        throw InvalidTransformationDescriptionException(
            "Element with ID " + std::to_string(tasklet_id) + " is not a tasklet"
        );
    }

    return EinsumLift(*tasklet);
}

} // namespace transformations
} // namespace sdfg
