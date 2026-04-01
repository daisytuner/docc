#include "sdfg/transformations/einsum2dot.h"

#include <cstddef>
#include <nlohmann/json_fwd.hpp>
#include <string>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/library_nodes/math/blas/dot_node.h"
#include "sdfg/data_flow/library_nodes/math/math.h"
#include "sdfg/einsum/einsum.h"
#include "sdfg/optimization_report/pass_report_consumer.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/transformations/transformation.h"
#include "sdfg/types/type.h"
#include "sdfg/types/utils.h"

namespace sdfg {
namespace transformations {

Einsum2Dot::Einsum2Dot(einsum::EinsumNode& einsum_node, const std::string& target_tune)
    : einsum_node_(einsum_node), target_tune_(target_tune) {}

std::string Einsum2Dot::name() const { return "Einsum2Dot"; }

std::optional<sdfg::data_flow::ImplementationType> Einsum2Dot::get_impl_type(types::PrimitiveType data_type) {
    std::optional<sdfg::data_flow::ImplementationType> impl_type; // TODO make generic for any target
    if (target_tune_ == "sequential") {
        impl_type = sdfg::data_flow::ImplementationType_NONE;
    } else if (target_tune_ == "openmp") {
        impl_type = sdfg::math::blas::ImplementationType_BLAS;
    } else if (target_tune_ == "cuda") {
        impl_type = sdfg::cuda::blas::ImplementationType_CUBLASWithTransfers;
    } else if (target_tune_ == "tenstorrent") {
        if (data_type == types::PrimitiveType::Float) {
            impl_type = data_flow::ImplementationType{"TENSTORRENT_WithTransfers"};
        }
    }

    if (impl_type) {
        return impl_type;
    } else {
        return std::nullopt;
    }
}

bool Einsum2Dot::can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    // Check dims
    if (this->einsum_node_.dims().size() != 1 || !symbolic::eq(this->einsum_node_.init(0), symbolic::zero())) {
        return false;
    }
    symbolic::Symbol indvar = this->einsum_node_.indvar(0);

    // Check out indices
    if (this->einsum_node_.out_indices().size() != 0) {
        return false;
    }

    // Check inputs
    if (this->einsum_node_.inputs().size() != 3 || this->einsum_node_.input(2) != this->einsum_node_.output(0)) {
        return false;
    }

    // Check in indices
    if (this->einsum_node_.in_indices(0).size() != 1 || !symbolic::eq(this->einsum_node_.in_index(0, 0), indvar)) {
        return false;
    }
    if (this->einsum_node_.in_indices(1).size() != 1 || !symbolic::eq(this->einsum_node_.in_index(1, 0), indvar)) {
        return false;
    }

    // Get the data flow graph
    auto& dfg = this->einsum_node_.get_parent();

    // Determine and check the base type of output
    auto& oedge = *dfg.out_edges(this->einsum_node_).begin();
    auto data_type = oedge.base_type().primitive_type();
    if (data_type != types::PrimitiveType::Float && data_type != types::PrimitiveType::Double) {
        return false;
    }

    // Check if all inputs have the same primitive type
    for (auto& iedge : dfg.in_edges(this->einsum_node_)) {
        if (iedge.base_type().primitive_type() != data_type) {
            return false;
        }
    }

    if (!get_impl_type(data_type)) { // no implementation for the given tune exists
        return false;
    }

    return true;
}

void Einsum2Dot::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    // Get the data flow graph
    auto& dfg = this->einsum_node_.get_parent();

    // Get the block in which the einsum node lives
    auto* block = dynamic_cast<structured_control_flow::Block*>(dfg.get_parent());
    assert(block);

    // Get the number of iterations (n)
    symbolic::Expression n = this->einsum_node_.bound(0);

    // Determine the BLAS precision
    math::blas::BLAS_Precision precision;
    auto& datatype_oedge = *dfg.out_edges(this->einsum_node_).begin();
    types::PrimitiveType data_type = datatype_oedge.base_type().primitive_type();
    if (data_type == types::PrimitiveType::Float) {
        precision = math::blas::BLAS_Precision::s;
    } else {
        precision = math::blas::BLAS_Precision::d;
    }

    // Add the dot node
    auto& libnode = builder.add_library_node<
        math::blas::DotNode,
        const data_flow::ImplementationType&,
        const math::blas::BLAS_Precision&,
        symbolic::Expression>(
        *block, this->einsum_node_.debug_info(), this->get_impl_type(data_type).value(), precision, n
    );

    // Copy the memlets
    data_flow::AccessNode* leftover_access_node = nullptr;
    for (auto& iedge : dfg.in_edges(this->einsum_node_)) {
        if (iedge.dst_conn() == this->einsum_node_.input(0)) {
            builder.add_memlet(
                *block,
                iedge.src(),
                iedge.src_conn(),
                libnode,
                "__x",
                iedge.subset(),
                iedge.base_type(),
                iedge.debug_info()
            );
        } else if (iedge.dst_conn() == this->einsum_node_.input(1)) {
            builder.add_memlet(
                *block,
                iedge.src(),
                iedge.src_conn(),
                libnode,
                "__y",
                iedge.subset(),
                iedge.base_type(),
                iedge.debug_info()
            );
        } else if (iedge.dst_conn() == this->einsum_node_.input(2)) {
            leftover_access_node = dynamic_cast<data_flow::AccessNode*>(&iedge.src());
        }
    }
    for (auto& oedge : dfg.out_edges(this->einsum_node_)) {
        if (oedge.src_conn() == this->einsum_node_.output(0)) {
            builder.add_memlet(
                *block,
                libnode,
                "__out",
                oedge.dst(),
                oedge.dst_conn(),
                oedge.subset(),
                oedge.base_type(),
                oedge.debug_info()
            );
        }
    }

    // Remove the old memlets
    while (dfg.in_edges(this->einsum_node_).begin() != dfg.in_edges(this->einsum_node_).end()) {
        auto& iedge = *dfg.in_edges(this->einsum_node_).begin();
        builder.remove_memlet(*block, iedge);
    }
    while (dfg.out_edges(this->einsum_node_).begin() != dfg.out_edges(this->einsum_node_).end()) {
        auto& oedge = *dfg.out_edges(this->einsum_node_).begin();
        builder.remove_memlet(*block, oedge);
    }

    // Remove leftover access node
    builder.remove_node(*block, *leftover_access_node);

    // Remove the einsum node
    builder.remove_node(*block, this->einsum_node_);

    analysis_manager.invalidate_all();
}

void Einsum2Dot::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["einsum_node_element_id"] = this->einsum_node_.element_id();
    j["target_tune"] = this->target_tune_;
}

Einsum2Dot Einsum2Dot::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j) {
    assert(j.contains("einsum_node_element_id"));
    assert(j["einsum_node_element_id"].is_number_unsigned());
    assert(j.contains("impl_type"));

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

    std::string target_tune;
    if (j.contains("target_tune")) {
        target_tune = j.at("target_tune").get<std::string>();
    } else {
        target_tune = "none";
    }

    return Einsum2Dot(*einsum_node, target_tune);
}

} // namespace transformations
} // namespace sdfg
