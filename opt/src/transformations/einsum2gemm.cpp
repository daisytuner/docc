#include "sdfg/transformations/einsum2gemm.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <nlohmann/json_fwd.hpp>
#include <optional>
#include <string>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/library_nodes/math/blas/gemm_node.h"
#include "sdfg/data_flow/library_nodes/math/math.h"
#include "sdfg/einsum/einsum.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/tenstorrent/library_node_mapping.h"
#include "sdfg/transformations/transformation.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/type.h"
#include "symengine/symengine_rcp.h"

namespace sdfg {
namespace transformations {

bool Einsum2Gemm::check_matrix_indices(long long mat, const symbolic::Symbol& indvar1, const symbolic::Symbol& indvar2) {
    return !symbolic::eq(this->einsum_node_.in_index(mat, 0), this->einsum_node_.in_index(mat, 1)) &&
           (symbolic::eq(this->einsum_node_.in_index(mat, 0), indvar1) ||
            symbolic::eq(this->einsum_node_.in_index(mat, 0), indvar2)) &&
           (symbolic::eq(this->einsum_node_.in_index(mat, 1), indvar1) ||
            symbolic::eq(this->einsum_node_.in_index(mat, 1), indvar2));
}

Einsum2Gemm::Einsum2Gemm(einsum::EinsumNode& einsum_node, const std::string& target_tune)
    : einsum_node_(einsum_node), target_tune_(target_tune) {}

std::string Einsum2Gemm::name() const { return "Einsum2Gemm"; }

std::optional<sdfg::data_flow::ImplementationType> Einsum2Gemm::get_impl_type(types::PrimitiveType data_type) {
    std::optional<sdfg::data_flow::ImplementationType> impl_type = std::nullopt;
    if (this->target_tune_ == "openmp") {
        impl_type = std::make_optional(sdfg::math::blas::ImplementationType_BLAS);
    } else if (this->target_tune_ == "tenstorrent") {
        impl_type = tenstorrent::try_map_library_node_implementation(math::blas::LibraryNodeType_GEMM, data_type);
    }
    // TODO: Implement GEMM dispatcher for CUBLAS

    return impl_type;
}

bool Einsum2Gemm::can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    // Check dims
    if (this->einsum_node_.dims().size() != 3) {
        return false;
    }

    // Check initial values
    for (size_t i = 0; i < 3; i++) {
        if (!symbolic::eq(this->einsum_node_.init(i), symbolic::zero())) {
            return false;
        }
    }

    // Check out indices
    if (this->einsum_node_.out_indices().size() != 2) {
        return false;
    }
    symbolic::Symbol indvar_outer_1 = SymEngine::null, indvar_outer_2 = SymEngine::null, indvar_inner = SymEngine::null;
    std::vector<size_t> permutation = {0, 1, 2};
    do {
        if (symbolic::eq(this->einsum_node_.out_index(0), this->einsum_node_.indvar(permutation[0])) &&
            symbolic::eq(this->einsum_node_.out_index(1), this->einsum_node_.indvar(permutation[1]))) {
            indvar_outer_1 = this->einsum_node_.indvar(permutation[0]);
            indvar_outer_2 = this->einsum_node_.indvar(permutation[1]);
            indvar_inner = this->einsum_node_.indvar(permutation[2]);
            break;
        }
    } while (std::next_permutation(permutation.begin(), permutation.end()));
    if (indvar_outer_1.is_null() || indvar_outer_2.is_null() || indvar_inner.is_null()) {
        return false;
    }

    // Check bounds, i.e., preven triangular access
    for (size_t i = 0; i < 3; i++) {
        if (symbolic::uses(this->einsum_node_.bound(i), indvar_outer_1) ||
            symbolic::uses(this->einsum_node_.bound(i), indvar_outer_2) ||
            symbolic::uses(this->einsum_node_.bound(i), indvar_inner)) {
            return false;
        }
    }

    // Check inputs
    long long A = -1, B = -1, C = -1;
    if (this->einsum_node_.inputs().size() == 3) {
        C = 2;
        for (size_t i = 0; i < this->einsum_node_.in_indices().size() - 1; i++) {
            if (this->einsum_node_.in_indices(i).size() != 2) {
                break;
            } else if (symbolic::eq(this->einsum_node_.in_index(i, 0), indvar_outer_1) ||
                       symbolic::eq(this->einsum_node_.in_index(i, 1), indvar_outer_1)) {
                A = i;
            } else if (symbolic::eq(this->einsum_node_.in_index(i, 0), indvar_outer_2) ||
                       symbolic::eq(this->einsum_node_.in_index(i, 1), indvar_outer_2)) {
                B = i;
            }
        }
    } else if (this->einsum_node_.inputs().size() == 4) {
        C = 3;
        long long alpha = -1;
        for (size_t i = 0; i < this->einsum_node_.in_indices().size() - 1; i++) {
            if (this->einsum_node_.in_indices(i).size() == 0) {
                alpha = i;
            } else if (this->einsum_node_.in_indices(i).size() != 2) {
                break;
            } else if (symbolic::eq(this->einsum_node_.in_index(i, 0), indvar_outer_1) ||
                       symbolic::eq(this->einsum_node_.in_index(i, 1), indvar_outer_1)) {
                A = i;
            } else if (symbolic::eq(this->einsum_node_.in_index(i, 0), indvar_outer_2) ||
                       symbolic::eq(this->einsum_node_.in_index(i, 1), indvar_outer_2)) {
                B = i;
            }
        }

        // Check alpha
        if (alpha == -1 || this->einsum_node_.in_indices(alpha).size() != 0) {
            return false;
        }
    } else {
        return false;
    }
    if (A == -1 || B == -1 || A == B || this->einsum_node_.input(C) != this->einsum_node_.output(0)) {
        return false;
    }

    // Check in indices
    if (this->einsum_node_.in_indices(A).size() != 2 || !this->check_matrix_indices(A, indvar_outer_1, indvar_inner)) {
        return false;
    }
    if (this->einsum_node_.in_indices(B).size() != 2 || !this->check_matrix_indices(B, indvar_inner, indvar_outer_2)) {
        return false;
    }
    if (this->einsum_node_.in_indices(C).size() != 2 ||
        !symbolic::eq(this->einsum_node_.in_index(C, 0), indvar_outer_1) ||
        !symbolic::eq(this->einsum_node_.in_index(C, 1), indvar_outer_2)) {
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

    if (!this->get_impl_type(data_type)) { // no implementation for the given tune exists
        return false;
    }

    return true;
}

void Einsum2Gemm::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    // Get the data flow graph
    auto& dfg = this->einsum_node_.get_parent();

    // Get the block in which the einsum node lives
    auto* block = dynamic_cast<structured_control_flow::Block*>(dfg.get_parent());
    assert(block);

    // Determine the BLAS precision
    math::blas::BLAS_Precision precision;
    auto& datatype_oedge = *dfg.out_edges(this->einsum_node_).begin();
    types::PrimitiveType data_type = datatype_oedge.base_type().primitive_type();
    if (data_type == types::PrimitiveType::Float) {
        precision = math::blas::BLAS_Precision::s;
    } else {
        precision = math::blas::BLAS_Precision::d;
    }

    // Determine indvars
    symbolic::Symbol indvar_outer_1 = SymEngine::null, indvar_outer_2 = SymEngine::null, indvar_inner = SymEngine::null;
    symbolic::Expression m = SymEngine::null, n = SymEngine::null, k = SymEngine::null;
    std::vector<size_t> permutation = {0, 1, 2};
    do {
        if (symbolic::eq(this->einsum_node_.out_index(0), this->einsum_node_.indvar(permutation[0])) &&
            symbolic::eq(this->einsum_node_.out_index(1), this->einsum_node_.indvar(permutation[1]))) {
            indvar_outer_1 = this->einsum_node_.indvar(permutation[0]);
            indvar_outer_2 = this->einsum_node_.indvar(permutation[1]);
            indvar_inner = this->einsum_node_.indvar(permutation[2]);
            m = this->einsum_node_.bound(permutation[0]);
            n = this->einsum_node_.bound(permutation[1]);
            k = this->einsum_node_.bound(permutation[2]);
            break;
        }
    } while (std::next_permutation(permutation.begin(), permutation.end()));
    assert(
        !indvar_outer_1.is_null() && !indvar_outer_2.is_null() && !indvar_inner.is_null() && !m.is_null() &&
        !n.is_null() && !k.is_null()
    );

    // Determine inputs
    long long alpha = -1, A = -1, B = -1, C = -1;
    bool has_alpha = false;
    if (this->einsum_node_.inputs().size() == 3) {
        C = 2;
        for (size_t i = 0; i < this->einsum_node_.in_indices().size() - 1; i++) {
            if (this->einsum_node_.in_indices(i).size() != 2) {
                break;
            } else if (symbolic::eq(this->einsum_node_.in_index(i, 0), indvar_outer_1) ||
                       symbolic::eq(this->einsum_node_.in_index(i, 1), indvar_outer_1)) {
                A = i;
            } else if (symbolic::eq(this->einsum_node_.in_index(i, 0), indvar_outer_2) ||
                       symbolic::eq(this->einsum_node_.in_index(i, 1), indvar_outer_2)) {
                B = i;
            }
        }
    } else if (this->einsum_node_.inputs().size() == 4) {
        C = 3;
        has_alpha = true;
        for (size_t i = 0; i < this->einsum_node_.in_indices().size() - 1; i++) {
            if (this->einsum_node_.in_indices(i).size() == 0) {
                alpha = i;
            } else if (this->einsum_node_.in_indices(i).size() != 2) {
                break;
            } else if (symbolic::eq(this->einsum_node_.in_index(i, 0), indvar_outer_1) ||
                       symbolic::eq(this->einsum_node_.in_index(i, 1), indvar_outer_1)) {
                A = i;
            } else if (symbolic::eq(this->einsum_node_.in_index(i, 0), indvar_outer_2) ||
                       symbolic::eq(this->einsum_node_.in_index(i, 1), indvar_outer_2)) {
                B = i;
            }
        }
    }

    // Determine transpose and leading dimensions
    math::blas::BLAS_Transpose transA, transB;
    symbolic::Expression ldA, ldB, ldC;
    if (symbolic::eq(this->einsum_node_.in_index(A, 0), indvar_outer_1)) {
        transA = math::blas::BLAS_Transpose::No;
        ldA = k;
    } else {
        transA = math::blas::BLAS_Transpose::Trans;
        ldA = m;
    }
    if (symbolic::eq(this->einsum_node_.in_index(B, 1), indvar_outer_2)) {
        transB = math::blas::BLAS_Transpose::No;
        ldB = n;
    } else {
        transB = math::blas::BLAS_Transpose::Trans;
        ldB = k;
    }
    ldC = n;

    // Add the BLAS node for gemm
    auto& libnode = builder.add_library_node<math::blas::GEMMNode>(
        *block,
        this->einsum_node_.debug_info(),
        this->get_impl_type(data_type).value(),
        precision,
        math::blas::BLAS_Layout::RowMajor,
        transA,
        transB,
        m,
        n,
        k,
        ldA,
        ldB,
        ldC
    );

    // Copy the memlets
    for (auto& iedge : dfg.in_edges(this->einsum_node_)) {
        if (has_alpha && iedge.dst_conn() == this->einsum_node_.input(alpha)) {
            builder.add_memlet(
                *block,
                iedge.src(),
                iedge.src_conn(),
                libnode,
                "__alpha",
                iedge.subset(),
                iedge.base_type(),
                iedge.debug_info()
            );
        } else if (iedge.dst_conn() == this->einsum_node_.input(A)) {
            builder.add_memlet(
                *block,
                iedge.src(),
                iedge.src_conn(),
                libnode,
                "__A",
                iedge.subset(),
                iedge.base_type(),
                iedge.debug_info()
            );
        } else if (iedge.dst_conn() == this->einsum_node_.input(B)) {
            builder.add_memlet(
                *block,
                iedge.src(),
                iedge.src_conn(),
                libnode,
                "__B",
                iedge.subset(),
                iedge.base_type(),
                iedge.debug_info()
            );
        } else if (iedge.dst_conn() == this->einsum_node_.input(C)) {
            builder.add_memlet(
                *block,
                iedge.src(),
                iedge.src_conn(),
                libnode,
                "__C",
                iedge.subset(),
                iedge.base_type(),
                iedge.debug_info()
            );
        }
    }
    for (auto& oedge : dfg.out_edges(this->einsum_node_)) {
        if (oedge.src_conn() == this->einsum_node_.output(0)) {
            builder.add_memlet(
                *block,
                libnode,
                "__C",
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

    // Add constant scalars alpha and beta (if needed)
    types::Scalar data_type_scalar(data_type);
    if (!has_alpha) {
        auto& alpha_access_node =
            builder.add_constant(*block, "1.0", data_type_scalar, this->einsum_node_.debug_info());
        builder.add_memlet(
            *block, alpha_access_node, "void", libnode, "__alpha", {}, data_type_scalar, this->einsum_node_.debug_info()
        );
    }
    auto& beta_access_node = builder.add_constant(*block, "1.0", data_type_scalar, this->einsum_node_.debug_info());
    builder.add_memlet(
        *block, beta_access_node, "void", libnode, "__beta", {}, data_type_scalar, this->einsum_node_.debug_info()
    );

    // Remove the einsum node
    builder.remove_node(*block, this->einsum_node_);

    analysis_manager.invalidate_all();
}

void Einsum2Gemm::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["einsum_node_element_id"] = this->einsum_node_.element_id();
    j["target_tune"] = this->target_tune_;
}

Einsum2Gemm Einsum2Gemm::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j) {
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

    return Einsum2Gemm(*einsum_node, target_tune);
}

} // namespace transformations
} // namespace sdfg
