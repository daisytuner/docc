#include "sdfg/transformations/offloading/cublas_data_transfer_extraction.h"

#include <cassert>
#include <cstddef>
#include <nlohmann/json_fwd.hpp>
#include <string>
#include <unordered_map>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_nodes/math/blas/batched_gemm_node.h"
#include "sdfg/data_flow/library_nodes/math/blas/dot_node.h"
#include "sdfg/data_flow/library_nodes/math/blas/gemm_node.h"
#include "sdfg/data_flow/library_nodes/math/math.h"
#include "sdfg/element.h"
#include "sdfg/exceptions.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/targets/cuda/cuda_data_offloading_node.h"
#include "sdfg/transformations/transformation.h"
#include "sdfg/types/type.h"
#include "sdfg/types/utils.h"
#include "symengine/symengine_rcp.h"

namespace sdfg {
namespace cuda {

std::string CUBLASDataTransferExtraction::create_device_container(
    builder::StructuredSDFGBuilder& builder, const types::Pointer& type, const symbolic::Expression& size
) {
    auto new_type = type.clone();
    new_type->storage_type(types::StorageType(
        "NV_Generic", size, types::StorageType::AllocationType::Unmanaged, types::StorageType::AllocationType::Unmanaged
    ));
    auto device_container = builder.find_new_name(CUDA_DEVICE_PREFIX);
    builder.add_container(device_container, *new_type);
    return device_container;
}

void CUBLASDataTransferExtraction::create_allocate(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& sequence,
    structured_control_flow::Block& block,
    const std::string& device_container,
    const symbolic::Expression& size,
    const types::Pointer& type
) {
    auto& alloc_block = builder.add_block_before(sequence, block, {}, block.debug_info());
    offloading::add_offloading_node<CUDADataOffloadingNode>(
        builder,
        alloc_block,
        device_container,
        device_container,
        offloading::DataTransferDirection::NONE,
        offloading::BufferLifecycle::ALLOC,
        type,
        type,
        this->blas_node_.debug_info(),
        size,
        symbolic::zero()
    );
}

void CUBLASDataTransferExtraction::create_deallocate(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& sequence,
    structured_control_flow::Block& block,
    const std::string& device_container,
    const types::Pointer& type
) {
    auto& dealloc_block = builder.add_block_after(sequence, block, {}, block.debug_info());
    offloading::add_offloading_node<CUDADataOffloadingNode>(
        builder,
        dealloc_block,
        device_container,
        device_container,
        offloading::DataTransferDirection::NONE,
        offloading::BufferLifecycle::FREE,
        type,
        type,
        this->blas_node_.debug_info(),
        SymEngine::null,
        symbolic::zero()
    );
}

void CUBLASDataTransferExtraction::create_copy_to_device(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& sequence,
    structured_control_flow::Block& block,
    const std::string& host_container,
    const std::string& device_container,
    const symbolic::Expression& size,
    const types::Pointer& type
) {
    auto& copy_block = builder.add_block_before(sequence, block, {}, block.debug_info());
    offloading::add_offloading_node<CUDADataOffloadingNode>(
        builder,
        copy_block,
        host_container,
        device_container,
        offloading::DataTransferDirection::H2D,
        offloading::BufferLifecycle::NO_CHANGE,
        type,
        type,
        this->blas_node_.debug_info(),
        size,
        symbolic::zero()
    );
}

void CUBLASDataTransferExtraction::create_copy_from_device(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& sequence,
    structured_control_flow::Block& block,
    const std::string& host_container,
    const std::string& device_container,
    const symbolic::Expression& size,
    const types::Pointer& type
) {
    auto& copy_block = builder.add_block_after(sequence, block, {}, block.debug_info());
    offloading::add_offloading_node<CUDADataOffloadingNode>(
        builder,
        copy_block,
        host_container,
        device_container,
        offloading::DataTransferDirection::D2H,
        offloading::BufferLifecycle::NO_CHANGE,
        type,
        type,
        this->blas_node_.debug_info(),
        size,
        symbolic::zero()
    );
}

void CUBLASDataTransferExtraction::create_copy_to_device_with_allocation(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& sequence,
    structured_control_flow::Block& block,
    const std::string& host_container,
    const std::string& device_container,
    const symbolic::Expression& size,
    const types::Pointer& type
) {
    auto& copy_block = builder.add_block_before(sequence, block, {}, block.debug_info());
    offloading::add_offloading_node<CUDADataOffloadingNode>(
        builder,
        copy_block,
        host_container,
        device_container,
        offloading::DataTransferDirection::H2D,
        offloading::BufferLifecycle::ALLOC,
        type,
        type,
        this->blas_node_.debug_info(),
        size,
        symbolic::zero()
    );
}

void CUBLASDataTransferExtraction::create_copy_from_device_with_deallocation(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& sequence,
    structured_control_flow::Block& block,
    const std::string& host_container,
    const std::string& device_container,
    const symbolic::Expression& size,
    const types::Pointer& type
) {
    auto& copy_block = builder.add_block_after(sequence, block, {}, block.debug_info());
    offloading::add_offloading_node<CUDADataOffloadingNode>(
        builder,
        copy_block,
        host_container,
        device_container,
        offloading::DataTransferDirection::D2H,
        offloading::BufferLifecycle::FREE,
        type,
        type,
        this->blas_node_.debug_info(),
        size,
        symbolic::zero()
    );
}

CUBLASDataTransferExtraction::CUBLASDataTransferExtraction(math::blas::BLASNode& blas_node) : blas_node_(blas_node) {}

std::string CUBLASDataTransferExtraction::name() const { return "CUBLASDataTransferExtraction"; }

bool CUBLASDataTransferExtraction::
    can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    // BLAS node must have implementation type CUBLAS without data transfers
    if (this->blas_node_.implementation_type().value() != cuda::ImplementationType_CUDAWithTransfers.value()) {
        return false;
    }


    // Restrict to BLAS nodes in their own block
    auto& dfg = this->blas_node_.get_parent();
    if (dfg.nodes().size() != dfg.in_degree(this->blas_node_) + dfg.out_degree(this->blas_node_) + 1) {
        return false;
    }

    // Supported BLAS nodes
    if (dynamic_cast<math::blas::DotNode*>(&this->blas_node_)) {
        return true;
    } else if (dynamic_cast<math::blas::GEMMNode*>(&this->blas_node_)) {
        return true;
    } else if (dynamic_cast<math::blas::BatchedGEMMNode*>(&this->blas_node_)) {
        return true;
    } else {
        return false;
    }
}

void CUBLASDataTransferExtraction::
    apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    // Get data flow graph and block
    auto& dfg = this->blas_node_.get_parent();
    auto* block = dynamic_cast<structured_control_flow::Block*>(dfg.get_parent());
    assert(block);

    // Get sequence
    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto* sequence = dynamic_cast<structured_control_flow::Sequence*>(scope_analysis.parent_scope(block));
    assert(sequence);

    // Determine type
    types::PrimitiveType precision;
    switch (this->blas_node_.precision()) {
        case math::blas::h:
            precision = types::PrimitiveType::Half;
            break;
        case math::blas::s:
            precision = types::PrimitiveType::Float;
            break;
        case math::blas::d:
            precision = types::PrimitiveType::Double;
            break;
        default:
            throw InvalidSDFGException("CUBLASDataTransferExtraction: Unsupported precision");
    }
    types::Scalar base_type(precision);
    types::Pointer type(base_type);

    // Capture in and out accesses
    std::unordered_map<std::string, data_flow::AccessNode&> in_access, out_access;
    for (auto& iedge : dfg.in_edges(this->blas_node_)) {
        in_access.insert({iedge.dst_conn(), static_cast<data_flow::AccessNode&>(iedge.src())});
    }
    for (auto& oedge : dfg.out_edges(this->blas_node_)) {
        out_access.insert({oedge.src_conn(), static_cast<data_flow::AccessNode&>(oedge.dst())});
    }

    if (auto* dot_node = dynamic_cast<math::blas::DotNode*>(&this->blas_node_)) {
        auto x_size = symbolic::mul(
            symbolic::add(symbolic::mul(symbolic::sub(dot_node->n(), symbolic::one()), dot_node->incx()), symbolic::one()),
            types::get_contiguous_element_size(type, true)
        );
        auto y_size = symbolic::mul(
            symbolic::add(symbolic::mul(symbolic::sub(dot_node->n(), symbolic::one()), dot_node->incy()), symbolic::one()),
            types::get_contiguous_element_size(type, true)
        );
        auto dx = this->create_device_container(builder, type, x_size);
        auto dy = this->create_device_container(builder, type, y_size);

        this->create_copy_to_device_with_allocation(
            builder, *sequence, *block, in_access.at("__x").data(), dx, x_size, type
        );
        this->create_copy_to_device_with_allocation(
            builder, *sequence, *block, in_access.at("__y").data(), dy, y_size, type
        );

        this->create_deallocate(builder, *sequence, *block, dx, type);
        this->create_deallocate(builder, *sequence, *block, dy, type);

        in_access.at("__x").data(dx);
        in_access.at("__y").data(dy);
    } else if (auto* gemm_node = dynamic_cast<math::blas::GEMMNode*>(&this->blas_node_)) {
        auto elem_size = types::get_contiguous_element_size(type, true);
        auto a_size = symbolic::mul(symbolic::mul(gemm_node->m(), gemm_node->k()), elem_size);
        auto b_size = symbolic::mul(symbolic::mul(gemm_node->k(), gemm_node->n()), elem_size);
        auto c_size = symbolic::mul(symbolic::mul(gemm_node->m(), gemm_node->n()), elem_size);

        auto dA = this->create_device_container(builder, type, a_size);
        auto dB = this->create_device_container(builder, type, b_size);
        auto dC = this->create_device_container(builder, type, c_size);

        this->create_copy_to_device_with_allocation(
            builder, *sequence, *block, in_access.at("__A").data(), dA, a_size, type
        );
        this->create_copy_to_device_with_allocation(
            builder, *sequence, *block, in_access.at("__B").data(), dB, b_size, type
        );
        auto c_container = in_access.at("__C").data();
        this->create_copy_to_device_with_allocation(builder, *sequence, *block, c_container, dC, c_size, type);

        this->create_copy_from_device_with_deallocation(builder, *sequence, *block, c_container, dC, c_size, type);
        this->create_deallocate(builder, *sequence, *block, dA, type);
        this->create_deallocate(builder, *sequence, *block, dB, type);

        in_access.at("__A").data(dA);
        in_access.at("__B").data(dB);
        in_access.at("__C").data(dC);
    } else if (auto* batched_gemm_node = dynamic_cast<math::blas::BatchedGEMMNode*>(&this->blas_node_)) {
        auto elem_size = types::get_contiguous_element_size(type, true);
        auto a_size =
            symbolic::mul(symbolic::mul(batched_gemm_node->batch_count(), batched_gemm_node->stride_a()), elem_size);
        auto b_size =
            symbolic::mul(symbolic::mul(batched_gemm_node->batch_count(), batched_gemm_node->stride_b()), elem_size);
        auto c_size =
            symbolic::mul(symbolic::mul(batched_gemm_node->batch_count(), batched_gemm_node->stride_c()), elem_size);

        auto dA = this->create_device_container(builder, type, a_size);
        auto dB = this->create_device_container(builder, type, b_size);
        auto dC = this->create_device_container(builder, type, c_size);

        this->create_copy_to_device_with_allocation(
            builder, *sequence, *block, in_access.at("__A").data(), dA, a_size, type
        );
        this->create_copy_to_device_with_allocation(
            builder, *sequence, *block, in_access.at("__B").data(), dB, b_size, type
        );
        auto c_container = in_access.at("__C").data();
        this->create_copy_to_device_with_allocation(builder, *sequence, *block, c_container, dC, c_size, type);

        this->create_copy_from_device_with_deallocation(builder, *sequence, *block, c_container, dC, c_size, type);
        this->create_deallocate(builder, *sequence, *block, dA, type);
        this->create_deallocate(builder, *sequence, *block, dB, type);

        in_access.at("__A").data(dA);
        in_access.at("__B").data(dB);
        in_access.at("__C").data(dC);
    } else {
        throw InvalidSDFGException("CUBLASDataTransferExtraction: Unsupported BLAS type");
    }

    // Change the implementation type to CUBLAS without data transfers
    this->blas_node_.implementation_type() = cuda::ImplementationType_CUDAWithoutTransfers;
}

void CUBLASDataTransferExtraction::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();

    // BLAS nodes are not loops; they appear as generic elements in GNN data.
    // Use type "unknown" to match the feature extractor's classification.
    j["subgraph"] = {{"0", {{"element_id", this->blas_node_.element_id()}, {"type", "unknown"}}}};

    // Legacy field for backward compatibility.
    j["blas_node_element_id"] = this->blas_node_.element_id();
}

CUBLASDataTransferExtraction CUBLASDataTransferExtraction::
    from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j) {
    size_t blas_node_id;
    if (j.contains("subgraph")) {
        const auto& node_desc = j.at("subgraph").at("0");
        blas_node_id = node_desc.at("element_id").get<size_t>();
    } else {
        blas_node_id = j.at("blas_node_element_id").get<size_t>();
    }
    auto* blas_node_element = builder.find_element_by_id(blas_node_id);
    if (!blas_node_element) {
        throw transformations::
            InvalidTransformationDescriptionException("Element with ID " + std::to_string(blas_node_id) + " not found");
    }
    auto* blas_node = dynamic_cast<math::blas::BLASNode*>(blas_node_element);
    if (!blas_node) {
        throw transformations::InvalidTransformationDescriptionException(
            "Element with ID " + std::to_string(blas_node_id) + " is not a BLASNode"
        );
    }

    return CUBLASDataTransferExtraction(*blas_node);
}

} // namespace cuda
} // namespace sdfg
