#include "docc/target/tenstorrent/tenstorrent_offloading_expansion.h"

#include <cassert>
#include <utility>

#include "../../../../sdfg/include/sdfg/targets/offloading/data_offloading_node.h"
#include "docc/target/tenstorrent/plugin.h"
#include "docc/target/tenstorrent/tenstorrent_offloading_node.h"
#include "docc/target/tenstorrent/tenstorrent_transfer_arg.h"
#include "sdfg/analysis/arguments_analysis.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/symbolic/symbolic.h"

#include "sdfg/optimization_report/pass_report_consumer.h"
#include "symengine/symengine_rcp.h"


namespace sdfg::tenstorrent {

constexpr int32_t TENSTORRENT_TILE_ELEMS = 1024;

using namespace sdfg::transformations;

void TenstorrentOffloadingExpansion::expand_blas(sdfg::math::blas::BLASNode& node) {
    std::vector<TransferArg> transferred_args;

    if (auto* dotNode = dynamic_cast<sdfg::math::blas::DotNode*>(&node)) {
        types::PrimitiveType primitive = dotNode->scalar_primitive();
        auto prim_bytes = types::bit_width(primitive) / 8;

        auto& dflow = dotNode->get_parent();

        for (auto& iedge : dflow.in_edges(*dotNode)) {
            auto& in_name = iedge.dst_conn();
            if (in_name == "__x") {
                // ok
            } else if (in_name == "__y") {
                // ok
            } else {
                throw InvalidSDFGException("DotNode has unexpected input: " + in_name);
            }

            auto* srcNode = dynamic_cast<data_flow::AccessNode*>(&iedge.src());
            transferred_args.emplace_back(
                srcNode->data(),
                iedge.base_type(),
                symbolic::mul(dotNode->n(), symbolic::integer(prim_bytes)),
                symbolic::integer(TENSTORRENT_TILE_ELEMS * prim_bytes),
                analysis::RegionArgument(true, true, false, false, true)
            );
        }

        auto& out_edge = *dflow.out_edges(*dotNode).begin();
        assert(out_edge.src_conn() == "__out");
        if (report_) report_->transform_possible(name());
    } else {
        if (report_) report_->transform_impossible(name(), "Unsupported BLAS operation: " + node.code().value());
        return;
    }

    create_offloaded_memory_handling(transferred_args);

    node.implementation_type() = ImplementationType_Tenstorrent_WithoutTransfers;
    if (report_) report_->transform_applied(name());
}

structured_control_flow::Sequence& TenstorrentOffloadingExpansion::require_parent_scope() {
    if (parent_scope_) {
        return *parent_scope_;
    } else {
        auto& scope_analysis = analysis_mgr_.get<analysis::ScopeAnalysis>();
        auto* parent_scope = static_cast<structured_control_flow::Sequence*>(scope_analysis.parent_scope(&scope_));
        parent_scope_ = parent_scope;
        return *parent_scope;
    }
}

structured_control_flow::Block& TenstorrentOffloadingExpansion::require_allocate_block() {
    if (alloc_block_) {
        return *alloc_block_;
    } else {
        auto& parent_scope = require_parent_scope();
        auto& copy_in_block = require_copy_in_block();
        ControlFlowNode& scope = scope_;
        auto& before = builder_.add_block_before(parent_scope, copy_in_block, {}, scope.debug_info());
        alloc_block_ = &before;
        return before;
    }
}

structured_control_flow::Block& TenstorrentOffloadingExpansion::require_copy_in_block() {
    if (copy_in_block_) {
        return *copy_in_block_;
    } else {
        auto& parent_scope = require_parent_scope();
        ControlFlowNode& scope = scope_;
        auto& before = builder_.add_block_before(parent_scope, scope, {}, scope.debug_info());
        copy_in_block_ = &before;
        return before;
    }
}

structured_control_flow::Block& TenstorrentOffloadingExpansion::require_copy_out_block() {
    if (copy_out_block_) {
        return *copy_out_block_;
    } else {
        auto& parent_scope = require_parent_scope();
        ControlFlowNode& scope = scope_;
        auto& after = builder_.add_block_after(parent_scope, scope, {}, scope.debug_info());
        copy_out_block_ = &after;
        return after;
    }
}

void TenstorrentOffloadingExpansion::set_report(sdfg::PassReportConsumer* report) { this->report_ = report; }

void TenstorrentOffloadingExpansion::allocate_device_arg(
    builder::StructuredSDFGBuilder& builder,
    const ControlFlowNode& org,
    Block& alloc_block,
    std::string host_arg_name,
    std::string device_arg_name,
    symbolic::Expression arg_size,
    symbolic::Expression page_size
) {
    auto& access_node_out_device = builder.add_access(alloc_block, device_arg_name);

    auto& malloc_node = builder.add_library_node<TTDataOffloadingNode>(
        alloc_block,
        org.debug_info(),
        force_synchronous_,
        device_handle_,
        std::move(arg_size),
        std::move(page_size),
        offloading::DataTransferDirection::NONE,
        offloading::BufferLifecycle::ALLOC
    );

    auto& out_type = builder.subject().type(device_arg_name);
    builder.add_computational_memlet(alloc_block, malloc_node, "_ret", access_node_out_device, {}, out_type);
}

void TenstorrentOffloadingExpansion::deallocate_device_arg(
    builder::StructuredSDFGBuilder& builder,
    const ControlFlowNode& org,
    Block& dealloc_block,
    std::string device_arg_name,
    symbolic::Expression arg_size,
    symbolic::Expression page_size
) {
    auto& access_node_in_device = builder.add_access(dealloc_block, device_arg_name);
    auto& access_node_out_device = builder.add_access(dealloc_block, device_arg_name);

    auto& free_node = builder.add_library_node<TTDataOffloadingNode>(
        dealloc_block,
        org.debug_info(),
        force_synchronous_,
        device_handle_,
        std::move(arg_size),
        std::move(page_size),
        offloading::DataTransferDirection::NONE,
        offloading::BufferLifecycle::FREE
    );

    auto& device_type = builder.subject().type(device_arg_name);
    builder.add_computational_memlet(dealloc_block, access_node_in_device, free_node, "_ptr", {}, device_type);
    builder.add_computational_memlet(dealloc_block, free_node, "_ptr", access_node_out_device, {}, device_type);
}

void TenstorrentOffloadingExpansion::copy_to_device(
    builder::StructuredSDFGBuilder& builder,
    const ControlFlowNode& org,
    const std::string host_arg_name,
    std::string device_arg_name,
    symbolic::Expression size,
    symbolic::Expression page_size,
    Block& copy_block
) {
    auto& access_node_in = builder.add_access(copy_block, host_arg_name);
    auto& access_node_out = builder.add_access(copy_block, device_arg_name);

    auto& memcpy_node = builder.add_library_node<TTDataOffloadingNode>(
        copy_block,
        org.debug_info(),
        force_synchronous_,
        device_handle_,
        size,
        page_size,
        offloading::DataTransferDirection::H2D,
        offloading::BufferLifecycle::NO_CHANGE
    );

    auto& in_type = builder.subject().type(host_arg_name);
    builder.add_computational_memlet(copy_block, access_node_in, memcpy_node, "_src", {}, in_type);

    auto& out_type = builder.subject().type(device_arg_name);
    builder.add_computational_memlet(copy_block, memcpy_node, "_dst", access_node_out, {}, out_type);
}

void TenstorrentOffloadingExpansion::copy_to_device_with_allocation(
    builder::StructuredSDFGBuilder& builder,
    const ControlFlowNode& org,
    const std::string host_arg_name,
    std::string device_arg_name,
    symbolic::Expression size,
    Block& block,
    symbolic::Expression page_size
) {
    auto& access_node_in = builder.add_access(block, host_arg_name);
    auto& access_node_out = builder.add_access(block, device_arg_name);

    auto& memcpy_node = builder.add_library_node<TTDataOffloadingNode>(
        block,
        org.debug_info(),
        force_synchronous_,
        device_handle_,
        size,
        page_size,
        offloading::DataTransferDirection::H2D,
        offloading::BufferLifecycle::ALLOC
    );

    auto& in_type = builder.subject().type(host_arg_name);
    builder.add_computational_memlet(block, access_node_in, memcpy_node, "_src", {}, in_type);

    auto& out_type = builder.subject().type(device_arg_name);
    builder.add_computational_memlet(block, memcpy_node, "_dst", access_node_out, {}, out_type);
}

void TenstorrentOffloadingExpansion::copy_from_device(
    builder::StructuredSDFGBuilder& builder,
    const ControlFlowNode& org,
    Block& copy_out_block,
    const std::string host_arg_name,
    std::string device_arg_name,
    symbolic::Expression size,
    symbolic::Expression page_size
) {
    auto& access_node_in = builder.add_access(copy_out_block, device_arg_name);
    auto& access_node_out = builder.add_access(copy_out_block, host_arg_name);

    bool blocking_copy_from_device = true;
    auto& memcpy_node = builder.add_library_node<TTDataOffloadingNode>(
        copy_out_block,
        org.debug_info(),
        blocking_copy_from_device,
        device_handle_,
        size,
        page_size,
        offloading::DataTransferDirection::D2H,
        offloading::BufferLifecycle::NO_CHANGE
    );

    auto& in_type = builder.subject().type(device_arg_name);
    builder.add_computational_memlet(copy_out_block, access_node_in, memcpy_node, "_src", {}, in_type);

    auto& out_type = builder.subject().type(host_arg_name);
    builder.add_computational_memlet(copy_out_block, memcpy_node, "_dst", access_node_out, {}, out_type);
}

void TenstorrentOffloadingExpansion::create_offloaded_memory_handling(std::vector<TransferArg>& transfer_args) {
    // Create containers for on-device data
    for (auto& [argument, type, arg_size, page_size, alloc_size, meta] : transfer_args) {
        if (builder_.subject().exists(copy_prefix() + argument)) {
            continue;
        }
        auto argument_device = copy_prefix() + argument;

        auto buf_type = types::Structure("std::shared_ptr<tt::tt_metal::Buffer>");
        buf_type.storage_type(global_device_storage_type(alloc_size, page_size));
        builder_.add_container(argument_device, buf_type);
    }

    //  Copy-In & allocate arguments on device memory
    for (auto& [argument, type, arg_size, page_size, alloc_size, meta] : transfer_args) {
        if (!meta.is_ptr) {
            continue;
        }
        auto argument_device = copy_prefix() + argument;
        auto& new_block = builder_.add_block_before(require_parent_scope(), scope_, {}, scope_.debug_info());
        if (meta.is_input) {
            copy_to_device_with_allocation(builder_, scope_, argument, argument_device, arg_size, new_block, page_size);
        } else {
            allocate_device_arg(builder_, scope_, new_block, argument, argument_device, arg_size, page_size);
        }
    }

    // Replace args inside with on-device pointers
    for (auto& [argument, type, arg_size, page_size, alloc_size, meta] : transfer_args) {
        if (meta.is_ptr) {
            auto argument_device = copy_prefix() + argument;
            scope_.replace(symbolic::symbol(argument), symbolic::symbol(argument_device));
        }
    }

    // Copy-Out & free
    for (auto& [argument, type, arg_size, page_size, alloc_size, meta] : transfer_args) {
        if (!meta.is_ptr) {
            continue;
        }
        auto argument_device = copy_prefix() + argument;
        auto& new_block = builder_.add_block_after(require_parent_scope(), scope_, {}, scope_.debug_info());
        if (meta.is_output) {
            copy_from_device(builder_, scope_, new_block, argument, argument_device, arg_size, page_size);
        } else {
            deallocate_device_arg(builder_, scope_, new_block, argument_device, arg_size, page_size);
        }
    }

    // Deallocate. Nothing to do, shared_ptr
    //    auto& dealloc_block = builder.add_block_after(parent_scope, copy_out_block, {}, this->map_.debug_info());
    //    for (auto& argument : allocated_device_args) {
    //        deallocate_device_arg(builder, dealloc_block, argument);
    //    }
}

std::string TenstorrentOffloadingExpansion::name() const { return "TenstorrentMovementExpansion"; }

} // namespace sdfg::tenstorrent
