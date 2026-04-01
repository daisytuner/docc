#pragma once

#include <sdfg/builder/structured_sdfg_builder.h>
#include <sdfg/data_flow/library_nodes/math/blas/blas_node.h>
#include <sdfg/optimization_report/pass_report_consumer.h>
#include "docc/target/tenstorrent/storage.h"
#include "tenstorrent_transfer_arg.h"

namespace sdfg::tenstorrent {

inline std::string TENSTORRENT_DEVICE_VAR_PREFIX = "__daisy_tt_";

class TenstorrentOffloadingExpansion {
protected:
    builder::StructuredSDFGBuilder& builder_;
    analysis::AnalysisManager& analysis_mgr_;
    structured_control_flow::ControlFlowNode& scope_;
    std::string device_handle_ = "tt_device";
    bool force_synchronous_;
    PassReportConsumer* report_;

    structured_control_flow::Sequence* parent_scope_ = nullptr;
    structured_control_flow::Block* copy_in_block_ = nullptr;
    structured_control_flow::Block* alloc_block_ = nullptr;
    structured_control_flow::Block* copy_out_block_ = nullptr;

public:
    TenstorrentOffloadingExpansion(
        sdfg::builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_mgr,
        structured_control_flow::ControlFlowNode& scope,
        bool force_synchronous = false,
        sdfg::PassReportConsumer* report = nullptr
    )
        : builder_(builder), analysis_mgr_(analysis_mgr), scope_(scope), force_synchronous_(force_synchronous),
          report_(report) {}

    std::string name() const;

    void set_report(sdfg::PassReportConsumer* report);

    void expand_blas(sdfg::math::blas::BLASNode& node);

protected:
    virtual types::StorageType local_device_storage_type() {
        return types::StorageType(
            StorageType_Tenstorrent_Local.value(),
            SymEngine::null,
            types::StorageType::AllocationType::Unmanaged,
            types::StorageType::AllocationType::Unmanaged
        );
    }

    virtual types::StorageType global_device_storage_type(symbolic::Expression arg_size, symbolic::Expression page_size) {
        types::StorageType
            st(StorageType_Tenstorrent_DRAM.value(),
               arg_size,
               types::StorageType::AllocationType::Unmanaged,
               types::StorageType::AllocationType::Unmanaged);
        st.arg1(page_size);
        return st;
    }

    virtual std::string copy_prefix() { return TENSTORRENT_DEVICE_VAR_PREFIX; }

    void allocate_device_arg(
        builder::StructuredSDFGBuilder& builder,
        const ControlFlowNode& org,
        Block& alloc_block,
        std::string host_arg_name,
        std::string device_arg_name,
        symbolic::Expression arg_size,
        symbolic::Expression page_size
    );

    void deallocate_device_arg(
        builder::StructuredSDFGBuilder& builder,
        const ControlFlowNode& org,
        Block& dealloc_block,
        std::string device_arg_name,
        symbolic::Expression arg_size,
        symbolic::Expression page_size
    );

    void copy_to_device(
        builder::StructuredSDFGBuilder& builder,
        const ControlFlowNode& org,
        const std::string host_arg_name,
        std::string device_arg_name,
        symbolic::Expression size,
        symbolic::Expression page_size,
        Block& copy_block
    );

    void copy_to_device_with_allocation(
        builder::StructuredSDFGBuilder& builder,
        const ControlFlowNode& org,
        const std::string host_arg_name,
        std::string device_arg_name,
        symbolic::Expression size,
        Block& block,
        symbolic::Expression page_size
    );

    void copy_from_device(
        builder::StructuredSDFGBuilder& builder,
        const ControlFlowNode& org,
        Block& copy_out_block,
        const std::string host_arg_name,
        std::string device_arg_name,
        symbolic::Expression size,
        symbolic::Expression page_size
    );

    structured_control_flow::Sequence& require_parent_scope();

    structured_control_flow::Block& require_copy_in_block();

    structured_control_flow::Block& require_copy_out_block();

    structured_control_flow::Block& require_allocate_block();

    void create_offloaded_memory_handling(std::vector<TransferArg>& transfer_args);
};

} // namespace sdfg::tenstorrent
