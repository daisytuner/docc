#pragma once

#include <utility>

#include "docc/target/tenstorrent/plugin.h"
#include "docc/target/tenstorrent/tenstorrent_offloading_expansion.h"
#include "docc/target/tenstorrent/tenstorrent_transfer_arg.h"
#include "sdfg/transformations/offloading/offload_transform.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace tenstorrent {

/// 4K pages often fits with TT. 32 vars in a vector. This is 32 vectors at 4 byte size. 64 vectors at 2 bytes
const int TENSTORRENT_DEFAULT_BLOCK_SIZE_BYTES = 4096;

class TransformPlan;

class TenstorrentTransform : public transformations::OffloadTransform, TenstorrentOffloadingExpansion {
public:
    explicit TenstorrentTransform(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysisManager,
        structured_control_flow::Map& map,
        bool force_synchronous = false,
        bool allow_dynamic_sizes = false
    )
        : OffloadTransform(map, allow_dynamic_sizes),
          TenstorrentOffloadingExpansion(builder, analysisManager, map, force_synchronous, nullptr) {}

    std::string name() const override;

    bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    bool try_apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    std::unique_ptr<TransformPlan>
    try_create_transform_plan(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);

    void apply_plan(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        std::unique_ptr<TransformPlan> plan
    );

    static TenstorrentTransform from_json(
        builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager, const nlohmann::json& j
    ) {
        size_t map_id;
        if (j.contains("subgraph")) {
            const auto& node_desc = j.at("subgraph").at("0");
            map_id = node_desc.at("element_id").get<size_t>();
        } else if (j.contains("map_element_id")) {
            map_id = j.at("map_element_id").get<size_t>();
        } else {
            throw transformations::InvalidTransformationDescriptionException(
                "TenstorrentTransform descriptor missing 'subgraph' or 'map_element_id'."
            );
        }

        auto element = builder.find_element_by_id(map_id);
        if (!element) {
            throw transformations::
                InvalidTransformationDescriptionException("Element with ID " + std::to_string(map_id) + " not found.");
        }
        auto* map = dynamic_cast<structured_control_flow::Map*>(element);
        if (!map) {
            throw transformations::InvalidTransformationDescriptionException(
                "Element with ID " + std::to_string(map_id) + " is not a Map."
            );
        }

        bool force_synchronous = false;
        bool allow_dynamic_sizes = false;
        if (j.contains("parameters")) {
            const auto& params = j.at("parameters");
            if (params.contains("force_synchronous")) {
                force_synchronous = params.at("force_synchronous").get<bool>();
            }
            if (params.contains("allow_dynamic_sizes")) {
                allow_dynamic_sizes = params.at("allow_dynamic_sizes").get<bool>();
            }
        }

        return TenstorrentTransform(builder, analysis_manager, *map, force_synchronous, allow_dynamic_sizes);
    }

    void to_json(nlohmann::json& j) const override {
        j["transformation_type"] = this->name();

        j["subgraph"] = {{"0", {{"element_id", this->map_.element_id()}, {"type", "map"}}}};
        j["parameters"] = {
            {"force_synchronous", this->force_synchronous_}, {"allow_dynamic_sizes", this->allow_dynamic_sizes_}
        };
    }

    void set_report(sdfg::PassReportConsumer* report) override;

protected:
    using transformations::Transformation::report_;

    types::StorageType local_device_storage_type() override {
        return TenstorrentOffloadingExpansion::local_device_storage_type();
    }

    types::StorageType global_device_storage_type(symbolic::Expression arg_size) override {
        throw std::runtime_error("Global memory always requires page size on Tenstorrent");
    }

    ScheduleType transformed_schedule_type() override { return ScheduleType_Tenstorrent_Kernel::create(); }

    std::string copy_prefix() override { return TenstorrentOffloadingExpansion::copy_prefix(); }

    void setup_device(builder::StructuredSDFGBuilder& builder, Block& global_alloc_block) override;
    void teardown_device(builder::StructuredSDFGBuilder& builder, Block& global_alloc_block) override;

    void add_device_buffer(
        builder::StructuredSDFGBuilder& builder,
        std::string host_arg_name,
        std::string device_arg_name,
        symbolic::Expression arg_size
    ) override;

    void allocate_device_arg(
        builder::StructuredSDFGBuilder& builder,
        Block& alloc_block,
        std::string host_arg_name,
        std::string device_arg_name,
        symbolic::Expression arg_size,
        symbolic::Expression page_size
    ) override {
        TenstorrentOffloadingExpansion::
            allocate_device_arg(builder, map_, alloc_block, host_arg_name, device_arg_name, arg_size, page_size);
    }

    void deallocate_device_arg(
        builder::StructuredSDFGBuilder& builder,
        Block& dealloc_block,
        std::string device_arg_name,
        symbolic::Expression arg_size,
        symbolic::Expression page_size
    ) override {
        TenstorrentOffloadingExpansion::
            deallocate_device_arg(builder, map_, dealloc_block, device_arg_name, arg_size, page_size);
    }

    void copy_to_device(
        builder::StructuredSDFGBuilder& builder,
        std::string host_arg_name,
        std::string device_arg_name,
        symbolic::Expression size,
        symbolic::Expression page_size,
        Block& copy_block
    ) override {
        TenstorrentOffloadingExpansion::
            copy_to_device(builder, map_, host_arg_name, device_arg_name, size, page_size, copy_block);
    }

    void copy_to_device_with_allocation(
        builder::StructuredSDFGBuilder& builder,
        std::string host_arg_name,
        std::string device_arg_name,
        symbolic::Expression size,
        symbolic::Expression page_size,
        Block& copy_block
    ) override {
        TenstorrentOffloadingExpansion::
            copy_to_device_with_allocation(builder, map_, host_arg_name, device_arg_name, size, copy_block, page_size);
    }

    void copy_from_device(
        builder::StructuredSDFGBuilder& builder,
        Block& copy_out_block,
        std::string host_arg_name,
        std::string device_arg_name,
        symbolic::Expression size,
        symbolic::Expression page_size
    ) override {
        TenstorrentOffloadingExpansion::
            copy_from_device(builder, map_, copy_out_block, host_arg_name, device_arg_name, size, page_size);
    }

    void copy_from_device_with_free(
        builder::StructuredSDFGBuilder& builder,
        Block& copy_out_block,
        std::string host_arg_name,
        std::string device_arg_name,
        symbolic::Expression size,
        symbolic::Expression page_size
    ) override {
        TenstorrentOffloadingExpansion::
            copy_from_device(builder, map_, copy_out_block, host_arg_name, device_arg_name, size, page_size);
    }
};


class TransformPlan {
    friend class TenstorrentTransform;

private:
    std::vector<TransferArg> transferred_arguments_;
    std::vector<std::tuple<std::string, const types::IType&>> scalar_args_;
    std::vector<std::string> locals_;

    uint32_t tile_entries_;
};

} // namespace tenstorrent
} // namespace sdfg
