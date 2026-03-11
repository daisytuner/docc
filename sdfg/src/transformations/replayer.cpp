#include <sdfg/transformations/loop_distribute.h>
#include <sdfg/transformations/loop_interchange.h>
#include <sdfg/transformations/loop_tiling.h>
#include <sdfg/transformations/out_local_storage.h>
#include <sdfg/transformations/replayer.h>
#include <sdfg/transformations/tile_fusion.h>

namespace sdfg {
namespace transformations {

void Replayer::replay(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    const nlohmann::json& transformation_data,
    bool skip_if_not_applicable,
    size_t loopnest_index
) {
    if (!transformation_data.is_array()) {
        throw std::runtime_error("Transformation data must be an array.");
    }

    for (const auto& desc : transformation_data) {
        auto transformation_name = desc["transformation_type"];

        if (transformation_name == "LoopTiling") {
            this->apply<transformations::LoopTiling>(builder, analysis_manager, desc, skip_if_not_applicable);
        } else if (transformation_name == "LoopDistribute") {
            this->apply<transformations::LoopDistribute>(builder, analysis_manager, desc, skip_if_not_applicable);
        } else if (transformation_name == "LoopInterchange") {
            this->apply<transformations::LoopInterchange>(builder, analysis_manager, desc, skip_if_not_applicable);
        } else if (transformation_name == "OutLocalStorage") {
            this->apply<transformations::OutLocalStorage>(builder, analysis_manager, desc, skip_if_not_applicable);
        } else if (transformation_name == "TileFusion") {
            this->apply<transformations::TileFusion>(builder, analysis_manager, desc, skip_if_not_applicable);
        } else {
            throw transformations::InvalidTransformationDescriptionException(
                "Unknown transformation: " + transformation_name.get<std::string>()
            );
        }

#ifndef NDEBUG
        std::cout << "Applied transformation: " << transformation_name << std::endl;
        builder.subject().validate();
#endif

        analysis_manager.invalidate_all();
    }
}


} // namespace transformations
} // namespace sdfg
