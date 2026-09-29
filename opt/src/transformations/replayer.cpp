#include <sdfg/transformations/replayer.h>

#include <iostream>

#include <sdfg/transformations/transformation_registry.h>

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
        auto transformation_name = desc["transformation_type"].get<std::string>();

        dispatch_transformation(transformation_name, desc, [&]<typename T>() {
#ifndef NDEBUG
            // Verify that from_json and to_json are consistent inverses for this description.
            std::string round_trip_error;
            if (!verify_round_trip<T>(builder, desc, round_trip_error)) {
                throw transformations::InvalidTransformationDescriptionException(
                    "Transformation '" + transformation_name +
                    "' failed from_json/to_json verification: " + round_trip_error
                );
            }
#endif
            this->apply<T>(builder, analysis_manager, desc, skip_if_not_applicable);
        });

#ifndef NDEBUG
        std::cout << "Applied transformation: " << transformation_name << std::endl;
        builder.subject().validate();
#endif

        analysis_manager.invalidate_all();
    }
}


} // namespace transformations
} // namespace sdfg
