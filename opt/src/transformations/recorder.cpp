#include <sdfg/tiles/transformations/tile_fusion.h>
#include <sdfg/transformations/loop_distribute.h>
#include <sdfg/transformations/loop_interchange.h>
#include <sdfg/transformations/loop_tiling.h>
#include <sdfg/transformations/recorder.h>
#include <sdfg/transformations/transformation_registry.h>

namespace sdfg {
namespace transformations {

void verify_transformation_replayable(builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc) {
    // Resolving the type through the registry throws for an unregistered name, so this both asserts
    // replayability and checks the from_json/to_json round-trip in one call.
    std::string error;
    if (!verify_transformation_round_trip(builder, desc, error)) {
        const std::string name =
            desc.contains("transformation_type") && desc.at("transformation_type").is_string()
                ? desc.at("transformation_type").get<std::string>()
                : "<unknown>";
        throw InvalidTransformationDescriptionException(
            "Transformation '" + name + "' failed replay verification: " + error
        );
    }
}

Recorder::Recorder() : history_(nlohmann::json::array()) {
}


void Recorder::save(std::filesystem::path path) const {
    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for saving transformations: " + path.string());
    }
    file << history_.dump(4); // Pretty print with an indent of 4 spaces
    file.close();
}

} // namespace transformations
} // namespace sdfg
