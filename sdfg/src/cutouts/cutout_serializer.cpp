#include "sdfg/cutouts/cutout_serializer.h"

#include <cassert>

#include "sdfg/analysis/arguments_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/codegen/utils.h"


#include "symengine/expression.h"
#include "symengine/logic.h"
#include "symengine/symengine_rcp.h"

namespace sdfg {
namespace serializer {

nlohmann::json CutoutSerializer::serialize(
    const sdfg::StructuredSDFG& sdfg,
    analysis::AnalysisManager* analysis_manager,
    structured_control_flow::Sequence* cutout_root
) {
    nlohmann::json j;

    j["name"] = sdfg.name() + std::to_string(cutout_root->element_id());
    j["element_counter"] = sdfg.element_counter();
    j["type"] = std::string(sdfg.type().value());

    // return type is always void
    nlohmann::json return_type_json;
    sdfg::types::Scalar return_type(sdfg::types::StorageType::CPU_Stack(), 0, "", sdfg::types::PrimitiveType::Void);
    type_to_json(return_type_json, return_type);
    j["return_type"] = return_type_json;

    j["structures"] = nlohmann::json::array();
    for (const auto& structure_name : sdfg.structures()) {
        const auto& structure = sdfg.structure(structure_name);
        nlohmann::json structure_json;
        structure_definition_to_json(structure_json, structure);
        j["structures"].push_back(structure_json);
    }

    auto& users_analysis = analysis_manager->get<analysis::Users>();
    sdfg::analysis::UsersView node_users(users_analysis, *cutout_root);

    // Get region arguments
    auto& arguments_analysis = analysis_manager->get<analysis::ArgumentsAnalysis>();
    auto& arguments = arguments_analysis.arguments(*analysis_manager, *cutout_root);

    j["containers"] = nlohmann::json::object();
    std::unordered_set<std::string> cutout_containers;
    for (const auto& use : node_users.uses()) {
        nlohmann::json desc;
        std::string& container = use->container();


        if ((arguments.find(container) != arguments.end()) && (arguments.at(container).is_scalar)) {
            type_to_json(desc, sdfg::codegen::Reference(sdfg.type(container)));
        } else {
            type_to_json(desc, sdfg.type(container));
        }
        j["containers"][container] = desc;
        cutout_containers.insert(container);
    }

    j["arguments"] = nlohmann::json::array();
    for (const auto& argument : arguments) {
        // Externals stay externals in the cutout (preserved below with their
        // linkage). The argument/external sets must be disjoint, otherwise the
        // deserializer throws "Container <name> already exists".
        if (sdfg.is_external(argument.first)) {
            continue;
        }
        j["arguments"].push_back(argument.first);
        j["containers"][argument.first]["storage_type"]["allocation"] = 0;
        j["containers"][argument.first]["storage_type"]["deallocation"] = 0;
    }

    j["externals"] = nlohmann::json::array();
    for (const auto& external : sdfg.externals()) {
        if (cutout_containers.find(external) == cutout_containers.end()) {
            continue;
        }
        nlohmann::json external_json;
        external_json["name"] = external;
        external_json["linkage_type"] = sdfg.linkage_type(external);
        j["externals"].push_back(external_json);
    }

    j["metadata"] = nlohmann::json::object();

    // Walk the SDFG
    nlohmann::json root_json;
    sequence_to_json(root_json, *cutout_root);
    j["root"] = root_json;

    return j;
}

} // namespace serializer
} // namespace sdfg
