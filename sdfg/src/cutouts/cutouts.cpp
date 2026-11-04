#include "sdfg/cutouts/cutouts.h"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/arguments_analysis.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/codegen/utils.h"
#include "sdfg/cutouts/cutout_serializer.h"
#include "sdfg/deepcopy/structured_sdfg_deep_copy.h"
#include "sdfg/metadata/loop_provenance.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace util {

std::unique_ptr<StructuredSDFG> cutout(
    sdfg::StructuredSDFG& sdfg,
    sdfg::analysis::AnalysisManager& analysis_manager,
    structured_control_flow::ControlFlowNode& node
) {
    auto local_sdfg = sdfg.clone();
    sdfg::builder::StructuredSDFGBuilder builder(local_sdfg);

    structured_control_flow::Sequence* sequence;
    sdfg::analysis::AnalysisManager local_analysis_manager(builder.subject());

    structured_control_flow::Sequence* parent_scope = nullptr;
    structured_control_flow::Sequence* new_scope = nullptr;
    int index = -1;
    auto copied_node =
        static_cast<structured_control_flow::ControlFlowNode*>(builder.find_element_by_id(node.element_id()));
    assert(copied_node != nullptr);
    auto parent_element = copied_node->get_parent();
    assert(parent_element != nullptr);
    parent_scope = static_cast<structured_control_flow::Sequence*>(parent_element);
    assert(parent_scope != nullptr);
    index = parent_scope->index(*copied_node);

    new_scope = &builder.add_sequence_before(*parent_scope, *copied_node);
    builder.move_child(*parent_scope, index + 1, *new_scope);
    local_analysis_manager.invalidate_all();
    sequence = new_scope;

    serializer::CutoutSerializer serializer;
    nlohmann::json cutout_json = serializer.serialize(builder.subject(), &local_analysis_manager, sequence);
    auto cutout_sdfg = serializer.deserialize(cutout_json);

    return cutout_sdfg;
}

void apply_cutout(builder::StructuredSDFGBuilder& builder, size_t element_id, StructuredSDFG& cutout_sdfg) {
    auto* node = dynamic_cast<structured_control_flow::ControlFlowNode*>(builder.find_element_by_id(element_id));
    if (node == nullptr) {
        throw std::runtime_error("No control-flow node with id " + std::to_string(element_id) + " found in SDFG");
    }

    auto* parent_scope = dynamic_cast<structured_control_flow::Sequence*>(node->get_parent());
    if (parent_scope == nullptr) {
        throw std::runtime_error("Element " + std::to_string(element_id) + " has no parent scope");
    }

    // Arguments and externals must keep their names to bind to the target's containers.
    builder::StructuredSDFGBuilder cutout_builder(cutout_sdfg);
    std::vector<std::string> cutout_containers;
    for (const auto& container : cutout_sdfg.containers()) {
        cutout_containers.push_back(container);
    }
    std::vector<std::pair<std::string, std::string>> renames;
    for (const auto& container : cutout_containers) {
        if (cutout_sdfg.is_argument(container) || cutout_sdfg.is_external(container)) {
            continue;
        }
        if (!builder.subject().exists(container)) {
            continue;
        }
        std::string new_name;
        size_t suffix = 0;
        do {
            new_name = container + "_" + std::to_string(suffix++);
        } while (builder.subject().exists(new_name) || cutout_sdfg.exists(new_name));
        renames.emplace_back(container, new_name);
    }
    cutout_builder.rename_containers(renames);

    for (auto& container : cutout_sdfg.containers()) {
        if (builder.subject().exists(container)) {
            continue;
        }
        auto& type = cutout_sdfg.type(container);
        if (type.type_id() == types::TypeID::Reference) {
            auto& reference_type = dynamic_cast<const codegen::Reference&>(type).reference_type();
            builder.add_container(container, reference_type, false, false);
        } else {
            builder.add_container(container, type, false, false);
        }
    }

    analysis::AnalysisManager cutout_analysis_manager(cutout_sdfg);
    auto& cutout_loop_analysis = cutout_analysis_manager.get<analysis::LoopAnalysis>();
    const auto cutout_outermost_loops = cutout_loop_analysis.outermost_loops();

    auto& staging = builder.add_sequence_before(*parent_scope, *node);
    auto copied_nodes = deepcopy::StructuredSDFGDeepCopy(builder, staging, cutout_sdfg.root()).insert();
    for (const auto* cutout_loop : cutout_outermost_loops) {
        auto copied = copied_nodes.find(cutout_loop);
        if (copied == copied_nodes.end()) {
            throw std::runtime_error("Deep copy did not return a mapping for an outermost cutout loop");
        }
        metadata::set_source_loop_id(*const_cast<structured_control_flow::ControlFlowNode*>(copied->second), element_id);
    }

    builder.move_children(staging, *parent_scope, parent_scope->index(staging));
    builder.remove_from_parent(staging);
    builder.remove_from_parent(*node);
}

} // namespace util
} // namespace sdfg
