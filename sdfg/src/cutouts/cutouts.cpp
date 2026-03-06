#include "sdfg/cutouts/cutouts.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/arguments_analysis.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/cutouts/cutout_serializer.h"
#include "sdfg/structured_sdfg.h"

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
    auto& scope_analysis = local_analysis_manager.get<analysis::ScopeAnalysis>();

    structured_control_flow::Sequence* parent_scope = nullptr;
    structured_control_flow::Sequence* new_scope = nullptr;
    int index = -1;
    auto copied_node =
        static_cast<structured_control_flow::ControlFlowNode*>(builder.find_element_by_id(node.element_id()));
    assert(copied_node != nullptr);
    auto parent_element = scope_analysis.parent_scope(copied_node);
    assert(parent_element != nullptr);
    parent_scope = static_cast<structured_control_flow::Sequence*>(parent_element);
    assert(parent_scope != nullptr);
    index = parent_scope->index(*copied_node);

    new_scope = &builder.add_sequence_before(*parent_scope, *copied_node, {}, {});
    builder.move_child(*parent_scope, index + 1, *new_scope);
    local_analysis_manager.invalidate_all();
    sequence = new_scope;

    serializer::CutoutSerializer serializer;
    nlohmann::json cutout_json = serializer.serialize(builder.subject(), &local_analysis_manager, sequence);
    auto cutout_sdfg = serializer.deserialize(cutout_json);

    return cutout_sdfg;
}

} // namespace util
} // namespace sdfg
