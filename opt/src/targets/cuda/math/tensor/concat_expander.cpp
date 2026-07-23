#include "sdfg/targets/cuda/math/tensor/concat_expander.h"

#include <cstddef>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/concat_node.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/exceptions.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace offloading {

bool CudaConcatExpander::expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    return CudaConcatExpander::expand_concat_separately(builder, analysis_manager, this->node_);
}

bool CudaConcatExpander::expand_concat_separately(
    builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager, math::tensor::ConcatNode& node
) {
    auto& dfg = node.get_parent();
    auto* parent_block = dyn_cast<structured_control_flow::Block*>(dfg.get_parent());
    if (!parent_block) {
        return false;
    }
    auto* parent_sequence = dyn_cast<structured_control_flow::Sequence*>(parent_block->get_parent());
    if (!parent_sequence) {
        return false;
    }
    int parent_block_index = parent_sequence->index(*parent_block);
    auto& new_sequence = builder.add_sequence_before(*parent_sequence, *parent_block, parent_block->debug_info());

    types::Scalar indvar_type(types::PrimitiveType::UInt64);
    size_t num_tensors = node.inputs().size() - 1;
    const auto* iedge_result = dfg.in_edge_for_connector(node, node.result());
    if (!iedge_result) {
        throw InvalidSDFGException("ConcatNode: Cannot get in edge for connector: " + node.result());
    }
    const auto& iedge_result_src = static_cast<const data_flow::AccessNode&>(iedge_result->src());
    symbolic::Expression offset = symbolic::zero();

    for (size_t i = 0; i < num_tensors; i++) {
        structured_control_flow::Sequence* current_seq = &new_sequence;
        data_flow::Subset subset;
        subset.reserve(node.tensor_layouts()[i].dims());
        for (auto dim : node.tensor_layouts()[i].shape()) {
            auto indvar_container = builder.find_new_name("_i");
            builder.add_container(indvar_container, indvar_type);
            auto indvar = symbolic::symbol(indvar_container);
            subset.push_back(indvar);
            auto& map = builder.add_map(
                *current_seq,
                indvar,
                symbolic::Lt(indvar, dim),
                symbolic::zero(),
                symbolic::add(indvar, symbolic::one()),
                structured_control_flow::ScheduleType_Sequential::create(),
                parent_block->debug_info()
            );
            current_seq = &map.root();
        }

        const auto* iedge_tensor = dfg.in_edge_for_connector(node, node.input(i));
        if (!iedge_tensor) {
            throw InvalidSDFGException("ConcatNode: Cannot get in edge for connector: " + node.input(i));
        }
        const auto& iedge_tensor_src = static_cast<const data_flow::AccessNode&>(iedge_tensor->src());

        auto& block = builder.add_block(*current_seq, {}, parent_block->debug_info());
        auto& tensor_access = builder.add_access(block, iedge_tensor_src.data(), iedge_tensor_src.debug_info());
        auto& result_access = builder.add_access(block, iedge_result_src.data(), iedge_result_src.debug_info());
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"}, node.debug_info());

        builder.add_computational_memlet(
            block, tensor_access, tasklet, "_in", subset, iedge_tensor->base_type(), iedge_tensor->debug_info()
        );

        data_flow::Subset offset_subset(subset);
        offset_subset[node.dim()] = symbolic::add(subset[node.dim()], offset);
        builder.add_computational_memlet(
            block, tasklet, "_out", result_access, offset_subset, iedge_result->base_type(), iedge_result->debug_info()
        );
        offset = symbolic::add(offset, node.tensor_layouts()[i].get_dim(node.dim()));
    }

    // Clean up the original block
    builder.clear_code_node_legacy(*parent_block, node);
    builder.remove_child(*parent_sequence, parent_block_index + 1);

    return true;
}

} // namespace offloading
} // namespace sdfg
