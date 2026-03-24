#include "sdfg/passes/extended_data_transfer_minimization_pass.h"

#include <string>
#include <utility>

#include "sdfg/analysis/analysis.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/code_node.h"
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/element.h"
#include "sdfg/exceptions.h"
#include "sdfg/passes/offloading/data_transfer_minimization_pass.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/targets/cuda/cuda_data_offloading_node.h"
#include "sdfg/targets/offloading/data_offloading_node.h"
#include "sdfg/targets/offloading/external_offloading_node.h"
#include "sdfg/targets/tenstorrent/tenstorrent_offloading_node.h"

namespace sdfg {
namespace passes {

ExtendedDataTransferMinimization::ExtendedDataTransferMinimization(
    builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager
)
    : sdfg::passes::DataTransferMinimization(builder, analysis_manager) {}

std::pair<data_flow::AccessNode*, data_flow::AccessNode*> ExtendedDataTransferMinimization::
    get_src_and_dst(data_flow::DataFlowGraph& dfg, offloading::DataOffloadingNode* offloading_node) {
    if (!offloading_node->has_transfer()) {
        throw InvalidSDFGException(
            "ExtendedDataTransferMinimization: Cannot get copy access nodes for offloading node without data transfers"
        );
    }
    data_flow::AccessNode *src, *dst;
    if (dynamic_cast<cuda::CUDADataOffloadingNode*>(offloading_node) ||
        dynamic_cast<tenstorrent::TTDataOffloadingNode*>(offloading_node)) {
        src = this->get_in_access(offloading_node, "_src");
        dst = this->get_out_access(offloading_node, "_dst");
    } else if (auto* external_offload = dynamic_cast<offloading::ExternalDataOffloadingNode*>(offloading_node)) {
        if (external_offload->is_d2h()) {
            src = this->get_in_access(offloading_node, external_offload->inputs().back());
            dst = this->get_out_access(offloading_node, external_offload->input(external_offload->transfer_index()));
        } else {
            src = this->get_in_access(offloading_node, external_offload->input(external_offload->transfer_index()));
            dst = this->get_out_access(offloading_node, "_ret");
        }
    } else {
        throw InvalidSDFGException(
            "ExtendedDataTransferMinimization: Unknown offloading node encountered: " + offloading_node->code().value()
        );
    }
    return {src, dst};
}

} // namespace passes
} // namespace sdfg
