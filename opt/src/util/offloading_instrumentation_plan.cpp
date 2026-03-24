#include "sdfg/util/offloading_instrumentation_plan.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/codegen/instrumentation/instrumentation_plan.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/library_nodes/math/math.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/targets/offloading/data_offloading_node.h"
#include "sdfg/targets/offloading/external_offloading_node.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg::auto_util {

class LibNodeFinder : public visitor::ActualStructuredSDFGVisitor {
private:
    std::vector<const offloading::DataOffloadingNode*> transfer_nodes;
    std::vector<const math::blas::BLASNode*> blas_nodes;

public:
    using visitor::ActualStructuredSDFGVisitor::visit;

    const std::vector<const offloading::DataOffloadingNode*>& get_transfer_nodes() const { return transfer_nodes; }
    const std::vector<const math::blas::BLASNode*>& get_blas_nodes() const { return blas_nodes; }

    bool visit(structured_control_flow::Block& node) override;
};

bool LibNodeFinder::visit(structured_control_flow::Block& node) {
    for (auto libNode : node.dataflow().library_nodes()) {
        if (auto d2h = dynamic_cast<offloading::DataOffloadingNode*>(libNode)) {
            transfer_nodes.push_back(d2h);
        }
        if (auto blasNode = dynamic_cast<math::blas::BLASNode*>(libNode)) {
            blas_nodes.push_back(blasNode);
        }
    }
    return false;
}

void add_offloading_instrumentations(codegen::InstrumentationPlan& plan, sdfg::StructuredSDFG& sdfg) {
    LibNodeFinder lib_node_finder;
    lib_node_finder.dispatch(sdfg.root());
    for (auto* lib_node : lib_node_finder.get_transfer_nodes()) {
        if (dynamic_cast<const offloading::ExternalDataOffloadingNode*>(lib_node) != nullptr) {
            continue;
        }
        plan.update(*lib_node, codegen::InstrumentationEventType::NONE);
    }
    for (auto* lib_node : lib_node_finder.get_blas_nodes()) {
        plan.update(*lib_node, codegen::InstrumentationEventType::NONE);
    }
}

} // namespace sdfg::auto_util
