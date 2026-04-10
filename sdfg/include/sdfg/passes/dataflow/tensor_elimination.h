#pragma once

#include <string>
#include <typeinfo>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/math.h"
#include "sdfg/data_flow/library_nodes/stdlib/malloc.h"
#include "sdfg/helpers/helpers.h"
#include "sdfg/passes/dataflow/dead_reference_elimination.h"
#include "sdfg/passes/dataflow/reference_propagation.h"
#include "sdfg/passes/pass.h"
#include "sdfg/passes/pipeline.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {

template<class T, class S>
class TensorElimination : public visitor::NonStoppingStructuredSDFGVisitor {
private:
    std::string references(const structured_control_flow::Block& block, const std::string& container) {
        auto& dataflow = block.dataflow();
        if (dataflow.nodes().size() != 2) {
            return "";
        }
        if (dataflow.edges().size() != 1) {
            return "";
        }
        auto& edge = *dataflow.edges().begin();
        auto* access_src = dynamic_cast<const data_flow::AccessNode*>(&edge.src());
        if (access_src == nullptr || access_src->data() != container) {
            return "";
        }
        auto* access_dst = dynamic_cast<const data_flow::AccessNode*>(&edge.dst());
        if (access_dst == nullptr) {
            return "";
        }
        return access_dst->data();
    }

public:
    TensorElimination(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        : visitor::NonStoppingStructuredSDFGVisitor(builder, analysis_manager) {}

    static std::string name() { return "TensorElimination"; }

    virtual bool accept(structured_control_flow::Sequence& node) override {
        bool applied = false;

        int i = 0;
        while (i + 6 < node.size()) {
            auto* blk1 = dynamic_cast<structured_control_flow::Block*>(&node.at(i).first);
            if (blk1 == nullptr || !node.at(i).second.empty()) {
                i++;
                continue;
            }
            auto* blk2 = dynamic_cast<structured_control_flow::Block*>(&node.at(i + 1).first);
            if (blk2 == nullptr || !node.at(i + 1).second.empty()) {
                i++;
                continue;
            }
            auto* blk3 = dynamic_cast<structured_control_flow::Block*>(&node.at(i + 2).first);
            if (blk3 == nullptr || !node.at(i + 2).second.empty()) {
                i++;
                continue;
            }
            auto* blk4 = dynamic_cast<structured_control_flow::Block*>(&node.at(i + 3).first);
            if (blk4 == nullptr || !node.at(i + 3).second.empty()) {
                i++;
                continue;
            }
            auto* blk5 = dynamic_cast<structured_control_flow::Block*>(&node.at(i + 4).first);
            if (blk5 == nullptr || !node.at(i + 4).second.empty()) {
                i++;
                continue;
            }
            auto* blk6 = dynamic_cast<structured_control_flow::Block*>(&node.at(i + 5).first);
            if (blk6 == nullptr || !node.at(i + 5).second.empty()) {
                i++;
                continue;
            }

            auto* malloc_1 = blk1->is<stdlib::MallocNode>();
            if (malloc_1 == nullptr) {
                i++;
                continue;
            }

            auto* malloc_4 = blk4->is<stdlib::MallocNode>();
            if (malloc_4 == nullptr) {
                i++;
                continue;
            }

            // Criterion 1: malloc sizes must match
            if (!symbolic::eq(malloc_1->size(), malloc_4->size())) {
                i++;
                continue;
            }

            data_flow::DataFlowNode& dst_1 = (*blk1->dataflow().out_edges(*malloc_1).begin()).dst();
            std::string malloc_container_1 = dynamic_cast<const data_flow::AccessNode&>(dst_1).data();
            data_flow::DataFlowNode& dst_4 = (*blk4->dataflow().out_edges(*malloc_4).begin()).dst();
            std::string malloc_container_4 = dynamic_cast<const data_flow::AccessNode&>(dst_4).data();

            auto& dataflow_2 = blk2->dataflow();
            std::string ref_container_1 = references(*blk2, malloc_container_1);
            if (ref_container_1.empty()) {
                i++;
                continue;
            }

            auto& dataflow_3 = blk3->dataflow();
            T* lib_node_3 = dataflow_3.is<T>();
            if (lib_node_3 == nullptr) {
                i++;
                continue;
            }
            if (dataflow_3.out_degree(*lib_node_3) != 1) {
                i++;
                continue;
            }
            auto& memlet_3 = *dataflow_3.out_edges(*lib_node_3).begin();
            auto& access_node_3 = dynamic_cast<data_flow::AccessNode&>(memlet_3.dst());
            if (access_node_3.data() != ref_container_1) {
                i++;
                continue;
            }

            auto& dataflow_5 = blk5->dataflow();
            std::string ref_container_2 = references(*blk5, malloc_container_4);
            if (ref_container_2.empty()) {
                i++;
                continue;
            }

            auto& dataflow_6 = blk6->dataflow();
            S* lib_node_6 = dataflow_6.is<S>();
            if (lib_node_6 == nullptr) {
                i++;
                continue;
            }
            // auto& memlet_6_in = *dataflow_6.in_edges(*lib_node_6).begin();
            // auto& access_node_6_in = dynamic_cast<data_flow::AccessNode&>(memlet_6_in.src());
            // if (access_node_6_in.data() != ref_container_1) {
            //     i++;
            //     continue;
            // }
            // auto& memlet_6_out = *dataflow_6.out_edges(*lib_node_6).begin();
            // auto& access_node_6_out = dynamic_cast<data_flow::AccessNode&>(memlet_6_out.dst());
            // if (access_node_6_out.data() != ref_container_2) {
            //     i++;
            //     continue;
            // }

            // Remove block 2, 3 and reuse block 1's malloc for block 6's output
            for (auto& node : blk5->dataflow().nodes()) {
                auto* access = dynamic_cast<data_flow::AccessNode*>(&node);
                if (access != nullptr && access->data() == malloc_container_4) {
                    access->data(ref_container_1);
                }
            }
            builder_.remove_child(node, i + 3);
            applied = true;

            i++;
        }

        return applied;
    }
};

typedef VisitorPass<TensorElimination<math::tensor::ConvNode, math::tensor::BatchNormNode>> ConvBatchNormEliminationPass;
typedef VisitorPass<TensorElimination<math::tensor::ConvNode, math::tensor::ReLUNode>> ConvReLUEliminationPass;
typedef VisitorPass<TensorElimination<math::tensor::BatchNormNode, math::tensor::ReLUNode>> BatchNormReLUEliminationPass;

Pipeline tensor_elimination_pipeline() {
    Pipeline pipeline("TensorEliminationPipeline");

    pipeline.register_pass<ConvBatchNormEliminationPass>();
    pipeline.register_pass<ConvReLUEliminationPass>();
    pipeline.register_pass<BatchNormReLUEliminationPass>();

    return pipeline;
}

} // namespace passes
} // namespace sdfg
