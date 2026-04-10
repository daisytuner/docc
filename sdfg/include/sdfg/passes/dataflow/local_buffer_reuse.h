#pragma once

#include <string>
#include <typeinfo>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/math.h"
#include "sdfg/data_flow/library_nodes/stdlib/free.h"
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
class LocalBufferReuse : public visitor::NonStoppingStructuredSDFGVisitor {
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
    LocalBufferReuse(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        : visitor::NonStoppingStructuredSDFGVisitor(builder, analysis_manager) {}

    static std::string name() { return "LocalBufferReuse"; }

    virtual bool accept(structured_control_flow::Sequence& node) override {
        bool applied = false;

        auto& users_analysis = this->analysis_manager_.get<analysis::Users>();
        auto& scope_analysis = this->analysis_manager_.get<analysis::ScopeAnalysis>();

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

            auto* malloc_1 = blk1->is_a_library_node<stdlib::MallocNode>();
            if (malloc_1 == nullptr) {
                i++;
                continue;
            }

            auto* malloc_4 = blk4->is_a_library_node<stdlib::MallocNode>();
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
            T* lib_node_3 = dataflow_3.is_a_library_node<T>();
            if (lib_node_3 == nullptr) {
                i++;
                continue;
            }
            if (dataflow_3.out_degree(*lib_node_3) > 0) {
                bool found_ref_1 = false;
                for (auto& edge : dataflow_3.out_edges(*lib_node_3)) {
                    auto* access = dynamic_cast<const data_flow::AccessNode*>(&edge.dst());
                    if (access != nullptr && access->data() == ref_container_1) {
                        found_ref_1 = true;
                        break;
                    }
                }
                if (!found_ref_1) {
                    i++;
                    continue;
                }
            }

            auto& dataflow_5 = blk5->dataflow();
            std::string ref_container_2 = references(*blk5, malloc_container_4);
            if (ref_container_2.empty()) {
                i++;
                continue;
            }

            auto& dataflow_6 = blk6->dataflow();
            S* lib_node_6 = dataflow_6.is_a_library_node<S>();
            if (lib_node_6 == nullptr) {
                i++;
                continue;
            }
            bool found_ref_1 = false;
            for (auto& edge : dataflow_6.in_edges(*lib_node_6)) {
                auto* access = dynamic_cast<const data_flow::AccessNode*>(&edge.src());
                if (access != nullptr && access->data() == ref_container_1) {
                    found_ref_1 = true;
                    break;
                }
            }
            if (!found_ref_1) {
                i++;
                continue;
            }
            if (dataflow_6.out_degree(*lib_node_6) > 0) {
                bool found_ref_2 = false;
                for (auto& edge : dataflow_6.out_edges(*lib_node_6)) {
                    auto* access = dynamic_cast<const data_flow::AccessNode*>(&edge.dst());
                    if (access != nullptr && access->data() == ref_container_2) {
                        found_ref_2 = true;
                        break;
                    }
                }
                if (!found_ref_2) {
                    i++;
                    continue;
                }
            } else {
                bool found_ref_2 = false;
                for (auto& edge : dataflow_6.in_edges(*lib_node_6)) {
                    auto* access = dynamic_cast<const data_flow::AccessNode*>(&edge.src());
                    if (access != nullptr && access->data() == ref_container_2) {
                        found_ref_2 = true;
                        break;
                    }
                }
                if (!found_ref_2) {
                    i++;
                    continue;
                }
            }

            // Malloc container 4 may only have malloc, ref, free
            auto users_4 = users_analysis.uses(malloc_container_4);
            if (users_4.size() != 4) {
                i++;
                continue;
            }
            structured_control_flow::Block* free_blk = nullptr;
            for (auto& user : users_4) {
                if (auto* access_node = dynamic_cast<data_flow::AccessNode*>(user->element())) {
                    auto& parent_graph = access_node->get_parent();
                    auto parent_blk = dynamic_cast<structured_control_flow::Block*>(parent_graph.get_parent());
                    if (parent_blk != nullptr &&
                        parent_blk->dataflow().is_a_library_node<stdlib::FreeNode>() != nullptr) {
                        free_blk = parent_blk;
                        break;
                    }
                }
            }
            if (free_blk == nullptr) {
                i++;
                continue;
            }
            auto parent_scope = dynamic_cast<structured_control_flow::Sequence*>(scope_analysis.parent_scope(free_blk));
            if (parent_scope == nullptr) {
                i++;
                continue;
            }

            // Remove block 3 and reuse block ref_1 as new source of ref_2
            for (auto& node : blk5->dataflow().nodes()) {
                auto* access = dynamic_cast<data_flow::AccessNode*>(&node);
                if (access != nullptr && access->data() == malloc_container_4) {
                    access->data(ref_container_1);
                }
            }
            builder_.remove_child(node, i + 3);

            // Remove free blk
            int free_blk_index = parent_scope->index(*free_blk);
            builder_.remove_child(*parent_scope, free_blk_index);

            DEBUG_PRINTLN("Eliminated tensor with containers " << malloc_container_1 << " and " << malloc_container_4);

            applied = true;
            i++;
        }

        return applied;
    }
};

typedef VisitorPass<LocalBufferReuse<math::tensor::ConvNode, math::tensor::BatchNormNode>> ConvBatchNormEliminationPass;
typedef VisitorPass<LocalBufferReuse<math::tensor::BatchNormNode, math::tensor::ReLUNode>> BatchNormReLUEliminationPass;

Pipeline local_buffer_reuse_pipeline() {
    Pipeline pipeline("LocalBufferReusePipeline");

    pipeline.register_pass<ConvBatchNormEliminationPass>();
    pipeline.register_pass<BatchNormReLUEliminationPass>();

    return pipeline;
}

} // namespace passes
} // namespace sdfg
