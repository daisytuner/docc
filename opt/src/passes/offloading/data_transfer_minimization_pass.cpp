#include "sdfg/passes/offloading/data_transfer_minimization_pass.h"

#include <cstddef>
#include <string>
#include <unordered_set>
#include <utility>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/data_transfer_elimination_analysis.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/code_node.h"
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/element.h"
#include "sdfg/exceptions.h"
#include "sdfg/helpers/helpers.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/cuda/cuda_data_offloading_node.h"
#include "sdfg/targets/offloading/data_offloading_node.h"
#include "sdfg/targets/rocm/rocm_data_offloading_node.h"
#include "sdfg/types/pointer.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {

DataTransferMinimizationPass::DataTransferMinimizationPass() {}

bool DataTransferMinimizationPass::eliminate_malloc_first_transfer(
    builder::StructuredSDFGBuilder& builder, analysis::OffloadHolder& malloc_holder, analysis::OffloadHolder& copy_in
) {
    // Get all relevant information
    std::string copy_in_device_container = copy_in.dev_data->data();
    DebugInfo copy_in_dst_debinfo = copy_in.dev_data->debug_info();

    // leave the malloc itself, because we have not proven yet that there is no more d2H transfer that needs it
    // DDE needs to be able to find it

    auto* h2d_block = dynamic_cast<structured_control_flow::Block*>(copy_in.offload_node->get_parent().get_parent());
    builder.remove_memlet(*h2d_block, *copy_in.host_access);
    builder.remove_node(*h2d_block, *copy_in.host_data);
    copy_in.remove_h2d_parts();
    copy_in.offload_node->remove_h2d();

    return true;
}

bool DataTransferMinimizationPass::eliminate_redundant_d2h(
    builder::StructuredSDFGBuilder& builder, analysis::OffloadHolder& h2d, analysis::OffloadHolder& d2h
) {
    // Get all relevant information
    std::string copy_out_device_container = d2h.dev_data->data();
    DebugInfo copy_out_dst_debinfo = d2h.dev_data->debug_info();

    // leave the malloc itself, because we have not proven yet that there is no more d2H transfer that needs it
    // DDE needs to be able to find it

    auto* d2h_block = dynamic_cast<structured_control_flow::Block*>(d2h.offload_node->get_parent().get_parent());
    if (d2h.offload_node->is_d2h()) {
        if (d2h.offload_node->is_free()) {
            std::string out_conn = d2h.host_access->src_conn();
            auto access_type = d2h.host_access->base_type().clone();
            builder.remove_memlet(*d2h_block, *d2h.host_access);
            builder.remove_node(*d2h_block, *d2h.host_data);
            d2h.offload_node->remove_d2h();
            auto& free_write = builder.add_access(*d2h_block, copy_out_device_container, copy_out_dst_debinfo);
            builder.add_computational_memlet(
                *d2h_block, *d2h.offload_node, out_conn, free_write, {}, *access_type, copy_out_dst_debinfo
            );
        } else { // remove it entirely
            builder.clear_code_node_legacy(*d2h_block, *d2h.offload_node);
        }
        d2h.remove_d2h_parts();
        return true;
    } else {
        return false;
    }
}

bool DataTransferMinimizationPass::eliminate_transfer_pair(
    builder::StructuredSDFGBuilder& builder,
    analysis::OffloadHolder& copy_out,
    analysis::OffloadHolder& copy_in,
    bool remove_d2h
) {
    // Get all relevant information
    std::string copy_out_device_container = copy_out.dev_data->data();
    std::string copy_in_device_container = copy_in.dev_data->data();
    DebugInfo copy_out_src_debinfo = copy_out.dev_data->debug_info();
    DebugInfo copy_in_dst_debinfo = copy_in.dev_data->debug_info();

    bool remove_entirely = false;
    // Remove what you can remove
    if (!remove_d2h && copy_out.offload_node->is_free()) {
        if (copy_out.offload_node->is_d2h()) {
            copy_out.offload_node->remove_free();
        } else {
            remove_entirely = true;
        }
    } else if (remove_d2h) {
        remove_entirely = true;
    }
    if (remove_entirely) {
        auto* copy_out_block =
            dynamic_cast<structured_control_flow::Block*>(copy_out.offload_node->get_parent().get_parent());
        builder.clear_code_node_legacy(*copy_out_block, *copy_out.offload_node);
    }

    auto* copy_in_block = dynamic_cast<structured_control_flow::Block*>(copy_in.offload_node->get_parent().get_parent()
    );
    builder.clear_code_node_legacy(*copy_in_block, *copy_in.offload_node);

    // Maps the device pointers if necessary
    if (copy_out_device_container != copy_in_device_container) {
        auto& container_type = builder.subject().type(copy_out_device_container);
        auto ref_type = container_type.clone();
        auto& in_access = builder.add_access(*copy_in_block, copy_out_device_container, copy_out_src_debinfo);
        auto& out_access = builder.add_access(*copy_in_block, copy_in_device_container, copy_in_dst_debinfo);
        builder.add_reference_memlet(
            *copy_in_block,
            in_access,
            out_access,
            {symbolic::zero()},
            *ref_type,
            DebugInfo::merge(copy_out.offload_node->debug_info(), copy_in.offload_node->debug_info())
        );
    }

    return true;
}

bool DataTransferMinimizationPass::
    run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    analysis::DataTransferEliminationAnalysis transfer_analysis(builder.subject(), analysis_manager);
    transfer_analysis.run();

    int removed = 0;

    for (auto& [malloc_cand, first_h2d] : transfer_analysis.empty_malloc_candidates()) {
        auto& malloc_holder = *malloc_cand.offload;

        DEBUG_PRINTLN(
            "  Elim malloc: " << "#" << malloc_holder.malloc_node->element_id() << " -> "
                              << (malloc_holder.host_data ? malloc_holder.host_data->data() : "-") << " / "
                              << "h2d+malloc: #" << first_h2d.offload_node->element_id() << " "
                              << (first_h2d.host_data ? first_h2d.host_data->data() : "-") << " -> "
                              << first_h2d.dev_data->data()
        );

        bool success = eliminate_malloc_first_transfer(builder, malloc_holder, first_h2d);

        if (success) {
            ++removed;
        }
    }

    for (auto& [h2d_cand, redundant_d2h] : transfer_analysis.redundant_d2h_candidates()) {
        auto& h2d_holder = *h2d_cand.offload;

        if (h2d_holder.updates_on_dev) {
            DEBUG_PRINTLN(
                "  Elim h2d: " << "#" << h2d_holder.offload_node->element_id() << " -> " << h2d_holder.host_data->data()
                               << " -> " << h2d_holder.dev_data->data() << " / "
                               << "clean d2h: #" << redundant_d2h.offload_node->element_id() << " "
                               << redundant_d2h.dev_data->data() << " -> " << redundant_d2h.host_data->data()
            );
            bool success = eliminate_redundant_d2h(builder, h2d_holder, redundant_d2h);

            if (success) {
                ++removed;
            }
        }
    }

    for (auto& candidate : transfer_analysis.transfer_reuse_candidates()) {
        auto reads = candidate.first.read_count;
        auto& copy_out = *candidate.first.offload;
        auto& copy_in = candidate.second;
        auto& copy_in_container = copy_in.host_data->data();

        DEBUG_PRINTLN(
            "  Elim hd2: #" << copy_out.offload_node->element_id() << " " << copy_out.dev_data->data() << " -> "
                            << (copy_out.host_data ? copy_out.host_data->data() : "-") << " / "
                            << "d2h: #" << copy_in.offload_node->element_id() << " "
                            << (copy_in.host_data ? copy_in.host_data->data() : "-") << " -> "
                            << copy_in.dev_data->data()
        );

        bool success = eliminate_transfer_pair(builder, copy_out, copy_in, false);

        if (success) {
            ++removed;
        }
    }

    return removed > 0;
}

DataTransferMinimizationLegacy::
    DataTransferMinimizationLegacy(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
    : visitor::NonStoppingStructuredSDFGVisitor(builder, analysis_manager) {}

bool DataTransferMinimizationLegacy::visit() {
    DEBUG_PRINTLN("Running DataTransferMinimizationPass on " << this->builder_.subject().name());
    return visitor::NonStoppingStructuredSDFGVisitor::visit();
}

bool DataTransferMinimizationLegacy::accept(structured_control_flow::Sequence& sequence) {
    bool applied = false;
    offloading::DataOffloadingNode* copy_out = nullptr;
    structured_control_flow::Block* copy_out_block = nullptr;
    size_t copy_out_index = 0;

    // While a copy-out can be found:
    while (copy_out_index < sequence.size()) {
        // Find a new copy-out
        for (; copy_out_index < sequence.size(); copy_out_index++) {
            if (auto* block = dynamic_cast<structured_control_flow::Block*>(&sequence.at(copy_out_index).first)) {
                if (block->dataflow().library_nodes().size() == 1 && block->dataflow().tasklets().size() == 0) {
                    auto* libnode = *block->dataflow().library_nodes().begin();
                    if (auto* offloading_node = dynamic_cast<offloading::DataOffloadingNode*>(libnode)) {
                        if (offloading_node->is_d2h()) {
                            copy_out = offloading_node;
                            copy_out_block = block;
                            break;
                        }
                    }
                }
            }
        }

        // Find a matching copy-in
        size_t i;
        for (i = copy_out_index; i < sequence.size(); i++) {
            // Child must be a block
            auto* copy_in_block = dynamic_cast<structured_control_flow::Block*>(&sequence.at(i).first);
            if (!copy_in_block) {
                continue;
            }

            // Block must contain exactly one library node
            if (copy_in_block->dataflow().library_nodes().size() != 1 ||
                copy_in_block->dataflow().tasklets().size() != 0) {
                continue;
            }

            // Library node must be an offloading node
            auto* copy_in =
                dynamic_cast<offloading::DataOffloadingNode*>(*copy_in_block->dataflow().library_nodes().begin());
            if (!copy_in) {
                continue;
            }

            // Offloading node must be a copy-in
            if (!copy_in->is_h2d()) {
                continue;
            }

            // Copy-in and copy-out must be redundant
            if (!copy_out->redundant_with(*copy_in)) {
                continue;
            }

            // Get src and dst access nodes for copy-in & -out
            auto [copy_out_src, copy_out_dst] = this->get_src_and_dst(copy_out_block->dataflow(), copy_out);
            auto [copy_in_src, copy_in_dst] = this->get_src_and_dst(copy_in_block->dataflow(), copy_in);

            // Get the write and read users
            auto& users = this->analysis_manager_.get<analysis::Users>();
            analysis::User* write = users.get_user(copy_out_dst->data(), copy_out_dst, analysis::Use::WRITE);
            if (!write) {
                continue;
            }
            analysis::User* read = users.get_user(copy_in_src->data(), copy_in_src, analysis::Use::READ);
            if (!read) {
                continue;
            }

            if (copy_out_dst->data() == copy_in_src->data()) {
                // Ensure that the container is not written between the data transfer nodes
                bool used_between = false;
                for (auto* user : users.all_uses_between(*write, *read)) {
                    if (user->container() == copy_out_dst->data() && user->use() != analysis::Use::READ) {
                        used_between = true;
                        break;
                    }
                }
                if (used_between) {
                    continue;
                }
            } else {
                if (!this->check_container_dependency(
                        copy_out_block, copy_out_dst->data(), copy_in_block, copy_in_src->data()
                    )) {
                    continue;
                }
            }

            // Check that the container is not written after the data transfer nodes
            bool read_after = false;
            for (auto* user : users.all_uses_after(*write)) {
                if (user->container() == copy_out_dst->data() && user->use() == analysis::Use::READ && user != read) {
                    read_after = true;
                    break;
                }
            }

            // Debug output
            DEBUG_PRINTLN(
                "  Eliminating " << (read_after ? "(" : "") << "copy-out: #" << copy_out->element_id() << " "
                                 << copy_out_src->data() << " -> " << copy_out_dst->data() << (read_after ? ")" : "")
                                 << " / copy-in: #" << copy_in->element_id() << " " << copy_in_src->data() << " -> "
                                 << copy_in_dst->data()
            );

            // Get all relevant information
            std::string copy_out_device_container = copy_out_src->data();
            std::string copy_in_device_container = copy_in_dst->data();
            DebugInfo copy_out_src_debinfo = copy_out_src->debug_info();
            DebugInfo copy_in_dst_debinfo = copy_in_dst->debug_info();

            // Remove the data tranfers
            if (read_after && copy_out->is_free()) {
                copy_out->remove_free();
            } else if (!read_after) {
                this->builder_.clear_code_node_legacy(*copy_out_block, *copy_out);
            }
            this->builder_.clear_code_node_legacy(*copy_in_block, *copy_in);

            // Maps the device pointers if necessary
            if (copy_out_device_container != copy_in_device_container) {
                auto& container_type = this->builder_.subject().type(copy_out_device_container);
                auto ref_type = container_type.clone();
                auto& in_access =
                    this->builder_.add_access(*copy_in_block, copy_out_device_container, copy_out_src_debinfo);
                auto& out_access =
                    this->builder_.add_access(*copy_in_block, copy_in_device_container, copy_in_dst_debinfo);
                this->builder_.add_reference_memlet(
                    *copy_in_block,
                    in_access,
                    out_access,
                    {symbolic::zero()},
                    *ref_type,
                    DebugInfo::merge(copy_out->debug_info(), copy_in->debug_info())
                );
            }

            // Invalidate users analysis
            this->analysis_manager_.invalidate<analysis::Users>();
            applied = true;
            break;
        }

        // Skip if no matching copy-in was found
        if (i >= sequence.size()) {
            copy_out_index++;
        }
    }

    return applied;
}

std::pair<data_flow::AccessNode*, data_flow::AccessNode*> DataTransferMinimizationLegacy::
    get_src_and_dst(data_flow::DataFlowGraph& dfg, offloading::DataOffloadingNode* offloading_node) {
    if (!offloading_node->has_transfer()) {
        throw InvalidSDFGException(
            "DataTransferMinimization: Cannot get copy access nodes for offloading node without data transfers"
        );
    }
    data_flow::AccessNode *src, *dst;
    if (dynamic_cast<cuda::CUDADataOffloadingNode*>(offloading_node)) {
        src = this->get_in_access(offloading_node, "_src");
        dst = this->get_out_access(offloading_node, "_dst");
    } else if (dynamic_cast<rocm::ROCMDataOffloadingNode*>(offloading_node)) {
        src = this->get_in_access(offloading_node, "_src");
        dst = this->get_out_access(offloading_node, "_dst");
    } else {
        throw InvalidSDFGException(
            "DataTransferMinimization: Unknown offloading node encountered: " + offloading_node->code().value()
        );
    }
    return {src, dst};
}

data_flow::AccessNode* DataTransferMinimizationLegacy::
    get_in_access(data_flow::CodeNode* node, const std::string& dst_conn) {
    auto& dfg = node->get_parent();
    for (auto& iedge : dfg.in_edges(*node)) {
        if (iedge.dst_conn() == dst_conn) {
            return dynamic_cast<data_flow::AccessNode*>(&iedge.src());
        }
    }
    return nullptr;
}

data_flow::AccessNode* DataTransferMinimizationLegacy::
    get_out_access(data_flow::CodeNode* node, const std::string& src_conn) {
    auto& dfg = node->get_parent();
    for (auto& oedge : dfg.out_edges(*node)) {
        if (oedge.src_conn() == src_conn) {
            return static_cast<data_flow::AccessNode*>(&oedge.dst());
        }
    }
    return nullptr;
}

bool DataTransferMinimizationLegacy::check_container_dependency(
    structured_control_flow::Block* copy_out_block,
    const std::string& copy_out_container,
    structured_control_flow::Block* copy_in_block,
    const std::string& copy_in_container
) {
    // Simplification: Assume blocks are in the same sequence
    auto& scope_analysis = this->analysis_manager_.get<analysis::ScopeAnalysis>();
    auto* copy_out_block_parent = scope_analysis.parent_scope(copy_out_block);
    auto* copy_in_block_parent = scope_analysis.parent_scope(copy_in_block);
    auto* sequence = dynamic_cast<structured_control_flow::Sequence*>(copy_out_block_parent);
    if (copy_out_block_parent != copy_in_block_parent || !sequence) {
        return false;
    }

    std::unordered_set<std::string> copy_out_container_captures, copy_in_container_parts;
    size_t start = sequence->index(*copy_out_block);
    size_t stop = sequence->index(*copy_in_block);
    for (size_t i = start + 1; i < stop; i++) {
        auto* block = dynamic_cast<structured_control_flow::Block*>(&sequence->at(i).first);
        if (!block) {
            continue;
        }

        auto& dfg = block->dataflow();
        for (auto* access_node : dfg.data_nodes()) {
            if (access_node->data() == copy_in_container) {
                // Only allow constant assignments
                for (auto& iedge : dfg.in_edges(*access_node)) {
                    auto* tasklet = dynamic_cast<data_flow::Tasklet*>(&iedge.src());
                    if (!tasklet || tasklet->code() != data_flow::TaskletCode::assign) {
                        continue;
                    }

                    auto& iedge2 = *dfg.in_edges(*tasklet).begin();
                    if (!dynamic_cast<data_flow::ConstantNode*>(&iedge2.src())) {
                        return false;
                    }
                }

                // Collect H2D container parts
                for (auto& oedge : dfg.out_edges(*access_node)) {
                    if (oedge.type() != data_flow::MemletType::Reference) {
                        continue;
                    }

                    auto* access_node2 = dynamic_cast<data_flow::AccessNode*>(&oedge.dst());
                    if (!access_node2) {
                        continue;
                    }

                    copy_in_container_parts.insert(access_node2->data());
                }
            } else if (access_node->data() == copy_out_container) {
                // Collect D2H container captures
                for (auto& oedge : dfg.out_edges(*access_node)) {
                    if (oedge.type() != data_flow::MemletType::Dereference_Dst) {
                        continue;
                    }

                    auto* access_node2 = dynamic_cast<data_flow::AccessNode*>(&oedge.dst());
                    if (!access_node2) {
                        continue;
                    }

                    copy_out_container_captures.insert(access_node2->data());
                }
            }
        }
    }

    // Find all matches between captures and parts
    size_t matches = 0;
    for (auto& capture : copy_out_container_captures) {
        for (auto& part : copy_in_container_parts) {
            if (capture == part) {
                matches++;
            }
        }
    }

    return (matches == 1);
}

} // namespace passes
} // namespace sdfg
