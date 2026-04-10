#pragma once

#include <string>
#include <tuple>
#include <typeinfo>
#include <vector>

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

// Forward declarations for variadic template helpers
namespace detail {

// Helper to check if a block contains a specific library node type
template<typename LibNode>
bool block_has_lib_node(structured_control_flow::Block* blk) {
    return blk->dataflow().template is_a_library_node<LibNode>() != nullptr;
}

// Helper to get library node from block
template<typename LibNode>
LibNode* get_lib_node(structured_control_flow::Block* blk) {
    return blk->dataflow().template is_a_library_node<LibNode>();
}

// Check if lib node has ref_container in its outputs
template<typename LibNode>
bool lib_node_writes_to_ref(structured_control_flow::Block* blk, const std::string& ref_container) {
    auto* lib_node = get_lib_node<LibNode>(blk);
    if (lib_node == nullptr) return false;

    auto& dataflow = blk->dataflow();
    if (dataflow.out_degree(*lib_node) > 0) {
        for (auto& edge : dataflow.out_edges(*lib_node)) {
            auto* access = dynamic_cast<const data_flow::AccessNode*>(&edge.dst());
            if (access != nullptr && access->data() == ref_container) {
                return true;
            }
        }
        return false;
    }
    return true; // No output edges, assume ok for intermediate nodes
}

// Check if lib node has ref_container in its inputs
template<typename LibNode>
bool lib_node_reads_from_ref(structured_control_flow::Block* blk, const std::string& ref_container) {
    auto* lib_node = get_lib_node<LibNode>(blk);
    if (lib_node == nullptr) return false;

    auto& dataflow = blk->dataflow();
    for (auto& edge : dataflow.in_edges(*lib_node)) {
        auto* access = dynamic_cast<const data_flow::AccessNode*>(&edge.src());
        if (access != nullptr && access->data() == ref_container) {
            return true;
        }
    }
    return false;
}

// Check if lib node has ref_container in its outputs or inputs (for last node)
template<typename LibNode>
bool lib_node_uses_ref(structured_control_flow::Block* blk, const std::string& ref_container) {
    auto* lib_node = get_lib_node<LibNode>(blk);
    if (lib_node == nullptr) return false;

    auto& dataflow = blk->dataflow();
    if (dataflow.out_degree(*lib_node) > 0) {
        for (auto& edge : dataflow.out_edges(*lib_node)) {
            auto* access = dynamic_cast<const data_flow::AccessNode*>(&edge.dst());
            if (access != nullptr && access->data() == ref_container) {
                return true;
            }
        }
        return false;
    } else {
        for (auto& edge : dataflow.in_edges(*lib_node)) {
            auto* access = dynamic_cast<const data_flow::AccessNode*>(&edge.src());
            if (access != nullptr && access->data() == ref_container) {
                return true;
            }
        }
        return false;
    }
}

} // namespace detail

// Variadic template version supporting N library node types
template<class... LibNodes>
class LocalBufferReuseN : public visitor::NonStoppingStructuredSDFGVisitor {
private:
    static constexpr size_t N = sizeof...(LibNodes);

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

    // Get block at index, return nullptr if not valid
    structured_control_flow::Block* get_block(structured_control_flow::Sequence& seq, int idx) {
        if (idx >= static_cast<int>(seq.size())) return nullptr;
        auto* blk = dynamic_cast<structured_control_flow::Block*>(&seq.at(idx).first);
        if (blk == nullptr || !seq.at(idx).second.empty()) return nullptr;
        return blk;
    }

    // Helper to check lib node type at compile-time index
    template<size_t I>
    using LibNodeAt = typename std::tuple_element<I, std::tuple<LibNodes...>>::type;

    // Verify a single malloc-ref-lib triple at position k (0-indexed)
    // Returns: {malloc_container, ref_container} or {"", ""} on failure
    template<size_t K>
    std::pair<std::string, std::string>
    verify_triple(structured_control_flow::Sequence& seq, int base_idx, const std::string& prev_ref_container) {
        int malloc_idx = base_idx + K * 3;
        int ref_idx = malloc_idx + 1;
        int lib_idx = malloc_idx + 2;

        auto* malloc_blk = get_block(seq, malloc_idx);
        auto* ref_blk = get_block(seq, ref_idx);
        auto* lib_blk = get_block(seq, lib_idx);

        if (!malloc_blk || !ref_blk || !lib_blk) {
            return {"", ""};
        }

        auto* malloc_node = malloc_blk->template is_a_library_node<stdlib::MallocNode>();
        if (malloc_node == nullptr) {
            return {"", ""};
        }

        // Get malloc container
        auto& dst = (*malloc_blk->dataflow().out_edges(*malloc_node).begin()).dst();
        std::string malloc_container = dynamic_cast<const data_flow::AccessNode&>(dst).data();

        // Get ref container
        std::string ref_container = references(*ref_blk, malloc_container);
        if (ref_container.empty()) {
            return {"", ""};
        }

        // Check lib node type
        if (!detail::block_has_lib_node<LibNodeAt<K>>(lib_blk)) {
            return {"", ""};
        }

        // First lib node: just check it writes to its ref_container
        if constexpr (K == 0) {
            if (!detail::lib_node_writes_to_ref<LibNodeAt<K>>(lib_blk, ref_container)) {
                return {"", ""};
            }
        }
        // Last lib node: check it reads from prev ref and writes to its ref
        else if constexpr (K == N - 1) {
            if (!detail::lib_node_reads_from_ref<LibNodeAt<K>>(lib_blk, prev_ref_container)) {
                return {"", ""};
            }
            if (!detail::lib_node_uses_ref<LibNodeAt<K>>(lib_blk, ref_container)) {
                return {"", ""};
            }
        }
        // Middle lib nodes: check reads from prev, writes to current
        else {
            if (!detail::lib_node_reads_from_ref<LibNodeAt<K>>(lib_blk, prev_ref_container)) {
                return {"", ""};
            }
            if (!detail::lib_node_writes_to_ref<LibNodeAt<K>>(lib_blk, ref_container)) {
                return {"", ""};
            }
        }

        return {malloc_container, ref_container};
    }

    // Recursive verification of all triples
    template<size_t K>
    bool verify_all_triples(
        structured_control_flow::Sequence& seq,
        int base_idx,
        const symbolic::Expression& first_malloc_size,
        const std::string& prev_ref_container,
        std::vector<std::string>& malloc_containers,
        std::vector<std::string>& ref_containers
    ) {
        auto [malloc_container, ref_container] = verify_triple<K>(seq, base_idx, prev_ref_container);
        if (malloc_container.empty()) {
            return false;
        }

        // For K > 0, verify malloc sizes match
        if constexpr (K > 0) {
            int malloc_idx = base_idx + K * 3;
            auto* malloc_blk = get_block(seq, malloc_idx);
            auto* malloc_node = malloc_blk->template is_a_library_node<stdlib::MallocNode>();
            if (!symbolic::eq(first_malloc_size, malloc_node->size())) {
                return false;
            }
        }

        malloc_containers.push_back(malloc_container);
        ref_containers.push_back(ref_container);

        if constexpr (K + 1 < N) {
            return verify_all_triples<
                K + 1>(seq, base_idx, first_malloc_size, ref_container, malloc_containers, ref_containers);
        }
        return true;
    }

    // Find free block for a malloc container (returns nullptr if not found or constraints not met)
    structured_control_flow::Block* find_free_block(const std::string& malloc_container, analysis::Users& users_analysis) {
        auto users = users_analysis.uses(malloc_container);
        if (users.size() != 4) { // malloc (2 accesses) + ref (2 accesses) - but uses counts access nodes
            return nullptr;
        }

        for (auto& user : users) {
            if (auto* access_node = dynamic_cast<data_flow::AccessNode*>(user->element())) {
                auto& parent_graph = access_node->get_parent();
                auto parent_blk = dynamic_cast<structured_control_flow::Block*>(parent_graph.get_parent());
                if (parent_blk != nullptr && parent_blk->dataflow().is_a_library_node<stdlib::FreeNode>() != nullptr) {
                    return parent_blk;
                }
            }
        }
        return nullptr;
    }

    // Remove free block for a malloc container
    bool remove_free_block(structured_control_flow::Block* free_blk, analysis::ScopeAnalysis& scope_analysis) {
        if (free_blk == nullptr) {
            return false;
        }

        auto parent_scope = dynamic_cast<structured_control_flow::Sequence*>(scope_analysis.parent_scope(free_blk));
        if (parent_scope == nullptr) {
            return false;
        }

        int free_blk_index = parent_scope->index(*free_blk);
        builder_.remove_child(*parent_scope, free_blk_index);
        return true;
    }

public:
    LocalBufferReuseN(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        : visitor::NonStoppingStructuredSDFGVisitor(builder, analysis_manager) {}

    static std::string name() { return "LocalBufferReuseN"; }

    virtual bool accept(structured_control_flow::Sequence& node) override {
        bool applied = false;

        auto& users_analysis = this->analysis_manager_.get<analysis::Users>();
        auto& scope_analysis = this->analysis_manager_.get<analysis::ScopeAnalysis>();

        // Each pattern needs N * 3 blocks: (malloc, ref, lib) for each of N lib nodes
        constexpr int pattern_size = N * 3;

        int i = 0;
        while (i + pattern_size <= static_cast<int>(node.size())) {
            std::vector<std::string> malloc_containers;
            std::vector<std::string> ref_containers;

            // Get first malloc size for comparison
            auto* first_malloc_blk = get_block(node, i);
            if (!first_malloc_blk) {
                i++;
                continue;
            }
            auto* first_malloc = first_malloc_blk->template is_a_library_node<stdlib::MallocNode>();
            if (!first_malloc) {
                i++;
                continue;
            }

            // Verify all triples
            if (!verify_all_triples<0>(node, i, first_malloc->size(), "", malloc_containers, ref_containers)) {
                i++;
                continue;
            }

            // Pattern matched! Now verify all free block constraints before applying changes.
            // We keep the first malloc/ref and reuse it for subsequent ones.

            // Phase 1: Verify all constraints are met (find all free blocks)
            std::vector<structured_control_flow::Block*> free_blocks;
            bool all_constraints_met = true;
            for (int k = 1; k < static_cast<int>(N); k++) {
                auto* free_blk = find_free_block(malloc_containers[k], users_analysis);
                if (free_blk == nullptr) {
                    all_constraints_met = false;
                    break;
                }
                auto parent_scope =
                    dynamic_cast<structured_control_flow::Sequence*>(scope_analysis.parent_scope(free_blk));
                if (parent_scope == nullptr) {
                    all_constraints_met = false;
                    break;
                }
                free_blocks.push_back(free_blk);
            }

            if (!all_constraints_met) {
                i++;
                continue;
            }

            // Phase 2: Apply all changes (process from last to first to maintain correct indices)
            for (int k = N - 1; k >= 1; k--) {
                // Update reference block to use previous ref_container instead of this malloc_container
                int ref_idx = i + k * 3 + 1;
                auto* ref_blk = get_block(node, ref_idx);
                if (ref_blk) {
                    for (auto& n : ref_blk->dataflow().nodes()) {
                        auto* access = dynamic_cast<data_flow::AccessNode*>(&n);
                        if (access != nullptr && access->data() == malloc_containers[k]) {
                            access->data(ref_containers[k - 1]);
                        }
                    }
                }

                // Remove the malloc block (at index i + k * 3)
                int malloc_idx = i + k * 3;
                builder_.remove_child(node, malloc_idx);

                // Remove the free block for this malloc container
                remove_free_block(free_blocks[k - 1], scope_analysis);

                DEBUG_PRINTLN(
                    "Eliminated tensor with containers " << malloc_containers[0] << " and " << malloc_containers[k]
                );
            }

            applied = true;
            i++;
        }

        return applied;
    }
};

// Keep the original 2-lib-node version for backward compatibility
template<class T, class S>
using LocalBufferReuse = LocalBufferReuseN<T, S>;

// Type aliases for common patterns with 2 lib nodes
typedef VisitorPass<LocalBufferReuseN<math::tensor::ConvNode, math::tensor::BatchNormNode>> ConvBatchNormEliminationPass;
typedef VisitorPass<LocalBufferReuseN<math::tensor::BatchNormNode, math::tensor::ReLUNode>> BatchNormReLUEliminationPass;

// Type aliases for 3 lib nodes
typedef VisitorPass<LocalBufferReuseN<math::tensor::ConvNode, math::tensor::BatchNormNode, math::tensor::ReLUNode>>
    ConvBatchNormReLUEliminationPass;

// Type aliases for 4 lib nodes
typedef VisitorPass<
    LocalBufferReuseN<math::tensor::ConvNode, math::tensor::BatchNormNode, math::tensor::ReLUNode, math::tensor::ConvNode>>
    ConvBatchNormReLUConvEliminationPass;

// Type aliases for 5 lib nodes
typedef VisitorPass<LocalBufferReuseN<
    math::tensor::ConvNode,
    math::tensor::BatchNormNode,
    math::tensor::ReLUNode,
    math::tensor::ConvNode,
    math::tensor::BatchNormNode>>
    ConvBatchNormReLUConvBatchNormEliminationPass;

Pipeline local_buffer_reuse_pipeline() {
    Pipeline pipeline("LocalBufferReusePipeline");

    // Register passes from longest to shortest to maximize elimination
    // 5 lib nodes first
    pipeline.register_pass<ConvBatchNormReLUConvBatchNormEliminationPass>();

    // 4 lib nodes
    pipeline.register_pass<ConvBatchNormReLUConvEliminationPass>();

    // 3 lib nodes
    pipeline.register_pass<ConvBatchNormReLUEliminationPass>();

    // 2 lib nodes (original patterns)
    pipeline.register_pass<ConvBatchNormEliminationPass>();
    pipeline.register_pass<BatchNormReLUEliminationPass>();

    return pipeline;
}

} // namespace passes
} // namespace sdfg
