#include "sdfg/passes/scalar_promotion_pass.h"

#include <memory>
#include <unordered_set>
#include <vector>

#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/passes/dataflow/trivial_array_elimination.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/reduce.h"
#include "sdfg/tiles/library_nodes/tile_copy_node.h"
#include "sdfg/tiles/transformations/local_storage.h"
#include "sdfg/types/pointer.h"
#include "sdfg/visitor/for_each.h"

namespace sdfg::passes {

namespace {

/// The element of an array a loop accesses: an access node of it in the loop's body and the subset and base type
/// of a memlet through which the body accesses it.
struct Element {
    const data_flow::AccessNode* access = nullptr;
    data_flow::Subset subset;
    std::unique_ptr<types::IType> base_type;
};

Element element_of(structured_control_flow::StructuredLoop& loop, const std::string& container) {
    Element element;
    visitor::for_each_block(loop.root(), [&](structured_control_flow::Block& block) {
        auto& dataflow = block.dataflow();
        for (auto* access : dataflow.data_nodes()) {
            if (element.access || access->data() != container) {
                continue;
            }
            for (auto& memlet : dataflow.edges()) {
                if (&memlet.src() == access || &memlet.dst() == access) {
                    element.access = access;
                    element.subset = memlet.subset();
                    element.base_type = memlet.base_type().clone();
                    break;
                }
            }
        }
    });
    return element;
}

/// The tile copy @p node consists of, if it copies into or out of @p buffer.
const tiles::TileCopyNode* copy_of(structured_control_flow::ControlFlowNode& node, const std::string& buffer) {
    auto block = dynamic_cast<structured_control_flow::Block*>(&node);
    if (!block || block->dataflow().library_nodes().size() != 1) {
        return nullptr;
    }
    auto copy = dynamic_cast<const tiles::TileCopyNode*>(*block->dataflow().library_nodes().begin());
    if (!copy) {
        return nullptr;
    }
    for (auto& memlet : block->dataflow().in_edges(*copy)) {
        if (static_cast<const data_flow::AccessNode&>(memlet.src()).data() == buffer) {
            return copy;
        }
    }
    return nullptr;
}

/// Replaces each block that copies @p element of @p global into or out of the one-element @p buffer by an
/// assignment between the two.
void copies_to_assignments(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& parent,
    const std::string& global,
    const Element& element,
    const std::string& buffer
) {
    auto& buffer_type = builder.subject().type(buffer);
    for (size_t i = 0; i < parent.size(); ++i) {
        auto copy = copy_of(parent.at(i), buffer);
        if (!copy) {
            continue;
        }
        bool in = copy->direction() == tiles::CopyDirection::In;
        auto& block = builder.add_block_before(parent, parent.at(i), parent.at(i).debug_info());
        auto& source = builder.add_access(block, in ? global : buffer);
        auto& target = builder.add_access(block, in ? buffer : global);
        auto& assign = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        data_flow::Subset local{symbolic::zero()};
        builder.add_computational_memlet(
            block, source, assign, "_in", in ? element.subset : local, in ? *element.base_type : buffer_type
        );
        builder.add_computational_memlet(
            block, assign, "_out", target, in ? local : element.subset, in ? buffer_type : *element.base_type
        );
        builder.remove_child(parent, i + 1);
    }
}

} // namespace

bool ScalarPromotionPass::run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& sdfg = builder.subject();
    std::vector<structured_control_flow::Reduce*> reductions;
    for (auto* loop : analysis_manager.get<analysis::LoopAnalysis>().loops_in_pre_order()) {
        if (auto reduce = dynamic_cast<structured_control_flow::Reduce*>(loop)) {
            reductions.push_back(reduce);
        }
    }

    std::unordered_set<std::string> promoted;
    for (auto* reduce : reductions) {
        // LocalStorage retargets the reduction it localizes.
        auto infos = reduce->reductions();
        for (auto& reduction : infos) {
            auto& container = reduction.container;
            if (!dynamic_cast<const types::Pointer*>(&sdfg.type(container))) {
                continue;
            }
            auto element = element_of(*reduce, container);
            if (!element.access) {
                continue;
            }
            transformations::LocalStorage stage(*reduce, *element.access);
            // Only a register holding the single accumulated element becomes a scalar.
            if (!stage.can_be_applied(builder, analysis_manager) || !stage.tile_info().varying_dims().empty() ||
                !(stage.storage_type() == types::StorageType::CPU_Stack())) {
                continue;
            }
            stage.apply(builder, analysis_manager);
            auto& parent = static_cast<structured_control_flow::Sequence&>(*reduce->get_parent());
            copies_to_assignments(builder, parent, container, element, stage.local_container());
            promoted.insert(stage.local_container());
            analysis_manager.invalidate_all();
        }
    }
    if (promoted.empty()) {
        return false;
    }
    TrivialArrayElimination(promoted).run_pass(builder, analysis_manager);
    analysis_manager.invalidate_all();
    return true;
}

} // namespace sdfg::passes
