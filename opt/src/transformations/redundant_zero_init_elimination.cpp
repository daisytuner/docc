#include "sdfg/transformations/redundant_zero_init_elimination.h"

#include <cstdlib>
#include <optional>
#include <vector>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/targets/gpu/gpu_mma_fill_node.h"
#include "sdfg/targets/gpu/gpu_mma_fragment_load_node.h"
#include "sdfg/targets/gpu/gpu_mma_fragment_store_node.h"
#include "sdfg/targets/gpu/gpu_mma_matmul_node.h"

namespace sdfg {
namespace transformations {

namespace {

bool is_zero_literal(const std::string& literal) {
    std::string s = literal;
    while (!s.empty() && (s.back() == 'f' || s.back() == 'F' || s.back() == 'u' || s.back() == 'U' || s.back() == 'l' ||
                          s.back() == 'L')) {
        s.pop_back();
    }
    if (s.empty()) {
        return false;
    }
    char* end = nullptr;
    double value = std::strtod(s.c_str(), &end);
    return end == s.c_str() + s.size() && value == 0.0;
}

/// The container written by @p block if it is exactly `constant 0 -> assign -> container`.
std::optional<std::string> zero_store_container(Block& block) {
    auto& dataflow = block.dataflow();
    if (dataflow.nodes().size() != 3) {
        return std::nullopt;
    }
    auto tasklets = dataflow.tasklets();
    if (tasklets.size() != 1) {
        return std::nullopt;
    }
    auto* tasklet = *tasklets.begin();
    if (!tasklet->is_assign() || dataflow.in_degree(*tasklet) != 1 || dataflow.out_degree(*tasklet) != 1) {
        return std::nullopt;
    }
    auto* constant = dynamic_cast<const data_flow::ConstantNode*>(&(*dataflow.in_edges(*tasklet).begin()).src());
    if (constant == nullptr || !is_zero_literal(constant->data())) {
        return std::nullopt;
    }
    auto& dst = (*dataflow.out_edges(*tasklet).begin()).dst();
    auto* write = dynamic_cast<const data_flow::AccessNode*>(&dst);
    if (write == nullptr || dynamic_cast<const data_flow::ConstantNode*>(&dst) != nullptr ||
        dataflow.out_degree(*write) != 0) {
        return std::nullopt;
    }
    return write->data();
}

/// The container zero-initialized by @p loop if it is a perfect loop chain around a single zero store.
std::optional<std::string> zero_init_container(StructuredLoop& loop) {
    StructuredLoop* current = &loop;
    while (true) {
        auto& body = current->root();
        if (body.size() != 1) {
            return std::nullopt;
        }
        auto& child = body.at(0);
        if (auto* nested = dynamic_cast<StructuredLoop*>(&child)) {
            current = nested;
            continue;
        }
        if (auto* block = dynamic_cast<Block*>(&child)) {
            return zero_store_container(*block);
        }
        return std::nullopt;
    }
}

/// True if @p block reads or writes @p container.
bool block_accesses_container(const Block& block, const std::string& container) {
    for (auto* access : block.dataflow().data_nodes()) {
        if (dynamic_cast<const data_flow::ConstantNode*>(access) == nullptr && access->data() == container) {
            return true;
        }
    }
    return false;
}

/// The MMA loop nest following @p loop in its parent sequence, skipping intervening transfer/alias blocks.
///
/// After scheduling, host-to-device copies of the matmul's other operands and buffer alloc/alias blocks may sit
/// between the zero-init nest and the MMA nest. These do not touch the zero-init container, so they are skipped.
/// Any intervening block that does access the container (which could observe or clobber the zeros), or any
/// non-block/non-loop node, stops the search.
StructuredLoop* next_sibling_loop(StructuredLoop& loop, const std::string& container) {
    auto* parent = dynamic_cast<Sequence*>(loop.get_parent());
    if (parent == nullptr) {
        return nullptr;
    }
    int index = parent->index(loop);
    if (index < 0) {
        return nullptr;
    }
    for (size_t i = static_cast<size_t>(index) + 1; i < parent->size(); i++) {
        auto& sibling = parent->at(i);
        if (auto* nest = dynamic_cast<StructuredLoop*>(&sibling)) {
            return nest;
        }
        auto* block = dynamic_cast<Block*>(&sibling);
        if (block == nullptr || block_accesses_container(*block, container)) {
            return nullptr;
        }
    }
    return nullptr;
}

std::optional<std::vector<MmaNestOutputUses::CFragmentLoad>>
analyze(StructuredLoop& loop, StructuredSDFG& sdfg, analysis::AnalysisManager& analysis_manager) {
    if (!structured_control_flow::ScheduleType_ZeroInit::is_set(loop.schedule_type())) {
        return std::nullopt;
    }
    auto container = zero_init_container(loop);
    if (!container) {
        return std::nullopt;
    }
    auto* mma_nest = next_sibling_loop(loop, *container);
    if (mma_nest == nullptr) {
        return std::nullopt;
    }
    MmaNestOutputUses uses(sdfg, analysis_manager, *mma_nest, *container);
    uses.visit();
    if (!uses.valid() || !uses.has_mma() || !uses.has_store()) {
        return std::nullopt;
    }
    return uses.c_loads();
}

} // namespace

MmaNestOutputUses::MmaNestOutputUses(
    StructuredSDFG& sdfg,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::StructuredLoop& nest,
    const std::string& container
)
    : ImmutableStructuredSDFGVisitor(sdfg, analysis_manager), nest_(nest), container_(container) {
}

bool MmaNestOutputUses::visit() {
    return this->visit_internal(nest_.root());
}

bool MmaNestOutputUses::only_maps_or_branches_enclose(const structured_control_flow::Block& block) const {
    // Only maps and branches may enclose the C-fragment load: a sequential loop could re-read partial results
    // stored to the container in an earlier iteration, whereas a branch (e.g. the StreamK last-arrival guard,
    // which runs at most once per output tile) and maps (distinct output tiles) cannot.
    for (auto* node = block.get_parent(); node != nullptr; node = node->get_parent()) {
        if (dynamic_cast<const Sequence*>(node) == nullptr &&
            dynamic_cast<const structured_control_flow::Map*>(node) == nullptr &&
            dynamic_cast<const structured_control_flow::IfElse*>(node) == nullptr) {
            return false;
        }
        if (node == &nest_) {
            return true;
        }
    }
    return false;
}

bool MmaNestOutputUses::accept(structured_control_flow::Block& block) {
    auto& dataflow = block.dataflow();
    for (auto* lib_node : dataflow.library_nodes()) {
        if (dynamic_cast<gpu::GpuMmaMatmulNode*>(lib_node)) {
            has_mma_ = true;
        }
    }
    for (auto* access : dataflow.data_nodes()) {
        if (dynamic_cast<data_flow::ConstantNode*>(access) || access->data() != container_) {
            continue;
        }
        if (dataflow.in_degree(*access) != 0) {
            valid_ = false;
            return true;
        }
        for (auto& edge : dataflow.out_edges(*access)) {
            auto& dst = edge.dst();
            if (auto* store = dynamic_cast<gpu::GpuMmaFragmentStoreNode*>(&dst)) {
                if (edge.dst_conn() == store->input(gpu::GpuMmaFragmentStoreNode::PTR_INPUT_IDX)) {
                    has_store_ = true;
                    continue;
                }
            } else if (auto* load = dynamic_cast<gpu::GpuMmaFragmentLoadNode*>(&dst)) {
                if (load->fragment_type() == gpu::MmaFragmentType::C &&
                    edge.dst_conn() == load->input(gpu::GpuMmaFragmentLoadNode::PTR_INPUT_IDX) &&
                    only_maps_or_branches_enclose(block)) {
                    c_loads_.push_back({&block, load});
                    continue;
                }
            }
            valid_ = false;
            return true;
        }
    }
    return false;
}

RedundantZeroInitElimination::RedundantZeroInitElimination(structured_control_flow::StructuredLoop& loop)
    : loop_(loop) {
}

std::string RedundantZeroInitElimination::name() const {
    return "RedundantZeroInitElimination";
}

bool RedundantZeroInitElimination::
    can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    return analyze(loop_, builder.subject(), analysis_manager).has_value();
}

void RedundantZeroInitElimination::
    apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto c_loads = analyze(loop_, builder.subject(), analysis_manager);
    if (!c_loads) {
        throw InvalidTransformationException("RedundantZeroInitElimination: cannot be applied");
    }

    // The C fragments would only load the zeros of the init: fill them with zeros instead.
    for (auto& c_load : *c_loads) {
        auto& block = *c_load.block;
        auto& load = *c_load.node;
        auto& dataflow = block.dataflow();

        auto* frag_edge = dataflow.in_edge_for_connector(load, load.input(gpu::GpuMmaFragmentLoadNode::FRAG_INPUT_IDX));
        auto* ptr_edge = dataflow.in_edge_for_connector(load, load.input(gpu::GpuMmaFragmentLoadNode::PTR_INPUT_IDX));
        auto& frag_access = static_cast<data_flow::AccessNode&>(frag_edge->src());
        auto& ptr_access = static_cast<data_flow::AccessNode&>(ptr_edge->src());
        auto frag_type = frag_edge->base_type().clone();
        auto frag_debug_info = frag_edge->debug_info();

        auto& fill = builder.add_library_node<gpu::GpuMmaFillNode>(
            block, load.debug_info(), load.block_size(), load.element_type(), load.implementation_type()
        );
        builder.add_computational_memlet(
            block, frag_access, fill, fill.input(gpu::GpuMmaFillNode::Y_INPUT_IDX), {}, *frag_type, frag_debug_info
        );

        builder.remove_memlet(block, *frag_edge);
        builder.remove_memlet(block, *ptr_edge);
        builder.remove_node(block, load);
        if (dataflow.in_degree(ptr_access) == 0 && dataflow.out_degree(ptr_access) == 0) {
            builder.remove_node(block, ptr_access);
        }
    }

    auto& parent = static_cast<structured_control_flow::Sequence&>(*loop_.get_parent());
    builder.remove_child(parent, parent.index(loop_));

    analysis_manager.invalidate_all();
}

void RedundantZeroInitElimination::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["parameters"] = nlohmann::json::object();

    serializer::JSONSerializer ser_flat(false);
    j["subgraph"] = nlohmann::json::object();
    j["subgraph"]["0"] = nlohmann::json::object();
    ser_flat.serialize_node(j["subgraph"]["0"], loop_);
}

RedundantZeroInitElimination RedundantZeroInitElimination::
    from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j) {
    auto loop_id = j.at("subgraph").at("0").at("element_id").get<size_t>();
    auto* element = builder.find_element_by_id(loop_id);
    if (element == nullptr) {
        throw InvalidTransformationDescriptionException("Element with ID " + std::to_string(loop_id) + " not found.");
    }
    auto* loop = dynamic_cast<structured_control_flow::StructuredLoop*>(element);
    if (loop == nullptr) {
        throw InvalidTransformationDescriptionException(
            "Element with ID " + std::to_string(loop_id) + " is not a StructuredLoop."
        );
    }
    return RedundantZeroInitElimination(*loop);
}

} // namespace transformations
} // namespace sdfg
