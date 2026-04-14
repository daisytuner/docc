#include "sdfg/passes/dataflow/dead_data_elimination.h"

#include "sdfg/analysis/base_user_visitor.h"
#include "sdfg/analysis/data_dependency_analysis.h"
#include "sdfg/analysis/pointer_analyzers.h"
#include "sdfg/data_flow/library_nodes/stdlib/free.h"
#include "sdfg/data_flow/library_nodes/stdlib/malloc.h"
#include "sdfg/targets/offloading/data_offloading_node.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"
#include "sdfg/visualizer/dot_visualizer.h"

namespace sdfg {
namespace passes {

DeadDataElimination::DeadDataElimination() : Pass(), legacy_removals_(true) {};

DeadDataElimination::DeadDataElimination(bool legacy_removals) : Pass(), legacy_removals_(legacy_removals) {};

std::string DeadDataElimination::name() { return "DeadDataElimination"; };

/**
 * Simple escape policy that collects all escape and overwrite events
 * into a map of container -> {element -> type} entries.
 */
class BlockerListPolicy {
public:
    enum class BlockerType { Escape, Overwrite };

    using BlockerMap = std::unordered_map<std::string, std::unordered_map<const Element*, BlockerType>>;

protected:
    BlockerMap blockers_;

public:
    void on_escape(const std::string& container, const ControlFlowNode* node, const Element* user) {
        blockers_[container].emplace(user, BlockerType::Escape);
    }

    void on_overwrite(const std::string& container, const ControlFlowNode* node, const Element* user) {
        blockers_[container].emplace(user, BlockerType::Overwrite);
    }
};

/**
 * Finds memory areas (heap for now) that are wholly owned by the surrounding function. Owned memory can be removed if
 * its no longer used, writes to it can be ellided if the data is never read. This is not true for memory writes in
 * general, as you must prove no reference to that data ever escapes our control
 */
class MemoryOwnershipAnalysis : public analysis::BaseUserVisitor,
                                analysis::PointerEscapeAnalyzer<BlockerListPolicy>,
                                analysis::PointerOverwriteAnalyzer<BlockerListPolicy>,
                                BlockerListPolicy {
    struct FreeCluster {
        const Block* block;
        const data_flow::Memlet* in;
        const data_flow::Memlet* out;

        FreeCluster(const Block* b, const data_flow::Memlet* i, const data_flow::Memlet* o) : block(b), in(i), out(o) {}
    };

    struct OwnedArea {
        data_flow::Memlet* producer;
        structured_control_flow::Block* producer_block;
        symbolic::Expression allocation_size;
        bool non_ssa = false;
        std::vector<FreeCluster> free_clusters;

        OwnedArea(data_flow::Memlet* p, structured_control_flow::Block* pb, symbolic::Expression as, bool ns)
            : producer(p), producer_block(pb), allocation_size(std::move(as)), non_ssa(ns) {}

        void remove_from(builder::StructuredSDFGBuilder& builder) const;
    };

private:
    StructuredSDFG& sdfg_;
    // memory that is allocated by us and therefore 'owned' until it escapes
    std::unordered_map<std::string, OwnedArea> originally_owned_data_;
    std::unordered_set<std::string> fully_owned_; // never escaped

public:
    MemoryOwnershipAnalysis(StructuredSDFG& sdfg);

    void run(analysis::AnalysisManager& manager);

    bool visit(sdfg::structured_control_flow::Block& node) override;

    void use_as_symbol_write(
        const symbolic::Symbol& container, const ControlFlowNode* node, const Element* user, SymbolWriteLocation loc
    ) override {
        PointerEscapeAnalyzer::use_as_symbol_write(container, node, user, loc);
        PointerOverwriteAnalyzer::use_as_symbol_write(container, node, user, loc);
    }
    void use_as_symbol_read(
        const std::string& container,
        const ControlFlowNode* node,
        const Element* user,
        SymbolReadLocation loc,
        int loc_index,
        symbolic::Expression expr
    ) override {
        PointerEscapeAnalyzer::use_as_symbol_read(container, node, user, loc, loc_index, std::move(expr));
        PointerOverwriteAnalyzer::use_as_symbol_read(container, node, user, loc, loc_index, std::move(expr));
    }
    void use_as_src_node(
        const std::string& container,
        const data_flow::AccessNode& node,
        const data_flow::Memlet& edge,
        const Block& block
    ) override {
        PointerEscapeAnalyzer::use_as_src_node(container, node, edge, block);
        PointerOverwriteAnalyzer::use_as_src_node(container, node, edge, block);
    }
    void use_as_dst_node(
        const std::string& container,
        const data_flow::AccessNode& node,
        const data_flow::Memlet& edge,
        const Block& block
    ) override {
        PointerEscapeAnalyzer::use_as_dst_node(container, node, edge, block);
        PointerOverwriteAnalyzer::use_as_dst_node(container, node, edge, block);
    }
    void use_as_return_src(const std::string& container, const Return& ret) override {
        PointerEscapeAnalyzer::use_as_return_src(container, ret);
        PointerOverwriteAnalyzer::use_as_return_src(container, ret);
    }

    const std::unordered_set<std::string>& fully_owned_areas() const { return fully_owned_; }

    const OwnedArea& owned_area(const std::string& container) const { return originally_owned_data_.at(container); }

private:
    static bool excusedEscape(const Element* element, const OwnedArea& area);
    static bool excusedOverwrite(const Element* element, const OwnedArea& area);
};

void MemoryOwnershipAnalysis::OwnedArea::remove_from(builder::StructuredSDFGBuilder& builder) const {
    auto& malloc_write = this->producer->dst();
    auto& malloc_node = this->producer->src();
    builder.clear_code_node_legacy(*this->producer_block, dynamic_cast<const data_flow::CodeNode&>(malloc_node));
    // builder.clear_node(
    //     *this->producer_block, dynamic_cast<data_flow::AccessNode&>(malloc_write), {&malloc_write, &malloc_node}
    // );

    for (auto& free_cluster : this->free_clusters) {
        auto& memlet = *free_cluster.out;
        builder.clear_code_node_legacy(
            *const_cast<Block*>(free_cluster.block), dynamic_cast<const data_flow::CodeNode&>(memlet.src())
        );
        // builder.clear_node(*const_cast<Block*>(free_cluster.block), memlet.dst(), {&memlet.dst(), &memlet.src()});
    }
}

MemoryOwnershipAnalysis::MemoryOwnershipAnalysis(StructuredSDFG& sdfg)
    : sdfg_(sdfg), PointerEscapeAnalyzer(sdfg, *this), PointerOverwriteAnalyzer(sdfg, *this) {}

bool MemoryOwnershipAnalysis::excusedEscape(const Element* element, const OwnedArea& area) {
    // An escape is excused if it matches the input edge of one of the free_clusters.
    // Reading the pointer to pass it to free() is not a real escape.
    for (const auto& cluster : area.free_clusters) {
        if (element == cluster.in) {
            return true;
        }
    }
    return false;
}

bool MemoryOwnershipAnalysis::excusedOverwrite(const Element* element, const OwnedArea& area) {
    auto* memlet = dynamic_cast<const data_flow::Memlet*>(element);

    if (!memlet) {
        return false;
    }

    // The producer (malloc output edge) is an excused overwrite — it's the initial assignment.
    if (element == area.producer) {
        return true;
    }
    // DataOffloadNodes currently have a fake-output edge instead of a pointer input.
    // But they can only write to memory, never generate/overwrite the pointer
    if (auto* offload = dynamic_cast<const offloading::DataOffloadingNode*>(&memlet->src())) {
        if (offload->transfer_direction() != offloading::DataTransferDirection::NONE) {
            return true;
        }
    }

    // The output edge of a free cluster is an excused overwrite — free sets the pointer
    // to NULL (a fake overwrite that doesn't represent a meaningful reassignment).
    for (const auto& cluster : area.free_clusters) {
        if (element == cluster.out) {
            return true;
        }
    }
    return false;
}


void MemoryOwnershipAnalysis::run(analysis::AnalysisManager& manager) {
    dispatch(sdfg_.root());

    for (auto& [name, area] : originally_owned_data_) {
        if (!area.non_ssa && area.allocation_size != SymEngine::null) {
            auto it = blockers_.find(name);
            if (it != blockers_.end()) {
                bool killed = false;
                for (auto& [element, type] : it->second) {
                    if (type == BlockerType::Escape) {
                        if (!excusedEscape(element, area)) {
                            killed = true;
                            break;
                        }
                    } else if (type == BlockerType::Overwrite) {
                        if (!excusedOverwrite(element, area)) {
                            killed = true;
                            break;
                        }
                    }
                }
                if (!killed) {
                    fully_owned_.insert(name);
                }
            } else {
                fully_owned_.insert(name);
            }
        }
    }
}


bool MemoryOwnershipAnalysis::visit(sdfg::structured_control_flow::Block& node) {
    auto& dflow = node.dataflow();
    for (auto& library_node : dflow.library_nodes()) {
        if (library_node->code() == stdlib::LibraryNodeType_Malloc) {
            auto* malloc_node = dynamic_cast<const stdlib::MallocNode*>(library_node);
            auto& alloc_size = malloc_node->size();
            auto output_conn = malloc_node->output(0);
            auto oedges = dflow.out_edges(*malloc_node) |
                          std::views::filter([&](const auto& e) { return e.src_conn() == output_conn; });
            for (auto& oedge : oedges) {
                auto* access_node = dynamic_cast<data_flow::AccessNode*>(&oedge.dst());
                if (access_node && oedge.is_dst_write()) {
                    auto container = access_node->data();
                    auto it = originally_owned_data_.find(container);
                    if (it != originally_owned_data_.end()) {
                        auto& area = it->second;
                        area.non_ssa = true;
                        area.allocation_size = SymEngine::null;
                        area.producer_block = nullptr;
                        area.producer = nullptr;
                        // DEBUG_PRINTLN("Conflicting ownership of " << container);
                        continue;
                    }
                    originally_owned_data_.emplace(
                        std::piecewise_construct,
                        std::forward_as_tuple(container),
                        std::forward_as_tuple(&oedge, &node, alloc_size, false)
                    );
                }
            }
        } else if (library_node->code() == stdlib::LibraryNodeType_Free) {
            auto* free_node = dynamic_cast<const stdlib::FreeNode*>(library_node);
            auto input = dflow.in_edge_for_connector(*free_node, free_node->input(0));
            auto outputs = dflow.out_edges_for_connector(*free_node, free_node->output(0));
            if (input && outputs.size() == 1) {
                auto* in_access = dynamic_cast<const data_flow::AccessNode*>(&input->src());
                auto* out_access = dynamic_cast<const data_flow::AccessNode*>(&outputs.at(0)->dst());

                if (in_access && out_access && in_access->data() == out_access->data()) {
                    auto& container = in_access->data();
                    if (sdfg_.type(container).type_id() == types::TypeID::Pointer) {
                        auto area_it = originally_owned_data_.find(container);
                        if (area_it != originally_owned_data_.end()) { // we scan in execution order. Malloc needs to
                                                                       // have been found before
                            auto& area = area_it->second;
                            area.free_clusters.emplace_back(&node, input, outputs[0]);
                        }
                    }
                }
            }
        }
    }

    return BaseUserVisitor::visit(node);
}

/**
 * Does not care about other types of accesses.
 * Presumes, that the data cannot alias and there is only one SSA-like instance of backing data.
 * So that indirect reads and writes can be matched up with each other.
 * Any read of the pointer or getting of an address is considered an escape for which aliasing cannot be excluded,
 * in which case you must not rely on this analysis
 */
class IndirectMemoryAccessFinder : public analysis::BaseUserVisitor { // TODO update to use the PointerUsedAnalyzer and
                                                                      // a policy that filters the containstarg
private:
    std::unordered_map<std::string, std::unordered_set<const data_flow::Memlet*>> indirect_reads_;
    std::unordered_map<std::string, std::unordered_map<const data_flow::Memlet*, const Block*>> writes_to_remove_;
    const std::unordered_set<std::string>& target_containers_;

public:
    IndirectMemoryAccessFinder(const std::unordered_set<std::string>& target_containers);

    const std::unordered_map<std::string, std::unordered_set<const data_flow::Memlet*>>& indirect_reads() {
        return indirect_reads_;
    }
    const std::unordered_map<std::string, std::unordered_map<const data_flow::Memlet*, const Block*>>& writes_to_remove() {
        return writes_to_remove_;
    }

    void use_as_symbol_read(
        const std::string& container,
        const ControlFlowNode* node,
        const Element* user,
        SymbolReadLocation loc,
        int loc_index,
        symbolic::Expression expr
    ) override {}
    void use_as_symbol_write(
        const symbolic::Symbol& container, const ControlFlowNode* node, const Element* user, SymbolWriteLocation loc
    ) override {}
    void use_as_src_node(
        const std::string& container,
        const data_flow::AccessNode& node,
        const data_flow::Memlet& edge,
        const Block& block
    ) override;
    void use_as_dst_node(
        const std::string& container,
        const data_flow::AccessNode& node,
        const data_flow::Memlet& edge,
        const Block& block
    ) override;
    void use_as_return_src(const std::string& container, const Return& ret) override {}
};

IndirectMemoryAccessFinder::IndirectMemoryAccessFinder(const std::unordered_set<std::string>& target_containers)
    : target_containers_(target_containers) {}

void IndirectMemoryAccessFinder::use_as_src_node(
    const std::string& container, const data_flow::AccessNode& node, const data_flow::Memlet& edge, const Block& block
) {
    if (target_containers_.contains(container)) {
        if (edge.is_src_pointed_to_read()) {
            indirect_reads_[container].insert(&edge);
        }
        // Library nodes may get a pointer as input. But some of them we know enough about,
        // to know they are only borrowing the pointer for read access during their execution, not representing an
        // actual leak these we can instead cound as indirect readse
        if (edge.is_src_read()) {
            if (auto* libNode = dynamic_cast<const data_flow::LibraryNode*>(&edge.dst())) {
                auto conns = libNode->inputs();
                auto idx = std::find(conns.begin(), conns.end(), edge.dst_conn()) - conns.begin();
                auto access_type = libNode->pointer_access_type(idx);
                auto maybe_rd_only = std::get_if<data_flow::PointerReadOnly>(&access_type);
                if (maybe_rd_only && maybe_rd_only->no_ptr_escape()) {
                    indirect_reads_[container].insert(&edge);
                }
            }
        }
    }
}

void IndirectMemoryAccessFinder::use_as_dst_node(
    const std::string& container, const data_flow::AccessNode& node, const data_flow::Memlet& edge, const Block& block
) {
    if (target_containers_.contains(container)) {
        if (edge.is_dst_pointed_to_write()) {
            writes_to_remove_[container][&edge] = &block;
        }
        // hack to classify Offload nodes with D2H correctly. For historic reasons they use a direct output edge
        // to the host ptr, even though they will never write the pointer, but only write the memory the pointer points
        // to as that edge is destructive to many optimizations and scheduled to be removed, cuhere custom handleing per
        // node
        if (edge.is_dst_write()) {
            if (auto* offload = dynamic_cast<const offloading::DataOffloadingNode*>(&edge.src())) {
                if (offload->transfer_direction() != offloading::DataTransferDirection::NONE) {
                    writes_to_remove_[container][&edge] = &block;
                }
            }
        }
    }
}


bool DeadDataElimination::run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    bool applied = false;

    auto& sdfg = builder.subject();

    // Check for locally allocated memory that we "own" and understand (no reference to it escapes or can potentially
    // alias)
    MemoryOwnershipAnalysis ownership_analysis(sdfg);
    ownership_analysis.run(analysis_manager);

    std::unordered_set<std::string> dead_containers;

    auto& fully_owned_areas = ownership_analysis.fully_owned_areas();
    if (!fully_owned_areas.empty()) {
        // We found pointered accesses, where we could prove that the pointer is unique and SSA-like, such that we may
        // check accesses via the pointer as well
        IndirectMemoryAccessFinder remaining_indirects(fully_owned_areas);
        remaining_indirects.dispatch(sdfg.root());
        for (auto& owned_area_id : fully_owned_areas) {
            auto& all_reads = remaining_indirects.indirect_reads();
            auto cont_reads_it = all_reads.find(owned_area_id);
            if (cont_reads_it == all_reads.end() || cont_reads_it->second.empty()) {
                DEBUG_PRINTLN("Removing fully owned memory " << owned_area_id << ", never used!");
                auto writes = remaining_indirects.writes_to_remove();
                // [owned_area] is never read, no reference to it escapes our control. So any write of it is useless
                auto writes_it = writes.find(owned_area_id);

                bool all_removed = true;
                if (writes_it != writes.end()) {
                    auto& to_remove = writes_it->second;
                    for (auto& [edge_to_remove, w_block] : to_remove) {
                        auto& write_node = dynamic_cast<const data_flow::AccessNode&>(edge_to_remove->dst());
                        int removed =
                            builder.clear_node(*const_cast<structured_control_flow::Block*>(w_block), write_node);
                        if (removed == 0) {
                            all_removed = false;
                        } else {
                            applied = true;
                        }
                    }
                }
                // This is the malloc. We can remove this because we understand what malloc does. Otherwise the
                // sideeffect flag would stop us from removing a libNode
                if (all_removed) {
                    auto& area = ownership_analysis.owned_area(owned_area_id);
                    area.remove_from(builder);
                    applied = true;
                    dead_containers.insert(owned_area_id);
                }
            }
        }
    }

    if (legacy_removals_) {
        if (applied) { // if changes were made, any cached analysis will be out of date.
            analysis_manager.invalidate_all();
        }

        // slightly expensive, because for fully_owned_areas we already looked for uses. But classified differently and
        // did not look at, whether the entire container can be removed
        auto& users = analysis_manager.get<analysis::Users>();
        auto& data_dependency_analysis = analysis_manager.get<analysis::DataDependencyAnalysis>();

        for (auto& name : sdfg.containers()) {
            if (!sdfg.is_transient(name)) {
                continue;
            }
            if (users.num_views(name) > 0 || users.num_moves(name) > 0) {
                continue;
            }
            auto num_reads = users.num_reads(name);
            if (!num_reads && users.num_writes(name) == 0) { // no reference of [name] anywhere
                dead_containers.insert(name);
                applied = true;
                continue;
            }

            if (sdfg.type(name).type_id() == types::TypeID::Pointer) {
                continue;
                // use analysis does not return actual reads and writes for pointers. So if [name] is a pointer,
                // num reads/writes, does not actually mean no reads exist and any removal is problematic
                // more complex cases have been removed above already
            }

            bool completely_unused = !num_reads; // if there are reads left, we can never remove the container, but
                                                 // maybe
            // some writes
            auto raws = data_dependency_analysis.definitions(name);
            for (auto set : raws) {
                bool no_reads = false;
                if (set.second.size() == 0) {
                    no_reads = true;
                }
                if (data_dependency_analysis.is_undefined_user(*set.first)) {
                    continue;
                }

                if (no_reads) {
                    bool could_eliminate_write = false;
                    auto write = set.first;
                    if (auto transition = dynamic_cast<structured_control_flow::Transition*>(write->element())) {
                        transition->assignments().erase(symbolic::symbol(name));
                        applied = true;
                        could_eliminate_write = true;
                    } else if (auto access_node = dynamic_cast<data_flow::AccessNode*>(write->element())) {
                        auto& graph = access_node->get_parent();
                        auto& block = dynamic_cast<structured_control_flow::Block&>(*graph.get_parent());

                        if (builder.clear_node(block, *access_node)) {
                            applied = true;
                            could_eliminate_write = true;
                        }
                    }

                    completely_unused &= could_eliminate_write;
                }
            }

            if (completely_unused) { // no reads, and all remaining writes could be removed
                dead_containers.insert(name);
            }
        }
    }

    for (auto& name : dead_containers) {
        builder.remove_container(name);
    }

    return applied;
};

} // namespace passes
} // namespace sdfg
