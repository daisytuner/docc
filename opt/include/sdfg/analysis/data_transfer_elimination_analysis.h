#pragma once

#include "sdfg/analysis/base_user_visitor.h"
#include "sdfg/analysis/pointer_analyzers.h"
#include "sdfg/analysis/structured_data_flow_analysis.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/library_nodes/stdlib/malloc.h"
#include "sdfg/targets/offloading/data_offloading_node.h"

namespace sdfg::analysis {

class ForwardingEscapePolicy {
protected:

public:
    virtual void on_escape(const std::string& container, const ControlFlowNode* node, const Element* user) = 0;
    virtual void on_write_via(const std::string& container, const ControlFlowNode* node, const data_flow::Memlet* user) = 0;
    virtual void on_read_via(const std::string& container, const ControlFlowNode* node, const data_flow::Memlet* user) = 0;
};

struct OffloadHolder {
    offloading::DataOffloadingNode* offload_node;
    stdlib::MallocNode* malloc_node;
    const data_flow::AccessNode* host_data;
    const data_flow::Memlet* host_access;
    const data_flow::AccessNode* dev_data;

    OffloadHolder(
        offloading::DataOffloadingNode* offload_node,
        const data_flow::AccessNode* host_data,
        const data_flow::Memlet* host_access,
        const data_flow::AccessNode* dev_data
    )
        : offload_node(offload_node), malloc_node(nullptr), host_data(host_data), host_access(host_access),
          dev_data(dev_data) {}

    OffloadHolder(
        stdlib::MallocNode* malloc_node, const data_flow::AccessNode* host_data, const data_flow::Memlet* host_access
    )
        : offload_node(nullptr), malloc_node(malloc_node), host_data(host_data), host_access(host_access),
          dev_data(nullptr) {}

    void remove_host_side();
};

struct ExposedOffload {
    OffloadHolder* offload;
    int read_count = 0;
};

class DataTransferEliminationCandidateCollector {
protected:
    std::vector<std::pair<ExposedOffload, OffloadHolder&>> transfer_reuse_candidates_;
    std::vector<std::pair<ExposedOffload, OffloadHolder&>> empty_malloc_candidates_;

public:
    void found_transfer_reuse_pair(const ExposedOffload& src, OffloadHolder& dst) {
        transfer_reuse_candidates_.emplace_back(src, dst);
    }

    void found_empty_host_malloc(const ExposedOffload malloc, OffloadHolder& h2d_transfer) {
        empty_malloc_candidates_.emplace_back(malloc, h2d_transfer);
    }

    const std::vector<std::pair<ExposedOffload, OffloadHolder&>>& transfer_reuse_candidates() const {
        return transfer_reuse_candidates_;
    }
    const std::vector<std::pair<ExposedOffload, OffloadHolder&>>& empty_malloc_candidates() const {
        return empty_malloc_candidates_;
    }
};

struct OffloadState : public ElementIdMapDataFlowState<ExposedOffload, OffloadHolder> {
protected:
    /**
     * These containers are used in a way that makes us lose track of their contents (pointer-leaks & aliasing
     * situations) or accesses it on the host, such that data on device would become out of sync
     * Needs to not include the accesses of offloading nodes, as we want to handle them more intelligently
     */
    std::unordered_set<std::string> kills_containers_;
    /**
     * All H2D offload needs. They kill whatever they could alias with.
     * If its a direct match (the inverse of an open D2H node), then its a candidate for removal
     */
    std::unordered_map<ElementId, std::unique_ptr<OffloadHolder>> h2d_nodes_;
    std::unordered_map<ElementId, std::unique_ptr<OffloadHolder>> generated_;
    std::unordered_map<std::string, std::vector<const data_flow::Memlet*>> reads_;
    bool full_kill_ = false;
    DataTransferEliminationCandidateCollector& collector_;

    enum class KillingType { None, Basic, DeviceReuse, EmptyHostMalloc };

public:
    OffloadState(DataTransferEliminationCandidateCollector& collector);
    void apply_kills_and_changes(ExposedType& exposed) const override;

    void found_escape(const std::string& container);
    void found_full_barrier(ControlFlowNode& node);
    void found_offload_node(Block& block, offloading::DataOffloadingNode& offload);
    void found_ptr_read(const std::string& container, const data_flow::Memlet* memlet);
    void found_ptr_write(const std::string& container, const data_flow::Memlet* memlet);
    void found_malloc(Block& block, stdlib::MallocNode& malloc);

    void add_h2d_entry(const OffloadHolder& entry);

    std::pair<KillingType, OffloadHolder*> find_killing_entry_node(const ExposedOffload& in_flight) const;
};

class DataTransferEliminationAnalysis : public BaseUserVisitor,
                                        public ForwardStructuredDataFlowAnalysis<OffloadState>,
                                        PointerEscapeAnalyzer<ForwardingEscapePolicy>,
                                        PointerUsedAnalyzer<ForwardingEscapePolicy>,
                                        ForwardingEscapePolicy,
                                        public DataTransferEliminationCandidateCollector {
private:
    StructuredSDFG& sdfg_;
    AnalysisManager& ana_;

public:
    DataTransferEliminationAnalysis(StructuredSDFG& sdfg, AnalysisManager& ana)
        : sdfg_(sdfg), ana_(ana), PointerEscapeAnalyzer(sdfg, *this), PointerUsedAnalyzer(sdfg, *this) {}

    void handle_lib_node(Block& block, data_flow::LibraryNode& node) override;

    void handle_structured_loop_before_body(StructuredLoop& loop) override;

    void on_escape(const std::string& container, const ControlFlowNode* node, const Element* user) override;
    void on_write_via(const std::string& container, const ControlFlowNode* node, const data_flow::Memlet* user) override;
    void on_read_via(const std::string& container, const ControlFlowNode* node, const data_flow::Memlet* user) override;

    void use_as_return_src(const std::string& container, const Return& ret) override {
        PointerEscapeAnalyzer::use_as_return_src(container, ret);
        PointerUsedAnalyzer::use_as_return_src(container, ret);
    }
    void use_as_symbol_read(
        const std::string& container,
        const ControlFlowNode* node,
        const Element* user,
        SymbolReadLocation loc,
        int loc_index,
        symbolic::Expression expr
    ) override {
        PointerEscapeAnalyzer::use_as_symbol_read(container, node, user, loc, loc_index, expr);
        PointerUsedAnalyzer::use_as_symbol_read(container, node, user, loc, loc_index, expr);
    }
    void use_as_src_node(
        const std::string& container,
        const data_flow::AccessNode& node,
        const data_flow::Memlet& edge,
        const Block& block
    ) override {
        PointerEscapeAnalyzer::use_as_src_node(container, node, edge, block);
        PointerUsedAnalyzer::use_as_src_node(container, node, edge, block);
    }
    void use_as_dst_node(
        const std::string& container,
        const data_flow::AccessNode& node,
        const data_flow::Memlet& edge,
        const Block& block
    ) override {
        PointerEscapeAnalyzer::use_as_dst_node(container, node, edge, block);
        PointerUsedAnalyzer::use_as_dst_node(container, node, edge, block);
    }
    void use_as_symbol_write(
        const symbolic::Symbol& container, const ControlFlowNode* node, const Element* user, SymbolWriteLocation loc
    ) override {
        PointerEscapeAnalyzer::use_as_symbol_write(container, node, user, loc);
        PointerUsedAnalyzer::use_as_symbol_write(container, node, user, loc);
    }

    std::unique_ptr<OffloadState> create_initial_state(const structured_control_flow::ControlFlowNode& node) override;

    void run();
};

} // namespace sdfg::analysis
