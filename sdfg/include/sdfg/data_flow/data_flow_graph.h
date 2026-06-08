#pragma once

#include <boost/graph/graphviz.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/data_flow_node.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/graph/graph.h"
#include "sdfg/helpers/helpers.h"
#include "sdfg/symbolic/symbolic.h"

using json = nlohmann::json;

namespace sdfg {

namespace builder {
class SDFGBuilder;
class StructuredSDFGBuilder;
} // namespace builder

namespace data_flow {

class DataFlowGraph {
    friend class sdfg::builder::SDFGBuilder;
    friend class sdfg::builder::StructuredSDFGBuilder;

private:
    // Remark: Exclusive resource
    graph::Graph graph_;
    std::unordered_map<graph::Vertex, std::unique_ptr<data_flow::DataFlowNode>, boost::hash<graph::Vertex>> nodes_;
    std::unordered_map<graph::Edge, std::unique_ptr<data_flow::Memlet>, boost::hash<graph::Edge>> edges_;

    Element* parent_;

public:
    DataFlowGraph() = default;
    ~DataFlowGraph() = default;

    DataFlowGraph(const DataFlowGraph& graph) = delete;
    DataFlowGraph& operator=(const DataFlowGraph&) = delete;

    void validate(const Function& function) const;

    const Element* get_parent() const;

    Element* get_parent();

    auto nodes() const {
        return std::views::values(this->nodes_) | std::views::transform(helpers::indirect<data_flow::DataFlowNode>) |
               std::views::transform(helpers::add_const<data_flow::DataFlowNode>);
    };

    auto nodes() {
        return std::views::values(this->nodes_) | std::views::transform(helpers::indirect<data_flow::DataFlowNode>);
    };

    auto edges() const {
        return std::views::values(this->edges_) | std::views::transform(helpers::indirect<data_flow::Memlet>) |
               std::views::transform(helpers::add_const<data_flow::Memlet>);
    };

    auto edges() {
        return std::views::values(this->edges_) | std::views::transform(helpers::indirect<data_flow::Memlet>);
    };

    auto in_edges(const data_flow::DataFlowNode& node) const {
        auto [eb, ee] = boost::in_edges(node.vertex(), this->graph_);
        auto edges = std::ranges::subrange(eb, ee);

        auto memlets = std::views::transform(
                           edges,
                           [&lookup_table = this->edges_](const graph::Edge& edge) -> data_flow::Memlet& {
                               return *(lookup_table.find(edge)->second);
                           }
                       ) |
                       std::views::transform(helpers::add_const<data_flow::Memlet>);

        return memlets;
    };

    auto in_edges(const data_flow::DataFlowNode& node) {
        auto [eb, ee] = boost::in_edges(node.vertex(), this->graph_);
        auto edges = std::ranges::subrange(eb, ee);

        auto memlets =
            std::views::transform(edges, [&lookup_table = this->edges_](const graph::Edge& edge) -> data_flow::Memlet& {
                return *(lookup_table.find(edge)->second);
            });

        return memlets;
    };

    const data_flow::Memlet* in_edge_for_connector(const data_flow::CodeNode& node, const std::string& conn) const;

    const data_flow::Memlet* in_edge(const data_flow::AccessNode& node) const;

    std::vector<data_flow::Memlet*> in_edges_by_connector(const data_flow::CodeNode& node);

    std::vector<const data_flow::Memlet*> in_edges_by_connector(const data_flow::CodeNode& node) const;

    auto out_edges(const data_flow::DataFlowNode& node) const {
        auto [eb, ee] = boost::out_edges(node.vertex(), this->graph_);
        auto edges = std::ranges::subrange(eb, ee);

        auto memlets = std::views::transform(
                           edges,
                           [&lookup_table = this->edges_](const graph::Edge& edge) -> data_flow::Memlet& {
                               return *(lookup_table.find(edge)->second);
                           }
                       ) |
                       std::views::transform(helpers::add_const<data_flow::Memlet>);

        return memlets;
    };

    auto out_edges(const data_flow::DataFlowNode& node) {
        auto [eb, ee] = boost::out_edges(node.vertex(), this->graph_);
        auto edges = std::ranges::subrange(eb, ee);

        auto memlets =
            std::views::transform(edges, [&lookup_table = this->edges_](const graph::Edge& edge) -> data_flow::Memlet& {
                return *(lookup_table.find(edge)->second);
            });

        return memlets;
    };

    std::vector<const data_flow::Memlet*> out_edges_for_connector(const data_flow::CodeNode& node, const std::string& conn)
        const;

    std::vector<data_flow::Memlet*> out_edges_by_connector(const data_flow::CodeNode& node);

    std::vector<const data_flow::Memlet*> out_edges_by_connector(const data_flow::CodeNode& node) const;

    size_t in_degree(const data_flow::DataFlowNode& node) const;

    size_t out_degree(const data_flow::DataFlowNode& node) const;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression);

    /***** Section: Analysis *****/

    std::unordered_set<const data_flow::Tasklet*> tasklets() const;

    std::unordered_set<data_flow::Tasklet*> tasklets();

    std::unordered_set<const data_flow::LibraryNode*> library_nodes() const;

    std::unordered_set<data_flow::LibraryNode*> library_nodes();

    std::unordered_set<const data_flow::AccessNode*> data_nodes() const;

    std::unordered_set<data_flow::AccessNode*> data_nodes();

    std::unordered_set<const data_flow::AccessNode*> reads() const;

    std::unordered_set<const data_flow::AccessNode*> writes() const;

    std::unordered_set<const data_flow::DataFlowNode*> sources() const;

    std::unordered_set<data_flow::DataFlowNode*> sources();

    std::unordered_set<const data_flow::DataFlowNode*> sinks() const;

    std::unordered_set<data_flow::DataFlowNode*> sinks();

    std::unordered_set<const data_flow::DataFlowNode*> predecessors(const data_flow::DataFlowNode& node) const;

    std::unordered_set<const data_flow::DataFlowNode*> successors(const data_flow::DataFlowNode& node) const;

    std::list<const data_flow::DataFlowNode*> topological_sort() const;

    std::list<data_flow::DataFlowNode*> topological_sort();

    std::unordered_map<std::string, const data_flow::AccessNode*> dominators() const;

    std::unordered_map<std::string, const data_flow::AccessNode*> post_dominators() const;

    std::unordered_map<std::string, data_flow::AccessNode*> post_dominators();

    auto all_simple_paths(const data_flow::DataFlowNode& src, const data_flow::DataFlowNode& dst) const;

    const std::pair<size_t, const std::unordered_map<const data_flow::DataFlowNode*, size_t>> weakly_connected_components()
        const;

    template<typename T>
    T* is_a_library_node() {
        if (this->library_nodes().size() != 1) {
            return nullptr;
        }
        if (!this->tasklets().empty()) {
            return nullptr;
        }
        data_flow::LibraryNode* lib_node = *this->library_nodes().begin();
        return dynamic_cast<T*>(lib_node);
    };

    /**
     * For the given input, identify a standalone input from which one could use to replace the following dataflow
     * For a Block that contains a libNode with 2 inputs, 1 output, the access nodes of the inputs would be those
     * standalone entry points. For expanding of the libNode one can reuse those accessNodes, if they are the
     * entrypoints into the dataflow
     *
     * This is to abstract away the expectation that all edges are memlets that always must either start or end in an
     * AccessNode in preparation of supporting pure data-edges between tasklets etc.
     *
     * @param input_edge the input edge for which we want to find a standalone cut point.
     * @return nullptr, if no viable standalone input could be found
     */
    const AccessNode* find_standalone_entry(const Memlet* input_edge) const;

    /**
     * Same as [find_standalone_entry] find an access node that is standalone, so that we can recreate the access node
     * in a different place without impacting correctness, as long as we repreduce the same accesses via it.
     *
     * @param output_edge the output edge for which we want to find a standalone cut point
     * @return nullptr, if no viable standalone output could be found
     */
    const AccessNode* find_standalone_exit(const Memlet* output_edge) const;
};

} // namespace data_flow
} // namespace sdfg
