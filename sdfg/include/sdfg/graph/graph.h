#pragma once

#include <boost/bimap.hpp>
#include <boost/functional/hash.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/copy.hpp>
#include <boost/graph/depth_first_search.hpp>
#include <boost/graph/dominator_tree.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/reverse_graph.hpp>
#include <boost/graph/strong_components.hpp>
#include <boost/graph/topological_sort.hpp>
#include <list>
#include <ranges>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

namespace sdfg {
namespace graph {

typedef boost::adjacency_list<boost::listS, boost::listS, boost::bidirectionalS> Graph;
typedef boost::graph_traits<Graph>::vertex_descriptor Vertex;
typedef boost::graph_traits<Graph>::edge_descriptor Edge;

typedef boost::adjacency_list<boost::listS, boost::listS, boost::undirectedS> UndirectedGraph;
typedef boost::graph_traits<UndirectedGraph>::edge_descriptor UndirectedEdge;

typedef boost::reverse_graph<Graph> ReverseGraph;
typedef boost::graph_traits<ReverseGraph>::edge_descriptor ReverseEdge;

typedef std::unordered_map<Vertex, size_t> IndexMap;

struct DFSVisitor : boost::default_dfs_visitor {
    std::list<Vertex>& nodes_;
    std::list<Edge>& back_edges_;

    DFSVisitor(std::list<Edge>& back_edges, std::list<Vertex>& nodes) : nodes_(nodes), back_edges_(back_edges) {};

    void discover_vertex(Vertex v, const Graph& g) {
        nodes_.push_back(v);
    };

    void back_edge(Edge e, const Graph& g) {
        back_edges_.push_back(e);
    };
};

const ReverseGraph reverse(const Graph& graph);

const std::tuple<
    std::unique_ptr<const UndirectedGraph>,
    const std::unordered_map<Vertex, Vertex>,
    const std::unordered_map<Vertex, Vertex>>
undirected(const Graph& graph);

const std::list<Vertex> depth_first_search(const Graph& graph, const graph::Vertex start);

std::pair<int, std::unordered_map<Vertex, size_t>> strongly_connected_components(const Graph& graph);

std::pair<size_t, const std::unordered_map<Vertex, size_t>> weakly_connected_components(const Graph& graph);

const std::unordered_map<Vertex, Vertex> dominator_tree(const Graph& graph, const Vertex src);

const std::unordered_map<Vertex, Vertex> post_dominator_tree(Graph& graph);

const std::list<graph::Vertex> topological_sort(const Graph& graph);

bool is_acyclic(const Graph& graph);

const std::list<Edge> back_edges(const Graph& graph, const graph::Vertex start);

void all_simple_paths_dfs(
    const Graph& graph,
    const Edge edge,
    const Vertex v,
    std::list<std::list<Edge>>& all_paths,
    std::list<Edge>& current_path,
    std::set<Vertex>& visited
);

const std::list<std::list<Edge>> all_simple_paths(const Graph& graph, const Vertex src, const Vertex dst);

} // namespace graph
} // namespace sdfg
