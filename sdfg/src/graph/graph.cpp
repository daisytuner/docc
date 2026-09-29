#include "sdfg/graph/graph.h"

namespace sdfg {
namespace graph {

const ReverseGraph reverse(const Graph& graph) {
    return boost::reverse_graph<Graph>(graph);
};

const std::tuple<
    std::unique_ptr<const UndirectedGraph>,
    const std::unordered_map<Vertex, Vertex>,
    const std::unordered_map<Vertex, Vertex>>
undirected(const Graph& graph) {
    auto undirected_graph = std::make_unique<UndirectedGraph>();
    std::unordered_map<Vertex, Vertex> mapping;
    std::unordered_map<Vertex, Vertex> reverse_mapping;

    boost::graph_traits<Graph>::vertex_iterator vi, vend;
    for (boost::tie(vi, vend) = boost::vertices(graph); vi != vend; ++vi) {
        auto new_vertex = boost::add_vertex(*undirected_graph);
        mapping.emplace(*vi, new_vertex);
        reverse_mapping.emplace(new_vertex, *vi);
    }

    auto [eb, ee] = boost::edges(graph);
    auto edges = std::ranges::subrange(eb, ee);
    for (const auto& edge : edges) {
        const auto& u = boost::source(edge, graph);
        const auto& v = boost::target(edge, graph);
        boost::add_edge(mapping.at(u), mapping.at(v), *undirected_graph);
    }

    return {std::move(undirected_graph), mapping, reverse_mapping};
};

const std::list<Vertex> depth_first_search(const Graph& graph, const graph::Vertex start) {
    std::list<Vertex> nodes;
    std::list<Edge> back_edges;
    DFSVisitor visitor(back_edges, nodes);

    std::unordered_map<graph::Vertex, boost::default_color_type> vertex_colors;
    boost::depth_first_search(graph, visitor, boost::make_assoc_property_map(vertex_colors), start);

    return nodes;
};

std::pair<int, std::unordered_map<Vertex, size_t>> strongly_connected_components(const Graph& graph) {
    IndexMap vertex_index_map;
    boost::associative_property_map<IndexMap> boost_index_map(vertex_index_map);
    boost::graph_traits<Graph>::vertex_iterator vi, vend;
    size_t i = 0;
    for (boost::tie(vi, vend) = boost::vertices(graph); vi != vend; ++vi, ++i) {
        boost::put(boost_index_map, *vi, i);
    }

    std::unordered_map<Vertex, size_t> component_map;
    boost::associative_property_map<IndexMap> boost_component_map(component_map);

    size_t num_ccs = boost::strong_components(graph, boost_component_map, boost::vertex_index_map(boost_index_map));

    return {num_ccs, component_map};
};

std::pair<size_t, const std::unordered_map<Vertex, size_t>> weakly_connected_components(const Graph& graph) {
    const auto& undirected_graph = undirected(graph);

    IndexMap vertex_index_map;
    boost::associative_property_map<IndexMap> boost_index_map(vertex_index_map);
    boost::graph_traits<UndirectedGraph>::vertex_iterator vi, vend;
    size_t i = 0;
    for (boost::tie(vi, vend) = boost::vertices(*std::get<0>(undirected_graph)); vi != vend; ++vi, ++i) {
        boost::put(boost_index_map, *vi, i);
    }

    std::unordered_map<Vertex, size_t> undirected_component_map;
    boost::associative_property_map<std::unordered_map<Vertex, size_t>> boost_component_map(undirected_component_map);

    size_t num_ccs = boost::connected_components(
        *std::get<0>(undirected_graph), boost_component_map, boost::vertex_index_map(boost_index_map)
    );

    std::unordered_map<Vertex, size_t> component_map;
    for (const auto& entry : undirected_component_map) {
        component_map.emplace(std::get<2>(undirected_graph).at(entry.first), entry.second);
    }

    return {num_ccs, component_map};
};

const std::unordered_map<Vertex, Vertex> dominator_tree(const Graph& graph, const Vertex src) {
    std::unordered_map<Vertex, Vertex> dom_tree;

    IndexMap vertex_index_map;
    boost::associative_property_map<IndexMap> boost_index_map(vertex_index_map);
    boost::graph_traits<Graph>::vertex_iterator vi, vend;
    size_t i = 0;
    for (boost::tie(vi, vend) = boost::vertices(graph); vi != vend; ++vi, ++i) {
        boost::put(boost_index_map, *vi, i);
    }

    std::vector<size_t> df_num(num_vertices(graph), 0);
    auto df_num_map(boost::make_iterator_property_map(df_num.begin(), boost_index_map));

    std::vector<Vertex> parent(num_vertices(graph), boost::graph_traits<Graph>::null_vertex());
    auto parent_map(boost::make_iterator_property_map(parent.begin(), boost_index_map));

    std::vector<Vertex> dom_tree_vec(num_vertices(graph), boost::graph_traits<Graph>::null_vertex());
    auto dom_tree_pred_map(boost::make_iterator_property_map(dom_tree_vec.begin(), boost_index_map));

    std::vector<Vertex> vertices_by_df_num(parent);

    boost::lengauer_tarjan_dominator_tree(
        graph, src, boost_index_map, df_num_map, parent_map, vertices_by_df_num, dom_tree_pred_map
    );

    for (const auto& entry : vertex_index_map) {
        dom_tree.insert({entry.first, dom_tree_vec[entry.second]});
    }

    return dom_tree;
};

const std::unordered_map<Vertex, Vertex> post_dominator_tree(Graph& graph) {
    // Determine terminal vertices
    std::unordered_set<Vertex> terminal_vertices;
    for (auto [vb, ve] = boost::vertices(graph); vb != ve; ++vb) {
        if (boost::out_degree(*vb, graph) == 0) {
            terminal_vertices.insert(*vb);
        }
    }

    // add synthetic super-terminal if needed
    bool modified = false;
    graph::Vertex src;
    if (terminal_vertices.size() == 1) {
        src = *terminal_vertices.begin();
        modified = false;
    } else {
        src = boost::add_vertex(graph);
        for (const auto& v : terminal_vertices) {
            boost::add_edge(v, src, graph);
        }
        modified = true;
    }

    auto& rgraph = reverse(graph);

    std::unordered_map<Vertex, Vertex> pdom_tree;

    IndexMap vertex_index_map;
    boost::associative_property_map<IndexMap> boost_index_map(vertex_index_map);
    boost::graph_traits<Graph>::vertex_iterator vi, vend;
    size_t i = 0;
    for (boost::tie(vi, vend) = boost::vertices(rgraph); vi != vend; ++vi, ++i) {
        boost::put(boost_index_map, *vi, i);
    }

    std::vector<size_t> df_num(num_vertices(rgraph), 0);
    auto df_num_map(boost::make_iterator_property_map(df_num.begin(), boost_index_map));

    std::vector<Vertex> parent(num_vertices(rgraph), boost::graph_traits<Graph>::null_vertex());
    auto parent_map(boost::make_iterator_property_map(parent.begin(), boost_index_map));

    std::vector<Vertex> pdom_tree_vec(num_vertices(rgraph), boost::graph_traits<Graph>::null_vertex());
    auto pdom_tree_pred_map(boost::make_iterator_property_map(pdom_tree_vec.begin(), boost_index_map));

    std::vector<Vertex> vertices_by_df_num(parent);

    boost::lengauer_tarjan_dominator_tree(
        rgraph, src, boost_index_map, df_num_map, parent_map, vertices_by_df_num, pdom_tree_pred_map
    );

    for (const auto& entry : vertex_index_map) {
        pdom_tree.insert({entry.first, pdom_tree_vec[entry.second]});
    }

    // Remove synthetic super-terminal if added
    if (modified) {
        boost::clear_vertex(src, graph);
    }

    return pdom_tree;
};

const std::list<graph::Vertex> topological_sort(const Graph& graph) {
    std::unordered_map<graph::Vertex, boost::default_color_type> vertex_colors;
    std::list<graph::Vertex> order;
    boost::topological_sort(
        graph, std::back_inserter(order), boost::color_map(boost::make_assoc_property_map(vertex_colors))
    );
    order.reverse();
    return order;
};

bool is_acyclic(const Graph& graph) {
    try {
        topological_sort(graph);
        return true;
    } catch (boost::not_a_dag e) {
        return false;
    }
};

const std::list<Edge> back_edges(const Graph& graph, const graph::Vertex start) {
    std::list<Vertex> nodes;
    std::list<Edge> back_edges;
    DFSVisitor visitor(back_edges, nodes);

    std::unordered_map<graph::Vertex, boost::default_color_type> vertex_colors;
    boost::depth_first_search(graph, visitor, boost::make_assoc_property_map(vertex_colors), start);

    return back_edges;
};

void all_simple_paths_dfs(
    const Graph& graph,
    const Edge edge,
    const Vertex v,
    std::list<std::list<Edge>>& all_paths,
    std::list<Edge>& current_path,
    std::set<Vertex>& visited
) {
    const Vertex u = boost::target(edge, graph);
    if (visited.find(u) != visited.end()) {
        return;
    }

    current_path.push_back(edge);
    visited.insert(u);

    if (u == v) {
        all_paths.push_back(std::list<Edge>(current_path));

        current_path.pop_back();
        visited.erase(u);
        return;
    }

    auto [eb, ee] = boost::out_edges(u, graph);
    auto edges = std::ranges::subrange(eb, ee);
    for (auto next_edge : edges) {
        all_simple_paths_dfs(graph, next_edge, v, all_paths, current_path, visited);
    }

    current_path.pop_back();
    visited.erase(u);
};

const std::list<std::list<Edge>> all_simple_paths(const Graph& graph, const Vertex src, const Vertex dst) {
    std::list<std::list<Edge>> all_paths;

    std::set<Vertex> visited;
    std::list<Edge> current_path;

    auto [eb, ee] = boost::out_edges(src, graph);
    auto edges = std::ranges::subrange(eb, ee);
    for (auto edge : edges) {
        all_simple_paths_dfs(graph, edge, dst, all_paths, current_path, visited);
    }

    return all_paths;
};

} // namespace graph
} // namespace sdfg
