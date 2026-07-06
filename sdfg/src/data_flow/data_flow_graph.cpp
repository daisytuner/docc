#include "sdfg/data_flow/data_flow_graph.h"

#include <algorithm>
#include <boost/graph/exception.hpp>
#include <cstddef>
#include <list>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/code_node.h"
#include "sdfg/data_flow/data_flow_node.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/graph/graph.h"

namespace sdfg {
namespace data_flow {

DataFlowGraph::DataFlowGraph(Element* parent) : parent_(parent) {}

void DataFlowGraph::validate(const Function& function) const {
    for (auto& node : this->nodes_) {
        node.second->validate(function);
        if (&node.second->get_parent() != this) {
            throw InvalidSDFGException("DataFlowGraph: Node parent mismatch.");
        }
    }
    for (auto& edge : this->edges_) {
        edge.second->validate(function);
    }
};

const Element* DataFlowGraph::get_parent() const { return this->parent_; };

Element* DataFlowGraph::get_parent() { return this->parent_; };

const data_flow::Memlet* DataFlowGraph::in_edge_for_connector(const data_flow::CodeNode& node, const std::string& conn)
    const {
    for (const auto& edge : this->in_edges(node)) {
        if (edge.dst_conn() == conn) {
            return &edge;
        }
    }
    return nullptr;
}

const data_flow::Memlet* DataFlowGraph::in_edge(const data_flow::AccessNode& node) const {
    auto edges = in_edges(node);
    auto it = edges.begin();
    if (it == edges.end()) {
        return nullptr;
    }
    const auto& edge = *it;
    if (++it != edges.end()) {
        throw InvalidSDFGException("Access node " + node.data() + " has multiple incoming edges.");
    }
    return &edge;
}

const data_flow::Memlet* DataFlowGraph::in_edge_if_single(const data_flow::AccessNode& node) const {
    auto edges = in_edges(node);
    auto it = edges.begin();
    if (it == edges.end()) {
        return nullptr;
    }
    const auto& edge = *it;
    if (++it != edges.end()) {
        return nullptr;
    }
    return &edge;
}

std::vector<data_flow::Memlet*> DataFlowGraph::in_edges_by_connector(const data_flow::CodeNode& node) {
    std::vector<data_flow::Memlet*> in_edges(node.inputs().size(), nullptr);
    for (auto& iedge : this->in_edges(node)) {
        for (size_t i = 0; i < node.inputs().size(); i++) {
            if (iedge.dst_conn() == node.input(i)) {
                in_edges[i] = &iedge;
                break;
            }
        }
    }
    return in_edges;
}

std::vector<const data_flow::Memlet*> DataFlowGraph::in_edges_by_connector(const data_flow::CodeNode& node) const {
    std::vector<const data_flow::Memlet*> in_edges(node.inputs().size(), nullptr);
    for (const auto& iedge : this->in_edges(node)) {
        for (size_t i = 0; i < node.inputs().size(); i++) {
            if (iedge.dst_conn() == node.input(i)) {
                in_edges[i] = &iedge;
                break;
            }
        }
    }
    return in_edges;
}

std::vector<data_flow::Memlet*> DataFlowGraph::out_edges_by_connector(const data_flow::CodeNode& node) {
    std::vector<data_flow::Memlet*> out_edges(node.outputs().size(), nullptr);
    for (auto& oedge : this->out_edges(node)) {
        for (size_t i = 0; i < node.outputs().size(); i++) {
            if (oedge.src_conn() == node.output(i)) {
                out_edges[i] = &oedge;
                break;
            }
        }
    }
    return out_edges;
}

std::vector<const data_flow::Memlet*> DataFlowGraph::out_edges_by_connector(const data_flow::CodeNode& node) const {
    std::vector<const data_flow::Memlet*> out_edges(node.outputs().size(), nullptr);
    for (const auto& oedge : this->out_edges(node)) {
        for (size_t i = 0; i < node.outputs().size(); i++) {
            if (oedge.src_conn() == node.output(i)) {
                out_edges[i] = &oedge;
                break;
            }
        }
    }
    return out_edges;
}

std::vector<const data_flow::Memlet*> DataFlowGraph::
    out_edges_for_connector(const data_flow::CodeNode& node, const std::string& conn) const {
    std::vector<const data_flow::Memlet*> outs;
    for (auto& edge : out_edges(node)) {
        if (edge.src_conn() == conn) {
            outs.push_back(&edge);
        }
    }
    return outs;
}

size_t DataFlowGraph::in_degree(const data_flow::DataFlowNode& node) const {
    return boost::in_degree(node.vertex(), this->graph_);
};

size_t DataFlowGraph::out_degree(const data_flow::DataFlowNode& node) const {
    return boost::out_degree(node.vertex(), this->graph_);
};

void DataFlowGraph::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    for (auto& node : this->nodes_) {
        node.second->replace(old_expression, new_expression);
    }

    for (auto& edge : this->edges_) {
        edge.second->replace(old_expression, new_expression);
    }
}

void DataFlowGraph::replace(const symbolic::ExpressionMapping& replacements) {
    for (auto& node : this->nodes_) {
        node.second->replace(replacements);
    }

    for (auto& edge : this->edges_) {
        edge.second->replace(replacements);
    }
}

/***** Section: Analysis *****/

std::unordered_set<const data_flow::Tasklet*> DataFlowGraph::tasklets() const {
    std::unordered_set<const data_flow::Tasklet*> ts;
    for (auto& node : this->nodes_) {
        if (auto tasklet = dynamic_cast<const data_flow::Tasklet*>(node.second.get())) {
            ts.insert(tasklet);
        }
    }

    return ts;
};

std::unordered_set<data_flow::Tasklet*> DataFlowGraph::tasklets() {
    std::unordered_set<data_flow::Tasklet*> ts;
    for (auto& node : this->nodes_) {
        if (auto tasklet = dynamic_cast<data_flow::Tasklet*>(node.second.get())) {
            ts.insert(tasklet);
        }
    }

    return ts;
};

std::unordered_set<const data_flow::LibraryNode*> DataFlowGraph::library_nodes() const {
    std::unordered_set<const data_flow::LibraryNode*> ls;
    for (auto& node : this->nodes_) {
        if (auto lib_node = dynamic_cast<const data_flow::LibraryNode*>(node.second.get())) {
            ls.insert(lib_node);
        }
    }

    return ls;
};

std::unordered_set<data_flow::LibraryNode*> DataFlowGraph::library_nodes() {
    std::unordered_set<data_flow::LibraryNode*> ls;
    for (auto& node : this->nodes_) {
        if (auto lib_node = dynamic_cast<data_flow::LibraryNode*>(node.second.get())) {
            ls.insert(lib_node);
        }
    }

    return ls;
};

std::unordered_set<const data_flow::AccessNode*> DataFlowGraph::data_nodes() const {
    std::unordered_set<const data_flow::AccessNode*> dnodes;
    for (auto& node : this->nodes_) {
        if (auto access_node = dynamic_cast<const data_flow::AccessNode*>(node.second.get())) {
            dnodes.insert(access_node);
        }
    }

    return dnodes;
};

std::unordered_set<data_flow::AccessNode*> DataFlowGraph::data_nodes() {
    std::unordered_set<data_flow::AccessNode*> dnodes;
    for (auto& node : this->nodes_) {
        if (auto access_node = dynamic_cast<data_flow::AccessNode*>(node.second.get())) {
            dnodes.insert(access_node);
        }
    }

    return dnodes;
};

std::unordered_set<const data_flow::AccessNode*> DataFlowGraph::reads() const {
    std::unordered_set<const data_flow::AccessNode*> rs;
    for (auto& node : this->nodes_) {
        if (auto access_node = dynamic_cast<const data_flow::AccessNode*>(node.second.get())) {
            if (this->out_degree(*access_node) > 0) {
                rs.insert(access_node);
            }
        }
    }

    return rs;
};

std::unordered_set<const data_flow::AccessNode*> DataFlowGraph::writes() const {
    std::unordered_set<const data_flow::AccessNode*> ws;
    for (auto& node : this->nodes_) {
        if (auto access_node = dynamic_cast<const data_flow::AccessNode*>(node.second.get())) {
            if (this->in_degree(*access_node) > 0) {
                ws.insert(access_node);
            }
        }
    }

    return ws;
};

std::unordered_set<const data_flow::DataFlowNode*> DataFlowGraph::sources() const {
    std::unordered_set<const data_flow::DataFlowNode*> ss;
    for (auto& node : this->nodes_) {
        if (this->in_degree(*node.second) == 0) {
            ss.insert(node.second.get());
        }
    }

    return ss;
};

std::unordered_set<data_flow::DataFlowNode*> DataFlowGraph::sources() {
    std::unordered_set<data_flow::DataFlowNode*> ss;
    for (auto& node : this->nodes_) {
        if (this->in_degree(*node.second) == 0) {
            ss.insert(node.second.get());
        }
    }

    return ss;
};

std::unordered_set<const data_flow::DataFlowNode*> DataFlowGraph::sinks() const {
    std::unordered_set<const data_flow::DataFlowNode*> ss;
    for (auto& node : this->nodes_) {
        if (this->out_degree(*node.second) == 0) {
            ss.insert(node.second.get());
        }
    }

    return ss;
};

std::unordered_set<data_flow::DataFlowNode*> DataFlowGraph::sinks() {
    std::unordered_set<data_flow::DataFlowNode*> ss;
    for (auto& node : this->nodes_) {
        if (this->out_degree(*node.second) == 0) {
            ss.insert(node.second.get());
        }
    }

    return ss;
};

std::unordered_set<const data_flow::DataFlowNode*> DataFlowGraph::predecessors(const data_flow::DataFlowNode& node
) const {
    std::unordered_set<const data_flow::DataFlowNode*> ss;
    for (auto& edge : this->in_edges(node)) {
        ss.insert(&edge.src());
    }

    return ss;
};

std::unordered_set<const data_flow::DataFlowNode*> DataFlowGraph::successors(const data_flow::DataFlowNode& node
) const {
    std::unordered_set<const data_flow::DataFlowNode*> ss;
    for (auto& edge : this->out_edges(node)) {
        ss.insert(&edge.dst());
    }

    return ss;
};

bool DataFlowGraph::is_valid_topological_order(const std::list<const DataFlowNode*>& order) const {
    std::unordered_map<const DataFlowNode*, size_t> pos;
    size_t idx = 0;
    for (const auto* node : order) {
        pos[node] = idx++;
    }
    for (const auto& edge : this->edges()) {
        auto src_it = pos.find(&edge.src());
        auto dst_it = pos.find(&edge.dst());
        if (src_it == pos.end() || dst_it == pos.end() || src_it->second >= dst_it->second) {
            return false;
        }
    }
    return true;
}

bool DataFlowGraph::is_valid_topological_order(const std::list<DataFlowNode*>& order) const {
    std::unordered_map<const DataFlowNode*, size_t> pos;
    size_t idx = 0;
    for (const auto* node : order) {
        pos[node] = idx++;
    }
    for (const auto& edge : this->edges()) {
        auto src_it = pos.find(&edge.src());
        auto dst_it = pos.find(&edge.dst());
        if (src_it == pos.end() || dst_it == pos.end() || src_it->second >= dst_it->second) {
            return false;
        }
    }
    return true;
}

std::list<DataFlowNode*> DataFlowGraph::boost_topological_sort() {
    auto order_vertices = graph::topological_sort(this->graph_);

    std::list<DataFlowNode*> order;
    for (const auto& v : order_vertices) {
        order.push_back(this->nodes_.at(v).get());
    }
    return order;
}

std::list<const DataFlowNode*> DataFlowGraph::boost_topological_sort() const {
    auto order_vertices = graph::topological_sort(this->graph_);

    std::list<const DataFlowNode*> order;
    for (const auto& v : order_vertices) {
        order.push_back(this->nodes_.at(v).get());
    }
    return order;
}

std::list<const DataFlowNode*> DataFlowGraph::semantic_topological_sort() const {
    auto [num_components, components_map] = graph::weakly_connected_components(this->graph_);

    // Build deterministic topological sort for each weakly connected component
    std::vector<std::list<const DataFlowNode*>> components(num_components);
    for (size_t i = 0; i < num_components; i++) {
        // Get all sinks of the current component
        std::vector<const DataFlowNode*> sinks;
        bool component_empty = true;
        for (auto [v, comp] : components_map) {
            if (comp == i) {
                component_empty = false;
                if (boost::out_degree(v, this->graph_) == 0) {
                    sinks.push_back(this->nodes_.at(v).get());
                }
            }
        }
        if (sinks.size() == 0) {
            if (component_empty) {
                continue;
            } else {
                throw boost::not_a_dag();
            }
        }

        // Create a queue with all sinks
        std::list<const DataFlowNode*> queue;
        queue.insert(queue.end(), sinks.begin(), sinks.end());

        // Perform a reversed DFS for each element in the queue
        std::unordered_map<const DataFlowNode*, std::list<const DataFlowNode*>> lists;
        std::unordered_map<const DataFlowNode*, std::list<std::pair<const DataFlowNode*, long long>>> dependencies;
        std::map<std::pair<const DataFlowNode*, long long>, const DataFlowNode*> backward_dependencies;
        std::unordered_set<const DataFlowNode*> visited;
        while (!queue.empty()) {
            const auto* start = queue.front();
            queue.pop_front();

            if (visited.contains(start)) {
                continue;
            }

            // Reversed DFS
            lists.insert({start, {}});
            dependencies.insert({start, {}});
            std::stack<std::pair<const DataFlowNode*, const DataFlowNode*>> stack({{start, nullptr}});
            while (!stack.empty()) {
                const auto* current = stack.top().first;
                const auto* successor = stack.top().second;
                stack.pop();

                // If multiple out edges, add to queue, add dependency, and skip
                if (current != start && this->out_degree(*current) > 1) {
                    queue.push_back(current);
                    long long dependency_id = -1;
                    if (const auto* code_node = dynamic_cast<const CodeNode*>(current)) {
                        for (const auto& oedge : this->out_edges(*current)) {
                            const auto* dst = &oedge.dst();
                            if (dst == successor) {
                                for (long long j = 0; j < code_node->outputs().size(); j++) {
                                    if (oedge.src_conn() == code_node->output(j)) {
                                        dependency_id = j;
                                        break;
                                    }
                                }
                                break;
                            }
                        }
                    } else {
                        std::vector<std::pair<const DataFlowNode*, size_t>> tmp_outputs;
                        std::unordered_set<const DataFlowNode*> local_visited;
                        for (const auto& oedge : this->out_edges(*current)) {
                            const auto* dst = &oedge.dst();
                            if (local_visited.contains(dst)) {
                                continue;
                            }
                            local_visited.insert(dst);
                            size_t value = 0;
                            if (const auto* tasklet = dynamic_cast<const Tasklet*>(dst)) {
                                value = tasklet->code();
                            } else if (const auto* libnode = dynamic_cast<const LibraryNode*>(dst)) {
                                value = 52;
                                for (char c : libnode->code().value()) {
                                    value += c;
                                }
                            }
                            tmp_outputs.push_back({dst, value});
                        }
                        std::sort(tmp_outputs.begin(), tmp_outputs.end(), [](const auto& a, const auto& b) {
                            return a.second > b.second ||
                                   (a.second == b.second && a.first->element_id() < b.first->element_id());
                        });
                        for (long long j = 0; j < tmp_outputs.size(); j++) {
                            if (tmp_outputs.at(j).first == successor) {
                                dependency_id = j;
                                break;
                            }
                        }
                    }
                    if (dependency_id == -1 || backward_dependencies.contains({current, dependency_id})) {
                        throw std::runtime_error("Could not create dependency in topological sort");
                    }
                    dependencies.at(start).push_front({current, dependency_id});
                    backward_dependencies.insert({{current, dependency_id}, start});
                    continue;
                }

                // Put the current element in the list
                if (visited.contains(current)) {
                    throw boost::not_a_dag();
                }
                visited.insert(current);
                lists.at(start).push_front(current);

                // Put all predecessors on the stack
                if (const auto* code_node = dynamic_cast<const CodeNode*>(current)) {
                    std::unordered_set<const DataFlowNode*> local_visited;
                    for (const auto& input : code_node->inputs()) {
                        const Memlet* iedge = nullptr;
                        for (const auto& in_edge : this->in_edges(*code_node)) {
                            if (in_edge.dst_conn() == input) {
                                iedge = &in_edge;
                                break;
                            }
                        }
                        if (!iedge) {
                            continue;
                        }
                        const auto* src = &iedge->src();
                        if (!local_visited.contains(src)) {
                            local_visited.insert(src);
                            stack.push({src, current});
                        }
                    }
                } else {
                    std::vector<std::pair<const DataFlowNode*, size_t>> tmp_inputs;
                    std::unordered_set<const DataFlowNode*> local_visited;
                    for (const auto& iedge : this->in_edges(*current)) {
                        const auto* src = &iedge.src();
                        if (local_visited.contains(src)) {
                            continue;
                        }
                        local_visited.insert(src);
                        size_t value = 0;
                        if (const auto* tasklet = dynamic_cast<const Tasklet*>(src)) {
                            value = tasklet->code();
                        } else if (const auto* libnode = dynamic_cast<const LibraryNode*>(src)) {
                            value = 52;
                            for (char c : libnode->code().value()) {
                                value += c;
                            }
                        }
                        tmp_inputs.push_back({src, value});
                    }
                    std::sort(tmp_inputs.begin(), tmp_inputs.end(), [](const auto& a, const auto& b) {
                        return a.second > b.second ||
                               (a.second == b.second && a.first->element_id() < b.first->element_id());
                    });
                    for (auto& tmp_input : tmp_inputs) {
                        stack.push({tmp_input.first, current});
                    }
                }
            }
        }

        // Sort sinks if necessary
        if (sinks.size() > 1) {
            std::sort(sinks.begin(), sinks.end(), [](const DataFlowNode* a, const DataFlowNode* b) {
                const auto* a_tasklet = dynamic_cast<const Tasklet*>(a);
                const auto* b_tasklet = dynamic_cast<const Tasklet*>(b);
                const auto* a_libnode = dynamic_cast<const LibraryNode*>(a);
                const auto* b_libnode = dynamic_cast<const LibraryNode*>(b);
                const auto* a_access_node = dynamic_cast<const AccessNode*>(a);
                const auto* b_access_node = dynamic_cast<const AccessNode*>(b);
                if (a_tasklet && b_tasklet) {
                    return a_tasklet->code() < b_tasklet->code() || (a_tasklet->code() == b_tasklet->code() &&
                                                                     a_tasklet->element_id() < b_tasklet->element_id());
                } else if (a_tasklet && b_libnode) {
                    return true;
                } else if (a_tasklet && b_access_node) {
                    return true;
                } else if (a_libnode && b_libnode) {
                    return a_libnode->code().value() < b_libnode->code().value() ||
                           (a_libnode->code().value() == b_libnode->code().value() &&
                            a_libnode->element_id() < b_libnode->element_id());
                } else if (a_libnode && b_access_node) {
                    return true;
                } else if (a_access_node && b_access_node) {
                    return a_access_node->data() < b_access_node->data() ||
                           (a_access_node->data() == b_access_node->data() &&
                            a_access_node->element_id() < b_access_node->element_id());
                } else {
                    return false;
                }
            });
        }

        // Stich together by resolving dependencies
        visited.clear();
        for (const auto* sink : sinks) {
            if (visited.contains(sink)) {
                continue;
            }

            std::stack<const DataFlowNode*> stack({sink});
            while (!stack.empty()) {
                const auto* current = stack.top();
                visited.insert(current);

                bool all_resolved = true;
                for (auto [node, dependency_id] : dependencies.at(current)) {
                    if (!visited.contains(node)) {
                        stack.push(node);
                        all_resolved = false;
                        break;
                    }

                    for (long long j = 0; j < dependency_id; j++) {
                        if (!backward_dependencies.contains({node, j})) {
                            continue;
                        }
                        const auto* node2 = backward_dependencies.at({node, j});
                        if (!visited.contains(node2)) {
                            stack.push(node2);
                            all_resolved = false;
                            break;
                        }
                    }
                    if (!all_resolved) {
                        break;
                    }
                }
                if (!all_resolved) {
                    continue;
                }

                components.at(i).insert(components.at(i).end(), lists.at(current).begin(), lists.at(current).end());
                stack.pop();
            }
        }
    }

    // Sort components
    std::sort(components.begin(), components.end(), [](const auto& a, const auto& b) {
        return a.size() > b.size() ||
               (a.size() == b.size() && a.size() > 0 && a.front()->element_id() < b.front()->element_id());
    });

    // Resulting data structure
    std::list<const DataFlowNode*> order;
    for (auto& component : components) {
        order.insert(order.end(), component.begin(), component.end());
    }

    return order;
}

std::list<DataFlowNode*> DataFlowGraph::semantic_topological_sort() {
    auto [num_components, components_map] = graph::weakly_connected_components(this->graph_);

    // Build deterministic topological sort for each weakly connected component
    std::vector<std::list<DataFlowNode*>> components(num_components);
    for (size_t i = 0; i < num_components; i++) {
        // Get all sinks of the current component
        std::vector<DataFlowNode*> sinks;
        bool component_empty = true;
        for (auto [v, comp] : components_map) {
            if (comp == i) {
                component_empty = false;
                if (boost::out_degree(v, this->graph_) == 0) {
                    sinks.push_back(this->nodes_.at(v).get());
                }
            }
        }
        if (sinks.size() == 0) {
            if (component_empty) {
                continue;
            } else {
                throw boost::not_a_dag();
            }
        }

        // Create a queue with all sinks
        std::list<DataFlowNode*> queue;
        queue.insert(queue.end(), sinks.begin(), sinks.end());

        // Perform a reversed DFS for each element in the queue
        std::unordered_map<DataFlowNode*, std::list<DataFlowNode*>> lists;
        std::unordered_map<DataFlowNode*, std::list<std::pair<DataFlowNode*, long long>>> dependencies;
        std::map<std::pair<DataFlowNode*, long long>, DataFlowNode*> backward_dependencies;
        std::unordered_set<DataFlowNode*> visited;
        while (!queue.empty()) {
            auto* start = queue.front();
            queue.pop_front();

            if (visited.contains(start)) {
                continue;
            }

            // Reversed DFS
            lists.insert({start, {}});
            dependencies.insert({start, {}});
            std::stack<std::pair<DataFlowNode*, DataFlowNode*>> stack({{start, nullptr}});
            while (!stack.empty()) {
                auto* current = stack.top().first;
                auto* successor = stack.top().second;
                stack.pop();

                // If multiple out edges, add to queue, add dependency, and skip
                if (current != start && this->out_degree(*current) > 1) {
                    queue.push_back(current);
                    long long dependency_id = -1;
                    if (auto* code_node = dynamic_cast<CodeNode*>(current)) {
                        for (auto& oedge : this->out_edges(*current)) {
                            auto* dst = &oedge.dst();
                            if (dst == successor) {
                                for (long long j = 0; j < code_node->outputs().size(); j++) {
                                    if (oedge.src_conn() == code_node->output(j)) {
                                        dependency_id = j;
                                        break;
                                    }
                                }
                                break;
                            }
                        }
                    } else {
                        std::vector<std::pair<DataFlowNode*, size_t>> tmp_outputs;
                        std::unordered_set<DataFlowNode*> local_visited;
                        for (auto& oedge : this->out_edges(*current)) {
                            auto* dst = &oedge.dst();
                            if (local_visited.contains(dst)) {
                                continue;
                            }
                            local_visited.insert(dst);
                            size_t value = 0;
                            if (auto* tasklet = dynamic_cast<Tasklet*>(dst)) {
                                value = tasklet->code();
                            } else if (auto* libnode = dynamic_cast<LibraryNode*>(dst)) {
                                value = 52;
                                for (char c : libnode->code().value()) {
                                    value += c;
                                }
                            }
                            tmp_outputs.push_back({dst, value});
                        }
                        std::sort(tmp_outputs.begin(), tmp_outputs.end(), [](const auto& a, const auto& b) {
                            return a.second > b.second ||
                                   (a.second == b.second && a.first->element_id() < b.first->element_id());
                        });
                        for (long long j = 0; j < tmp_outputs.size(); j++) {
                            if (tmp_outputs.at(j).first == successor) {
                                dependency_id = j;
                                break;
                            }
                        }
                    }
                    if (dependency_id == -1 || backward_dependencies.contains({current, dependency_id})) {
                        throw std::runtime_error("Could not create dependency in topological sort");
                    }
                    dependencies.at(start).push_front({current, dependency_id});
                    backward_dependencies.insert({{current, dependency_id}, start});
                    continue;
                }

                // Put the current element in the list
                if (visited.contains(current)) {
                    throw boost::not_a_dag();
                }
                visited.insert(current);
                lists.at(start).push_front(current);

                // Put all predecessors on the stack
                if (auto* code_node = dynamic_cast<CodeNode*>(current)) {
                    std::unordered_set<DataFlowNode*> local_visited;
                    for (auto& input : code_node->inputs()) {
                        Memlet* iedge = nullptr;
                        for (auto& in_edge : this->in_edges(*code_node)) {
                            if (in_edge.dst_conn() == input) {
                                iedge = &in_edge;
                                break;
                            }
                        }
                        if (!iedge) {
                            continue;
                        }
                        auto* src = &iedge->src();
                        if (!local_visited.contains(src)) {
                            local_visited.insert(src);
                            stack.push({src, current});
                        }
                    }
                } else {
                    std::vector<std::pair<DataFlowNode*, size_t>> tmp_inputs;
                    std::unordered_set<DataFlowNode*> local_visited;
                    for (auto& iedge : this->in_edges(*current)) {
                        auto* src = &iedge.src();
                        if (local_visited.contains(src)) {
                            continue;
                        }
                        local_visited.insert(src);
                        size_t value = 0;
                        if (auto* tasklet = dynamic_cast<Tasklet*>(src)) {
                            value = tasklet->code();
                        } else if (auto* libnode = dynamic_cast<LibraryNode*>(src)) {
                            value = 52;
                            for (char c : libnode->code().value()) {
                                value += c;
                            }
                        }
                        tmp_inputs.push_back({src, value});
                    }
                    std::sort(tmp_inputs.begin(), tmp_inputs.end(), [](const auto& a, const auto& b) {
                        return a.second > b.second ||
                               (a.second == b.second && a.first->element_id() < b.first->element_id());
                    });
                    for (auto& tmp_input : tmp_inputs) {
                        stack.push({tmp_input.first, current});
                    }
                }
            }
        }

        // Sort sinks if necessary
        if (sinks.size() > 1) {
            std::sort(sinks.begin(), sinks.end(), [](const DataFlowNode* a, const DataFlowNode* b) {
                const auto* a_tasklet = dynamic_cast<const Tasklet*>(a);
                const auto* b_tasklet = dynamic_cast<const Tasklet*>(b);
                const auto* a_libnode = dynamic_cast<const LibraryNode*>(a);
                const auto* b_libnode = dynamic_cast<const LibraryNode*>(b);
                const auto* a_access_node = dynamic_cast<const AccessNode*>(a);
                const auto* b_access_node = dynamic_cast<const AccessNode*>(b);
                if (a_tasklet && b_tasklet) {
                    return a_tasklet->code() < b_tasklet->code() || (a_tasklet->code() == b_tasklet->code() &&
                                                                     a_tasklet->element_id() < b_tasklet->element_id());
                } else if (a_tasklet && b_libnode) {
                    return true;
                } else if (a_tasklet && b_access_node) {
                    return true;
                } else if (a_libnode && b_libnode) {
                    return a_libnode->code().value() < b_libnode->code().value() ||
                           (a_libnode->code().value() == b_libnode->code().value() &&
                            a_libnode->element_id() < b_libnode->element_id());
                } else if (a_libnode && b_access_node) {
                    return true;
                } else if (a_access_node && b_access_node) {
                    return a_access_node->data() < b_access_node->data() ||
                           (a_access_node->data() == b_access_node->data() &&
                            a_access_node->element_id() < b_access_node->element_id());
                } else {
                    return false;
                }
            });
        }

        // Stich together by resolving dependencies
        visited.clear();
        for (auto* sink : sinks) {
            if (visited.contains(sink)) {
                continue;
            }

            std::stack<DataFlowNode*> stack({sink});
            while (!stack.empty()) {
                auto* current = stack.top();
                visited.insert(current);

                bool all_resolved = true;
                for (auto [node, dependency_id] : dependencies.at(current)) {
                    if (!visited.contains(node)) {
                        stack.push(node);
                        all_resolved = false;
                        break;
                    }

                    for (long long j = 0; j < dependency_id; j++) {
                        if (!backward_dependencies.contains({node, j})) {
                            continue;
                        }
                        auto* node2 = backward_dependencies.at({node, j});
                        if (!visited.contains(node2)) {
                            stack.push(node2);
                            all_resolved = false;
                            break;
                        }
                    }
                    if (!all_resolved) {
                        break;
                    }
                }
                if (!all_resolved) {
                    continue;
                }

                components.at(i).insert(components.at(i).end(), lists.at(current).begin(), lists.at(current).end());
                stack.pop();
            }
        }
    }

    // Sort components
    std::sort(components.begin(), components.end(), [](const auto& a, const auto& b) {
        return a.size() > b.size() ||
               (a.size() == b.size() && a.size() > 0 && a.front()->element_id() < b.front()->element_id());
    });

    // Resulting data structure
    std::list<DataFlowNode*> order;
    for (auto& component : components) {
        order.insert(order.end(), component.begin(), component.end());
    }

    return order;
}

std::list<const DataFlowNode*> DataFlowGraph::topological_sort() const {
    auto order = semantic_topological_sort();
    if (!is_valid_topological_order(order)) {
        order = boost_topological_sort();
    }
    return order;
}

std::list<DataFlowNode*> DataFlowGraph::topological_sort() {
    auto order = semantic_topological_sort();
    if (!is_valid_topological_order(order)) {
        order = boost_topological_sort();
    }
    return order;
}

std::unordered_map<std::string, const data_flow::AccessNode*> DataFlowGraph::dominators() const {
    std::unordered_map<std::string, const data_flow::AccessNode*> frontier;
    for (auto& node : this->topological_sort()) {
        if (auto access_node = dynamic_cast<const data_flow::AccessNode*>(node)) {
            if (frontier.find(access_node->data()) == frontier.end()) {
                frontier[access_node->data()] = access_node;
            }
        }
    }

    return frontier;
};

std::unordered_map<std::string, const data_flow::AccessNode*> DataFlowGraph::post_dominators() const {
    std::unordered_map<std::string, const data_flow::AccessNode*> frontier;
    for (auto& node : this->topological_sort()) {
        if (auto access_node = dynamic_cast<const data_flow::AccessNode*>(node)) {
            frontier[access_node->data()] = access_node;
        }
    }

    return frontier;
};

std::unordered_map<std::string, data_flow::AccessNode*> DataFlowGraph::post_dominators() {
    std::unordered_map<std::string, data_flow::AccessNode*> frontier;
    for (auto& node : this->topological_sort()) {
        if (auto access_node = dynamic_cast<data_flow::AccessNode*>(node)) {
            frontier[access_node->data()] = access_node;
        }
    }

    return frontier;
};

auto DataFlowGraph::all_simple_paths(const data_flow::DataFlowNode& src, const data_flow::DataFlowNode& dst) const {
    std::list<std::list<graph::Edge>> all_paths_raw = graph::all_simple_paths(this->graph_, src.vertex(), dst.vertex());

    std::list<std::list<std::reference_wrapper<data_flow::Memlet>>> all_paths;
    for (auto& path_raw : all_paths_raw) {
        std::list<std::reference_wrapper<data_flow::Memlet>> path;
        for (auto& edge : path_raw) {
            path.push_back(*this->edges_.at(edge));
        }
        all_paths.push_back(path);
    }

    return all_paths;
};

const std::pair<size_t, const std::unordered_map<const data_flow::DataFlowNode*, size_t>> DataFlowGraph::
    weakly_connected_components() const {
    auto ccs_vertex = graph::weakly_connected_components(this->graph_);

    std::unordered_map<const data_flow::DataFlowNode*, size_t> ccs;
    for (auto& entry : ccs_vertex.second) {
        ccs[this->nodes_.at(entry.first).get()] = entry.second;
    }

    return {ccs_vertex.first, ccs};
}

const AccessNode* DataFlowGraph::find_standalone_entry(const Memlet* input_edge) const {
    if (!input_edge) {
        return nullptr;
    }
    auto* access_node = dynamic_cast<const data_flow::AccessNode*>(&input_edge->src());
    if (access_node && in_degree(*access_node) == 0) {
        return access_node;
    } else {
        return nullptr;
    }
}

const AccessNode* DataFlowGraph::find_standalone_exit(const Memlet* output_edge) const {
    if (!output_edge) {
        return nullptr;
    }
    auto* access_node = dynamic_cast<const data_flow::AccessNode*>(&output_edge->dst());
    if (access_node && out_degree(*access_node) == 0) {
        return access_node;
    } else {
        return nullptr;
    }
}


} // namespace data_flow
} // namespace sdfg
