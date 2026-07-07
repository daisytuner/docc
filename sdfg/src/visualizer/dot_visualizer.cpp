#include "sdfg/visualizer/dot_visualizer.h"

#include <cstddef>
#include <string>
#include <utility>

#include <regex>
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace visualizer {

static std::regex dotIdBadChars("[^a-zA-Z0-9_]+");

static std::string escapeDotId(size_t id, const std::string& prefix = "") { return prefix + std::to_string(id); }

static std::string escapeDotId(const std::string& id, const std::string& prefix = "") {
    return prefix + std::regex_replace(id, dotIdBadChars, "_");
}

void DotVisualizer::register_chain_elem(const Element& element, const std::string& node_id, const std::string& cluster_id) {
    auto& scope = seq_scope_stack_.back();
    scope.last2_chain_elem = scope.last_chain_elem;
    scope.last_chain_elem = SeqChainElem{&element, node_id, cluster_id};
}

void DotVisualizer::enter_scope() { seq_scope_stack_.emplace_back(); }

void DotVisualizer::exit_scope() { seq_scope_stack_.pop_back(); }

void DotVisualizer::visualizeSDFG(const SDFG& sdfg) {
    this->stream_.clear();
    this->stream_ << "digraph SDFG {\n";
    this->stream_.setIndent(4);
    this->stream_ << "graph [compound=true];" << std::endl << "node [style=filled,fillcolor=white];" << std::endl;

    // State identifier in DOT
    std::unordered_map<size_t, std::string> node_ids;

    // States as nodes
    for (auto& state : sdfg.states()) {
        auto id = escapeDotId(state.element_id(), "state_");
        this->stream_ << "subgraph cluster_" << id << " {" << std::endl;
        this->stream_.setIndent(this->stream_.indent() + 4);
        this->stream_ << "style=filled;fillcolor=white;color=black;label=\"State " << state.element_id() << "\";"
                      << std::endl;
        if (auto* return_state = dynamic_cast<const control_flow::ReturnState*>(&state)) {
            this->stream_ << id << " [shape=cds,label=\" return " << return_state->data() << " \"];" << std::endl;
        } else {
            this->stream_ << id << " [shape=point,style=invis;label=\"\"];" << std::endl;
            this->visualizeDataFlowGraph(id, state.dataflow());
        }
        this->stream_.setIndent(this->stream_.indent() - 4);
        this->stream_ << "}" << std::endl;
        node_ids.insert({state.element_id(), id});
    }

    // Edges
    for (auto& edge : sdfg.edges()) {
        auto& src_id = node_ids.at(edge.src().element_id());
        auto& dst_id = node_ids.at(edge.dst().element_id());
        this->stream_ << src_id << " -> " << dst_id << " [ltail=cluster_" << src_id << ",lhead=cluster_" << dst_id
                      << ",label=\"";

        // Condition
        bool print_condition = !symbolic::eq(edge.condition(), symbolic::__true__());
        if (print_condition) {
            this->stream_ << edge.condition()->__str__();
        }

        // Assignments
        if (!edge.assignments().empty()) {
            if (print_condition) {
                this->stream_ << ",\\n";
            }
            this->stream_ << "{";
            bool first = true;
            for (auto& [var, expr] : edge.assignments()) {
                if (!first) {
                    this->stream_ << "; ";
                }
                this->stream_ << var->get_name() << " = " << expr->__str__();
                first = false;
            }
            this->stream_ << "}";
        }
        this->stream_ << "\"];" << std::endl;
    }

    this->stream_.setIndent(0);
    this->stream_ << "}" << std::endl;
}

void DotVisualizer::visualizeStructuredSDFG(const StructuredSDFG& sdfg) {
    this->stream_.clear();
    this->stream_ << "digraph " << escapeDotId(sdfg.name()) << " {" << std::endl;
    this->stream_.setIndent(4);
    this->stream_ << "graph [compound=true];" << std::endl;
    this->stream_ << "subgraph cluster_" << escapeDotId(sdfg.name()) << " {" << std::endl;
    this->stream_.setIndent(8);
    this->stream_ << "node [style=filled,fillcolor=white];" << std::endl
                  << "style=filled;color=lightblue;label=\"\";" << std::endl;
    this->visualizeSequence(sdfg, sdfg.root());
    this->stream_.setIndent(4);
    this->stream_ << "}" << std::endl;
    this->stream_.setIndent(0);
    this->stream_ << "}" << std::endl;
}

void DotVisualizer::visualizeBlock(const StructuredSDFG& sdfg, const structured_control_flow::Block& block) {
    auto id = escapeDotId(block.element_id(), "block_");
    this->stream_ << "subgraph cluster_" << id << " {" << std::endl;
    this->stream_.setIndent(this->stream_.indent() + 4);
    this->stream_ << "style=filled;shape=box;fillcolor=white;color=black;label=\"";
    if (show_block_ids) {
        this->stream_ << "#" << block.element_id() << " ";
    }
    this->stream_ << "\";" << std::endl;
    this->visualizeDataFlowGraph(id, block.dataflow());
    this->stream_.setIndent(this->stream_.indent() - 4);
    this->stream_ << "}" << std::endl;
}

void DotVisualizer::visualizeSequence(const StructuredSDFG& sdfg, const structured_control_flow::Sequence& sequence) {
    if (sequence.size() == 0) {
        auto id = escapeDotId(sequence.element_id(), "empty_seq_");
        this->stream_ << "subgraph cluster_" << id << " {" << std::endl;
        this->stream_.setIndent(this->stream_.indent() + 4);
        this->stream_ << "style=filled;shape=box;fillcolor=white;color=black;label=\"";
        if (show_block_ids) {
            this->stream_ << "Seq #" << sequence.element_id() << " ";
        }
        this->stream_ << id << " [shape=point,style=invis,label=\"\"];" << std::endl;
        register_chain_elem(sequence, id, "cluster_" + id);
        this->stream_ << "\";" << std::endl;
        this->stream_.setIndent(this->stream_.indent() - 4);
        this->stream_ << "}" << std::endl;
        return;
    }

    auto& seq_scope = this->seq_scope_stack_.back();

    for (size_t i = 0; i < sequence.size(); ++i) {
        std::pair<const structured_control_flow::ControlFlowNode&, const structured_control_flow::Transition&> child =
            sequence.at(i);

        this->visualizeNode(sdfg, child.first);

        if (seq_scope.last_chain_elem && seq_scope.last2_chain_elem) {
            this->stream_ << seq_scope.last2_chain_elem->node_id << " -> " << seq_scope.last_chain_elem->node_id
                          << " [";
            auto last2_cluster = seq_scope.last2_chain_elem->cluster_id;
            if (!last2_cluster.empty()) {
                this->stream_ << "ltail=\"" << last2_cluster << "\",";
            }
            auto last_cluster = seq_scope.last_chain_elem->cluster_id;
            if (!last_cluster.empty()) {
                this->stream_ << "lhead=\"" << last_cluster << "\",";
            }
            this->stream_ << "minlen=3";

            // visualize assignments on edge
            if (i > 0 && !sequence.at(i - 1).second.empty()) {
                this->stream_ << ",label=\"{";
                bool comma_sep = false;
                for (auto& [sym, expr] : sequence.at(i - 1).second.assignments()) {
                    if (comma_sep) {
                        this->stream_ << ",";
                        comma_sep = true;
                    }
                    this->stream_ << sym->get_name() << " = " << expr->__str__();
                }
                this->stream_ << "}\"";
            }

            this->stream_ << "];" << std::endl;
        }
    }

    seq_scope.last2_chain_elem = std::nullopt;
}

void DotVisualizer::visualizeIfElse(const StructuredSDFG& sdfg, const structured_control_flow::IfElse& if_else) {
    auto id = escapeDotId(if_else.element_id(), "if_");
    this->stream_ << "subgraph cluster_" << id << " {" << std::endl;
    this->stream_.setIndent(this->stream_.indent() + 4);
    this->stream_ << "style=filled;shape=box;fillcolor=white;color=black;label=\"";
    if (show_block_ids) {
        this->stream_ << "#" << if_else.element_id() << " ";
    }
    this->stream_ << "if:\";" << std::endl << id << " [shape=point,style=invis,label=\"\"];" << std::endl;
    for (size_t i = 0; i < if_else.size(); ++i) {
        this->stream_ << "subgraph cluster_" << id << "_" << std::to_string(i) << " {" << std::endl;
        this->stream_.setIndent(this->stream_.indent() + 4);
        this->stream_ << "style=filled;shape=box;fillcolor=white;color=black;label=\""
                      << this->expression(if_else.at(i).second->__str__()) << "\";" << std::endl;
        enter_scope();
        this->visualizeSequence(sdfg, if_else.at(i).first);
        exit_scope();
        this->stream_.setIndent(this->stream_.indent() - 4);
        this->stream_ << "}" << std::endl;
    }
    this->stream_.setIndent(this->stream_.indent() - 4);
    this->stream_ << "}" << std::endl;
    register_chain_elem(if_else, id, "cluster_" + id);
}

void DotVisualizer::visualizeWhile(const StructuredSDFG& sdfg, const structured_control_flow::While& while_loop) {
    auto id = escapeDotId(while_loop.element_id(), "while_");
    this->stream_ << "subgraph cluster_" << id << " {" << std::endl;
    this->stream_.setIndent(this->stream_.indent() + 4);
    this->stream_ << "style=filled;shape=box;fillcolor=white;color=black;label=\"";
    if (show_block_ids) {
        this->stream_ << "#" << while_loop.element_id() << " ";
    }
    this->stream_ << "while:\";" << std::endl << id << " [shape=point,style=invis,label=\"\"];" << std::endl;
    enter_scope();
    this->visualizeSequence(sdfg, while_loop.root());
    exit_scope();
    this->stream_.setIndent(this->stream_.indent() - 4);
    this->stream_ << "}" << std::endl;
    register_chain_elem(while_loop, id, "cluster_" + id);
}

void DotVisualizer::visualizeFor(const StructuredSDFG& sdfg, const structured_control_flow::For& loop) {
    auto id = escapeDotId(loop.element_id(), "for_");
    this->stream_ << "subgraph cluster_" << id << " {" << std::endl;
    this->stream_.setIndent(this->stream_.indent() + 4);
    this->stream_ << "style=filled;shape=box;fillcolor=white;color=black;label=\"";
    if (show_block_ids) {
        this->stream_ << "#" << loop.element_id() << " ";
    }
    this->stream_ << "for: ";
    this->visualizeForBounds(loop.indvar(), loop.init(), loop.condition(), loop.update());
    this->stream_ << "\";" << std::endl << id << " [shape=point,style=invis,label=\"\"];" << std::endl;
    enter_scope();
    this->visualizeSequence(sdfg, loop.root());
    exit_scope();
    this->stream_.setIndent(this->stream_.indent() - 4);
    this->stream_ << "}" << std::endl;
    register_chain_elem(loop, id, "cluster_" + id);
}

void DotVisualizer::visualizeReturn(const StructuredSDFG& sdfg, const structured_control_flow::Return& return_node) {
    auto id = escapeDotId(return_node.element_id(), "return_");
    this->stream_ << id << " [shape=cds,label=\" return " << return_node.data() << "\"];" << std::endl;
    register_chain_elem(return_node, id, "");
}
void DotVisualizer::visualizeBreak(const StructuredSDFG& sdfg, const structured_control_flow::Break& break_node) {
    auto id = escapeDotId(break_node.element_id(), "break_");
    this->stream_ << id << " [shape=cds,label=\" break  \"];" << std::endl;
    register_chain_elem(break_node, id, "");
}

void DotVisualizer::visualizeContinue(const StructuredSDFG& sdfg, const structured_control_flow::Continue& continue_node) {
    auto id = escapeDotId(continue_node.element_id(), "cont_");
    this->stream_ << id << " [shape=cds,label=\" continue  \"];" << std::endl;
    register_chain_elem(continue_node, id, "");
}

void DotVisualizer::visualizeMap(const StructuredSDFG& sdfg, const structured_control_flow::Map& map_node) {
    auto id = escapeDotId(map_node.element_id(), "map_");
    this->stream_ << "subgraph cluster_" << id << " {" << std::endl;
    this->stream_.setIndent(this->stream_.indent() + 4);
    this->stream_ << "style=filled;shape=box;fillcolor=white;color=black;label=\"";
    if (show_block_ids) {
        this->stream_ << "#" << map_node.element_id() << " ";
    }
    this->stream_ << "map [" << map_node.schedule_type().value() << "]: ";
    this->visualizeForBounds(map_node.indvar(), map_node.init(), map_node.condition(), map_node.update());

    this->stream_ << "\";" << std::endl << id << " [shape=point,style=invis,label=\"\"];" << std::endl;
    enter_scope();
    this->visualizeSequence(sdfg, map_node.root());
    exit_scope();
    this->stream_.setIndent(this->stream_.indent() - 4);
    this->stream_ << "}" << std::endl;
    register_chain_elem(map_node, id, "cluster_" + id);
}

void DotVisualizer::visualizeReduce(const StructuredSDFG& sdfg, const structured_control_flow::Reduce& reduce_node) {
    auto id = escapeDotId(reduce_node.element_id(), "reduce_");
    this->stream_ << "subgraph cluster_" << id << " {" << std::endl;
    this->stream_.setIndent(this->stream_.indent() + 4);
    this->stream_ << "style=filled;shape=box;fillcolor=white;color=black;label=\"";
    if (show_block_ids) {
        this->stream_ << "#" << reduce_node.element_id() << " ";
    }
    this->stream_ << "reduce [" << reduce_node.schedule_type().value() << "]";
    bool comma_sep = false;
    this->stream_ << " {";
    for (auto& reduction : reduce_node.reductions()) {
        if (comma_sep) {
            this->stream_ << ", ";
        }
        comma_sep = true;
        this->stream_ << structured_control_flow::reduction_operation_to_string(reduction.operation) << ": "
                      << reduction.container;
    }
    this->stream_ << "}: ";
    this->visualizeForBounds(reduce_node.indvar(), reduce_node.init(), reduce_node.condition(), reduce_node.update());

    this->stream_ << "\";" << std::endl << id << " [shape=point,style=invis,label=\"\"];" << std::endl;
    enter_scope();
    this->visualizeSequence(sdfg, reduce_node.root());
    exit_scope();
    this->stream_.setIndent(this->stream_.indent() - 4);
    this->stream_ << "}" << std::endl;
    register_chain_elem(reduce_node, id, "cluster_" + id);
}

void DotVisualizer::visualizeDataFlowGraph(const std::string& id, const data_flow::DataFlowGraph& dfg) {
    auto cluster_id = "cluster_" + id;
    if (dfg.nodes().empty()) {
        this->stream_ << id << " [shape=point,style=invis,label=\"\"];" << std::endl;
        register_chain_elem(*dfg.get_parent(), id, cluster_id);
        return;
    }
    std::string chain_node_id;
    std::list<const data_flow::DataFlowNode*> nodes = dfg.topological_sort();
    for (const data_flow::DataFlowNode* node : nodes) {
        std::vector<std::string> in_connectors;
        bool is_access_node = false;
        bool node_will_show_literal_connectors = false;
        auto nodeId = escapeDotId(node->element_id(), "id");
        if (chain_node_id.empty()) {
            chain_node_id = nodeId;
        }
        if (const data_flow::Tasklet* tasklet = dynamic_cast<const data_flow::Tasklet*>(node)) {
            this->stream_ << nodeId << " [shape=octagon,label=\"" << tasklet->output() << " = ";
            this->visualizeTasklet(*tasklet);
            this->stream_ << "\"];" << std::endl;

            in_connectors = tasklet->inputs();
            node_will_show_literal_connectors = true;
        } else if (const data_flow::ConstantNode* constant_node = dynamic_cast<const data_flow::ConstantNode*>(node)) {
            this->stream_ << nodeId << " [";
            this->stream_ << "penwidth=3.0,";
            if (this->sdfg_.is_transient(constant_node->data())) this->stream_ << "style=\"dashed,filled\",";
            this->stream_ << "label=\"" << constant_node->data() << "\"];" << std::endl;
            is_access_node = true;
        } else if (const data_flow::AccessNode* access_node = dynamic_cast<const data_flow::AccessNode*>(node)) {
            this->stream_ << nodeId << " [";
            this->stream_ << "penwidth=3.0,";
            if (this->sdfg_.is_transient(access_node->data())) this->stream_ << "style=\"dashed,filled\",";
            this->stream_ << "label=\"" << access_node->data() << "\"];" << std::endl;
            is_access_node = true;
        } else if (const data_flow::LibraryNode* libnode = dynamic_cast<const data_flow::LibraryNode*>(node)) {
            this->stream_ << nodeId << " [shape=doubleoctagon,label=\"" << libnode->toStr() << "\"];" << std::endl;
            in_connectors = libnode->inputs();
        }

        std::unordered_set<std::string> unused_connectors(in_connectors.begin(), in_connectors.end());
        for (const data_flow::Memlet& iedge : dfg.in_edges(*node)) {
            auto& src = iedge.src();
            auto& dst_conn = iedge.dst_conn();
            bool nonexistent_conn = false;

            if (!is_access_node) {
                auto it = unused_connectors.find(dst_conn);
                if (it != unused_connectors.end()) {
                    unused_connectors.erase(it); // remove connector from in_connectors, so it is not used again
                } else {
                    nonexistent_conn = true;
                }
            }

            this->stream_ << escapeDotId(src.element_id(), "id") << " -> " << nodeId << " [label=\"   ";
            bool dstIsVoid = dst_conn == "void";
            bool dstIsRef = dst_conn == "ref";
            bool dstIsDeref = dst_conn == "deref";
            auto& src_conn = iedge.src_conn();
            bool srcIsVoid = src_conn == "void";
            bool srcIsDeref = src_conn == "deref";

            if (nonexistent_conn) {
                this->stream_ << "!!"; // this should not happen, but if it does, we can still visualize the memlet
            }

            if (dstIsVoid || dstIsRef || dstIsDeref) { // subset applies to dst
                auto& dstVar = dynamic_cast<data_flow::AccessNode const&>(iedge.dst()).data();
                bool subsetOnDst = false;
                if (srcIsDeref && dstIsVoid) { // Pure Store by Memlet definition (Dereference Memlet Store)
                    auto& subset = iedge.subset();
                    if (subset.size() == 1 && symbolic::eq(subset[0], symbolic::integer(0))) {
                        this->stream_ << "*" << dstVar; // store to pointer without further address calc
                    } else { // fallback, this should not be allowed to happen
                        this->stream_ << dstVar; // use access node name instead of connector-name
                        subsetOnDst = true;
                    }
                } else if (dstIsVoid) { // computational memlet / output from tasklet / memory store
                    this->stream_ << dstVar; // use access node name instead of connector-name
                    subsetOnDst = true;
                } else {
                    this->stream_ << dstVar; // use access node name instead of connector-name
                }
                if (subsetOnDst) {
                    this->visualizeSubset(iedge.subset(), &iedge.base_type());
                }
            } else { // dst is a tasklet/library node
                this->stream_ << dst_conn;
            }

            this->stream_ << " = ";

            if (srcIsVoid || srcIsDeref) { // subset applies to src, could be computational, reference or dereference
                                           // memlet
                auto& srcVar = dynamic_cast<data_flow::AccessNode const&>(src).data();
                bool subsetOnSrc = false;
                if (srcIsVoid && dstIsRef) { // reference memlet / address-of / get-element-ptr equivalent
                    this->stream_ << "&";
                    subsetOnSrc = true;
                } else if (srcIsVoid && dstIsDeref) { // Dereference memlet / load from address
                    this->stream_ << "*";
                    auto& subset = iedge.subset();
                    if (subset.size() != 1 && symbolic::eq(subset[0], symbolic::integer(0))) { // does not match memlet
                                                                                               // definition -> fallback
                        subsetOnSrc = true;
                    }
                } else if (srcIsVoid) {
                    subsetOnSrc = true;
                }
                this->stream_ << srcVar;
                if (subsetOnSrc) {
                    this->visualizeSubset(iedge.subset(), &iedge.base_type());
                }
            } else {
                this->stream_ << src_conn;
            }
            this->stream_ << "   \"];" << std::endl;
        }

        if (!node_will_show_literal_connectors) {
            for (uint64_t i = 0; i < in_connectors.size(); ++i) {
                auto& in_conn = in_connectors[i];
                auto it = unused_connectors.find(in_conn);
                if (it != unused_connectors.end()) {
                    auto literal_id = escapeDotId(node->element_id(), "id") + "_" + escapeDotId(i, "in");
                    this->stream_ << literal_id << " [style=\"invis\", label=\"\"];" << std::endl;
                    this->stream_ << literal_id << " -> " << nodeId << " [style=\"dotted\", label=\"" << i << ":"
                                  << in_conn << "\"]" << ";" << std::endl;
                }
            }
        }
    }
    register_chain_elem(*dfg.get_parent(), chain_node_id, cluster_id);
}

void DotVisualizer::writeToFile(const Function& sdfg, const std::filesystem::path& file) { writeToFile(sdfg, &file); }

void DotVisualizer::writeToFile(const Function& sdfg, const std::filesystem::path* file) {
    DotVisualizer viz(sdfg);
    viz.visualize();

    std::filesystem::path fileName = file ? *file : std::filesystem::path(sdfg.name() + ".dot");

    auto parent_path = fileName.parent_path();
    if (!parent_path.empty()) {
        std::filesystem::create_directories(fileName.parent_path());
    }

    std::ofstream dotOutput(fileName, std::ofstream::out);
    if (!dotOutput.is_open()) {
        std::cerr << "Could not open file " << fileName << " for writing DOT output." << std::endl;
    }

    dotOutput << viz.getStream().str();
    dotOutput.close();
}

} // namespace visualizer
} // namespace sdfg
