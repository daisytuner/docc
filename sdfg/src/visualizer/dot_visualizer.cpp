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

namespace sdfg {
namespace visualizer {

static std::regex dotIdBadChars("[^a-zA-Z0-9_]+");

static std::string escapeDotId(size_t id, const std::string& prefix = "") { return prefix + std::to_string(id); }

static std::string escapeDotId(const std::string& id, const std::string& prefix = "") {
    return prefix + std::regex_replace(id, dotIdBadChars, "_");
}

void DotVisualizer::visualizeBlock(const StructuredSDFG& sdfg, const structured_control_flow::Block& block) {
    auto id = escapeDotId(block.element_id(), "block_");
    this->stream_ << "subgraph cluster_" << id << " {" << std::endl;
    this->stream_.setIndent(this->stream_.indent() + 4);
    this->stream_ << "style=filled;shape=box;fillcolor=white;color=black;label=\"\";" << std::endl;
    this->last_comp_name_cluster_ = "cluster_" + id;
    if (block.dataflow().nodes().empty()) {
        this->stream_ << id << " [shape=point,style=invis,label=\"\"];" << std::endl;
        this->stream_.setIndent(this->stream_.indent() - 4);
        this->stream_ << "}" << std::endl;
        this->last_comp_name_ = id;
        return;
    }
    this->last_comp_name_.clear();
    std::list<const data_flow::DataFlowNode*> nodes = block.dataflow().topological_sort();
    for (const data_flow::DataFlowNode* node : nodes) {
        std::vector<std::string> in_connectors;
        bool is_access_node = false;
        bool node_will_show_literal_connectors = false;
        auto nodeId = escapeDotId(node->element_id(), "n_");
        if (this->last_comp_name_.empty()) this->last_comp_name_ = nodeId;
        if (const data_flow::Tasklet* tasklet = dynamic_cast<const data_flow::Tasklet*>(node)) {
            this->stream_ << nodeId << " [shape=octagon,label=\"" << tasklet->output() << " = ";
            this->visualizeTasklet(*tasklet);
            this->stream_ << "\"];" << std::endl;

            in_connectors = tasklet->inputs();
            node_will_show_literal_connectors = true;
        } else if (const data_flow::ConstantNode* constant_node = dynamic_cast<const data_flow::ConstantNode*>(node)) {
            this->stream_ << nodeId << " [";
            this->stream_ << "penwidth=3.0,";
            if (sdfg.is_transient(constant_node->data())) this->stream_ << "style=\"dashed,filled\",";
            this->stream_ << "label=\"" << constant_node->data() << "\"];" << std::endl;
            is_access_node = true;
        } else if (const data_flow::AccessNode* access_node = dynamic_cast<const data_flow::AccessNode*>(node)) {
            this->stream_ << nodeId << " [";
            this->stream_ << "penwidth=3.0,";
            if (sdfg.is_transient(access_node->data())) this->stream_ << "style=\"dashed,filled\",";
            this->stream_ << "label=\"" << access_node->data() << "\"];" << std::endl;
            is_access_node = true;
        } else if (const data_flow::LibraryNode* libnode = dynamic_cast<const data_flow::LibraryNode*>(node)) {
            this->stream_ << nodeId << " [shape=doubleoctagon,label=\"" << libnode->toStr() << "\"];" << std::endl;
            in_connectors = libnode->inputs();
        }

        std::unordered_set<std::string> unused_connectors(in_connectors.begin(), in_connectors.end());
        for (const data_flow::Memlet& iedge : block.dataflow().in_edges(*node)) {
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

            this->stream_ << escapeDotId(src.element_id(), "n_") << " -> " << nodeId << " [label=\"   ";
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
                    this->visualizeSubset(sdfg, iedge.subset(), &iedge.base_type());
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
                    this->visualizeSubset(sdfg, iedge.subset(), &iedge.base_type());
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
                    auto literal_id = escapeDotId(node->element_id(), "n_") + "_" + escapeDotId(i, "in");
                    this->stream_ << literal_id << " [style=\"dotted\", label=\"" << in_conn << "\"];" << std::endl;
                    this->stream_ << literal_id << " -> " << nodeId << " [label=\"" << i << "\"]" << ";" << std::endl;
                }
            }
        }
    }
    this->stream_.setIndent(this->stream_.indent() - 4);
    this->stream_ << "}" << std::endl;
}

void DotVisualizer::visualizeSequence(const StructuredSDFG& sdfg, const structured_control_flow::Sequence& sequence) {
    std::string last_comp_name_tmp, last_comp_name_cluster_tmp;
    for (size_t i = 0; i < sequence.size(); ++i) {
        std::pair<const structured_control_flow::ControlFlowNode&, const structured_control_flow::Transition&> child =
            sequence.at(i);
        this->visualizeNode(sdfg, child.first);
        if ((i > 0) && !last_comp_name_tmp.empty() && !this->last_comp_name_.empty() &&
            last_comp_name_tmp != this->last_comp_name_) {
            this->stream_ << last_comp_name_tmp << " -> " << this->last_comp_name_ << " [";
            if (!last_comp_name_cluster_tmp.empty()) this->stream_ << "ltail=\"" << last_comp_name_cluster_tmp << "\",";
            if (!this->last_comp_name_cluster_.empty())
                this->stream_ << "lhead=\"" << this->last_comp_name_cluster_ << "\",";
            this->stream_ << "minlen=3]"
                          << ";" << std::endl;
        }
        last_comp_name_tmp = this->last_comp_name_;
        last_comp_name_cluster_tmp = this->last_comp_name_cluster_;
    }
}

void DotVisualizer::visualizeIfElse(const StructuredSDFG& sdfg, const structured_control_flow::IfElse& if_else) {
    auto id = escapeDotId(if_else.element_id(), "if_");
    this->stream_ << "subgraph cluster_" << id << " {" << std::endl;
    this->stream_.setIndent(this->stream_.indent() + 4);
    this->stream_ << "style=filled;shape=box;fillcolor=white;color=black;label=\"if:\";" << std::endl
                  << id << " [shape=point,style=invis,label=\"\"];" << std::endl;
    for (size_t i = 0; i < if_else.size(); ++i) {
        this->stream_ << "subgraph cluster_" << id << "_" << std::to_string(i) << " {" << std::endl;
        this->stream_.setIndent(this->stream_.indent() + 4);
        this->stream_ << "style=filled;shape=box;fillcolor=white;color=black;label=\""
                      << this->expression(if_else.at(i).second->__str__()) << "\";" << std::endl;
        this->visualizeSequence(sdfg, if_else.at(i).first);
        this->stream_.setIndent(this->stream_.indent() - 4);
        this->stream_ << "}" << std::endl;
    }
    this->stream_.setIndent(this->stream_.indent() - 4);
    this->stream_ << "}" << std::endl;
    this->last_comp_name_ = id;
    this->last_comp_name_cluster_ = "cluster_" + id;
}

void DotVisualizer::visualizeWhile(const StructuredSDFG& sdfg, const structured_control_flow::While& while_loop) {
    auto id = escapeDotId(while_loop.element_id(), "while_");
    this->stream_ << "subgraph cluster_" << id << " {" << std::endl;
    this->stream_.setIndent(this->stream_.indent() + 4);
    this->stream_ << "style=filled;shape=box;fillcolor=white;color=black;label=\"while:\";" << std::endl
                  << id << " [shape=point,style=invis,label=\"\"];" << std::endl;
    this->visualizeSequence(sdfg, while_loop.root());
    this->stream_.setIndent(this->stream_.indent() - 4);
    this->stream_ << "}" << std::endl;
    this->last_comp_name_ = id;
    this->last_comp_name_cluster_ = "cluster_" + id;
}

void DotVisualizer::visualizeFor(const StructuredSDFG& sdfg, const structured_control_flow::For& loop) {
    auto id = escapeDotId(loop.element_id(), "for_");
    this->stream_ << "subgraph cluster_" << id << " {" << std::endl;
    this->stream_.setIndent(this->stream_.indent() + 4);
    this->stream_ << "style=filled;shape=box;fillcolor=white;color=black;label=\"for: ";
    this->visualizeForBounds(loop.indvar(), loop.init(), loop.condition(), loop.update());
    this->stream_ << "\";" << std::endl << id << " [shape=point,style=invis,label=\"\"];" << std::endl;
    this->visualizeSequence(sdfg, loop.root());
    this->stream_.setIndent(this->stream_.indent() - 4);
    this->stream_ << "}" << std::endl;
    this->last_comp_name_ = id;
    this->last_comp_name_cluster_ = "cluster_" + id;
}

void DotVisualizer::visualizeReturn(const StructuredSDFG& sdfg, const structured_control_flow::Return& return_node) {
    auto id = escapeDotId(return_node.element_id(), "return_");
    this->stream_ << id << " [shape=cds,label=\" return " << return_node.data() << "\"];" << std::endl;
    this->last_comp_name_ = id;
    this->last_comp_name_cluster_.clear();
}
void DotVisualizer::visualizeBreak(const StructuredSDFG& sdfg, const structured_control_flow::Break& break_node) {
    auto id = escapeDotId(break_node.element_id(), "break_");
    this->stream_ << id << " [shape=cds,label=\" break  \"];" << std::endl;
    this->last_comp_name_ = id;
    this->last_comp_name_cluster_.clear();
}

void DotVisualizer::visualizeContinue(const StructuredSDFG& sdfg, const structured_control_flow::Continue& continue_node) {
    auto id = escapeDotId(continue_node.element_id(), "cont_");
    this->stream_ << id << " [shape=cds,label=\" continue  \"];" << std::endl;
    this->last_comp_name_ = id;
    this->last_comp_name_cluster_.clear();
}

void DotVisualizer::visualizeMap(const StructuredSDFG& sdfg, const structured_control_flow::Map& map_node) {
    auto id = escapeDotId(map_node.element_id(), "map_");
    this->stream_ << "subgraph cluster_" << id << " {" << std::endl;
    this->stream_.setIndent(this->stream_.indent() + 4);
    this->stream_ << "style=filled;shape=box;fillcolor=white;color=black;label=\"map: ";
    this->visualizeForBounds(map_node.indvar(), map_node.init(), map_node.condition(), map_node.update());
    this->stream_ << "\";" << std::endl << id << " [shape=point,style=invis,label=\"\"];" << std::endl;
    this->visualizeSequence(sdfg, map_node.root());
    this->stream_.setIndent(this->stream_.indent() - 4);
    this->stream_ << "}" << std::endl;
    this->last_comp_name_ = id;
    this->last_comp_name_cluster_ = "cluster_" + id;
}

void DotVisualizer::visualize() {
    this->stream_.clear();
    this->stream_ << "digraph " << escapeDotId(this->sdfg_.name()) << " {" << std::endl;
    this->stream_.setIndent(4);
    this->stream_ << "graph [compound=true];" << std::endl;
    this->stream_ << "subgraph cluster_" << escapeDotId(this->sdfg_.name()) << " {" << std::endl;
    this->stream_.setIndent(8);
    this->stream_ << "node [style=filled,fillcolor=white];" << std::endl
                  << "style=filled;color=lightblue;label=\"\";" << std::endl;
    this->visualizeSequence(this->sdfg_, this->sdfg_.root());
    this->stream_.setIndent(4);
    this->stream_ << "}" << std::endl;
    this->stream_.setIndent(0);
    this->stream_ << "}" << std::endl;
}

void DotVisualizer::writeToFile(const StructuredSDFG& sdfg, const std::filesystem::path* file) {
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
