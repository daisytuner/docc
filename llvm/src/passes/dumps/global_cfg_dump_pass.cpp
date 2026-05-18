#include "docc/passes/dumps/global_cfg_dump_pass.h"

#include "docc/analysis/global_cfg_analysis.h"

#include <fstream>
#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/bit.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>
#include <regex>
#include <vector>

extern llvm::cl::opt<bool> docc_debug_glbl;

namespace docc::passes {

static llvm::cl::opt<std::string> ViewNodes(
    "view-globalcfg-nodes",
    llvm::cl::init(".*"),
    llvm::cl::desc("For each entry-point dump reachable global CFG nodes that match the regex")
);

static llvm::cl::opt<std::string> PrintEntrypoints(
    "print-globalcfgs",
    llvm::cl::init(".*"),
    llvm::cl::desc("Print dot files of global CFG from the entry-points that match the regex")
);

static llvm::cl::opt<std::string> LibEntrypoints(
    "docc-globalcfg-lib-entry-points",
    llvm::cl::init(""),
    llvm::cl::desc("Path to a file containing a list of function names, defining relevant entry points into the "
                   "globalcfg. main is always included.")
);

static llvm::raw_ostream& warning() { return llvm::errs() << "[docc_llvm_plugin] warning: "; }

#define DOCC_DEBUG(X) \
    if (docc_debug_glbl) X

using Graph = boost::adjacency_list<boost::listS, boost::listS, boost::bidirectionalS>;
using Vertex = boost::graph_traits<Graph>::vertex_descriptor;
using Edge = boost::graph_traits<Graph>::edge_descriptor;
using AdjIter = boost::graph_traits<Graph>::adjacency_iterator;

struct FunScope {
    uint32_t modId;
    int64_t funcId;

    FunScope(uint32_t modId, int64_t funcId) : modId(modId), funcId(funcId) {};
    FunScope(const analysis::GlobalCFGNode& node) : modId(node.modId_), funcId(node.funcId_) {};
    FunScope(const analysis::GlobalCFGNode* node) : modId(node->modId_), funcId(node->funcId_) {};

    bool operator==(const FunScope& rhs) const {
        if (modId != rhs.modId) return false;
        return funcId == rhs.funcId;
    }
};

llvm::raw_ostream& operator<<(llvm::raw_ostream& a, const FunScope& scop) {
    a << scop.modId << ":" << scop.funcId;
    return a;
}

struct DAGCollector : boost::default_dfs_visitor {
    std::vector<const analysis::GlobalCFGNode*>& nodes;
    std::vector<const analysis::GlobalCFGEdge*>& edges;
    std::list<FunScope> modScopeStack;
    std::unordered_map<Vertex, boost::default_color_type>& colors;
    const analysis::GlobalCFGAnalysis& cfg;
    bool justLeft = false;

    DAGCollector(
        std::vector<const analysis::GlobalCFGNode*>& nodes,
        std::vector<const analysis::GlobalCFGEdge*>& edges,
        std::unordered_map<Vertex, boost::default_color_type>& colors,
        const analysis::GlobalCFGAnalysis& cfg
    )
        : nodes(nodes), edges(edges), colors(colors), cfg(cfg) {};

    void discover_vertex(Vertex v, const Graph& g) {
        auto& n = cfg.getNode(v);
        nodes.push_back(&n);
    };

    void start_vertex(Vertex v, const Graph& g) {
        auto& node = cfg.getNode(v);
        modScopeStack.push_back(std::move(FunScope(node)));
    }

    void forward_or_cross_edge(Edge e, const Graph& g) {
        auto& cE = cfg.getEdge(e);
        edges.push_back(&cE);
        justLeft = true;
    }

    void back_edge(Edge e, const Graph& g) {
        auto& cE = cfg.getEdge(e);
        edges.push_back(&cE);
        justLeft = true;
    }

    void examine_edge(Edge e, const Graph& g) {
        auto& cE = cfg.getEdge(e);
        if (cE.Type_ == analysis::EdgeType::Return) {
            auto it = modScopeStack.crbegin();
            auto last = *it++;

            if (last == FunScope(cE.From_)) {
                if (modScopeStack.size() >= 2) {
                    auto prev = *it;
                    if (prev != FunScope(cE.To_)) { // return edges we want to block
                        colors[cE.To_->Vertex_] = boost::black_color;
                        DOCC_DEBUG(
                            llvm::dbgs() << "Blocking return into different scope at " << cE.From_->node_id_ << "# -> "
                                         << cE.To_->node_id_ << "#\n"
                        );
                        justLeft = true;
                    }
                } else {
                    colors[cE.To_->Vertex_] = boost::black_color;
                    DOCC_DEBUG(
                        llvm::dbgs() << "Blocking return into non-existent scope at " << cE.From_->node_id_ << "# -> "
                                     << cE.To_->node_id_ << "#\n"
                    );
                    justLeft = true;
                }
            } else {
                llvm::dbgs() << "ERROR: walking edge " << analysis::getEdgeTypeName(cE.Type_) << " from "
                             << FunScope(cE.From_) << " (" << cE.From_->node_id_ << "#) to " << FunScope(cE.To_) << " ("
                             << cE.To_->node_id_ << "#) but stack has ";
                if (modScopeStack.size() >= 2) {
                    auto prev = *it;
                    llvm::dbgs() << prev;
                } else {
                    llvm::dbgs() << "[]";
                }
                llvm::dbgs() << "->" << last << "\n";
            }
        }
    }

    void tree_edge(Edge e, const Graph& g) {
        auto& cE = cfg.getEdge(e);
        edges.push_back(&cE);
        if (cE.Type_ == analysis::EdgeType::CallCrossModule || cE.Type_ == analysis::EdgeType::CallInternal ||
            cE.Type_ == analysis::EdgeType::Return) {
            modScopeStack.push_back(std::move(FunScope(cE.To_)));
        }
    };

    void finish_edge(Edge e, const Graph& g) {
        auto& cE = cfg.getEdge(e);
        if (!cE.incRet_ && (cE.Type_ == analysis::EdgeType::CallCrossModule ||
                            cE.Type_ == analysis::EdgeType::CallInternal || cE.Type_ == analysis::EdgeType::Return)) {
            auto it = modScopeStack.crbegin();
            auto last = *it++;

            if (last == FunScope(cE.To_)) {
                if (modScopeStack.size() >= 2) {
                    auto prev = *it;

                    if (prev == FunScope(cE.From_)) {
                        modScopeStack.pop_back();
                    } else if (cE.Type_ != analysis::EdgeType::Return) {
                        llvm::dbgs() << "ERROR: walking back edge " << analysis::getEdgeTypeName(cE.Type_) << " from "
                                     << FunScope(cE.From_) << " (" << cE.From_->node_id_ << "#) to " << FunScope(cE.To_)
                                     << " (" << cE.To_->node_id_ << "#) but stack has " << prev << "->" << last << "\n";
                    }
                }
            } else if (justLeft) {
                justLeft = false;
            } else { // we block some returns, so those are probably ok to not match, because we never walked them
                llvm::dbgs() << "ERROR: broke scope stack\n";
            }
        };
    }
};

static std::pair<std::vector<const analysis::GlobalCFGNode*>, std::vector<const analysis::GlobalCFGEdge*>>
populateEntryPointSubgraph(const analysis::GlobalCFGAnalysis& CFG, const analysis::GlobalCFGNode* EP) {
    std::vector<const analysis::GlobalCFGNode*> nodes;
    std::vector<const analysis::GlobalCFGEdge*> edges;
    std::unordered_map<Vertex, boost::default_color_type> vertex_colors;
    DAGCollector visitor(nodes, edges, vertex_colors, CFG);

    auto& g = CFG.getGraph();

    auto* exitPoints = CFG.getExitPoints(EP->Id_);
    if (exitPoints) {
        for (auto& exit : *exitPoints) {
            auto [out_begin, out_end] = boost::out_edges(exit->Vertex_, g);
            for (auto it = out_begin; it != out_end; ++it) {
                auto& edge = CFG.getEdge(*it);
                if (edge.Type_ == analysis::EdgeType::Return) { // we need to mark nodes that are behind returns from
                                                                // exit-node as DO-NOT-VISIT, not the node itself
                                                                // (because as we compress, there are also edges for
                                                                // calls / subgraphs from the same node as well)
                    auto& after_return_node = *edge.To_;
                    vertex_colors[after_return_node.Vertex_] = boost::black_color; // terminate at global function
                                                                                   // exit-points
                }
            }
        }
    } else {
        DOCC_DEBUG(llvm::dbgs() << "No exit points known for " << EP->Name_ << ", unlimited graph\n");
    }

    boost::depth_first_visit(g, EP->Vertex_, visitor, boost::make_assoc_property_map(vertex_colors));

    return {nodes, edges};
}

void GlobalCFGPrinterPass::dumpToConsole(llvm::Module& Mod, analysis::GlobalCFGAnalysis& cfg) {
    std::string Buffer;
    llvm::raw_string_ostream OS(Buffer);
    std::regex Filter(ViewNodes);

    for (auto& Func : Mod.functions()) {
        if (Func.isDeclaration()) continue;
        const analysis::GlobalCFGNode* EP = cfg.getEntryPoint(Func.getName());
        if (EP == nullptr) continue;

        OS << "Entrypoint: " << EP->Name_ << "\n";
        auto [Nodes, Edges] = populateEntryPointSubgraph(cfg, EP);
        for (const analysis::GlobalCFGNode* N : Nodes)
            if (!N->Name_.empty())
                if (std::regex_match(N->Name_.data(), Filter)) {
                    OS << "  " << N->Name_ << "\n";
                }
    }

    llvm::outs() << "==BEGIN DUMP_GLOBAL_CALLS_DFS==\n" << OS.str() << "==END   DUMP_GLOBAL_CALLS_DFS==\n";
}

static std::string limitColumns(std::string Input, unsigned Max, uint32_t lines) {
    std::istringstream IS(Input);
    std::ostringstream OS;
    std::string line;

    while (std::getline(IS, line) && lines > 0) {
        if (line.length() > Max) {
            OS << line.substr(0, Max - 3) << "...\n";
        } else {
            OS << line << "\n";
        }
        lines--;
    }

    return OS.str();
}

std::string GlobalCFGPrinterPass::escapeDot(std::string input) {
    std::string output;
    // Reserve memory to prevent re-allocation overhead.
    // We assume the string might grow by ~20% on average.
    output.reserve(input.length() * 1.2);

    for (char c : input) {
        switch (c) {
            // Cases for all the symbols you want to escape
            case '"':
            case '[':
            case ']':
            case '<':
            case '>':
            case '{':
            case '}':
                output += '\\'; // Append the backslash first
                output += c; // Then append the symbol
                break;
            default:
                output += c; // Just append the normal character
                break;
        }
    }
    return output;
}

static void printNode(llvm::raw_ostream& OS, const analysis::GlobalCFGNode& N, bool sparing = false) {
    OS << "[";
    bool as_record = N.evtSteps_ > 0 || (N.BB_ != nullptr && N.last_insn_interesting_) || N.sdfg_node_ != nullptr;
    std::string endLine;
    std::stringstream add_style;
    if (N.specialType_ == analysis::CfgSpecialType::H2D) {
        add_style << ", style=filled, fillcolor=lightgreen, shape=octagon";
    } else if (N.specialType_ == analysis::CfgSpecialType::D2H) {
        add_style << ", style=filled, fillcolor=lightsalmon, shape=octagon";
    } else if (N.specialType_ == analysis::CfgSpecialType::Return) {
        add_style << ", style=filled, fillcolor=sienna";
    } else if (N.specialType_ == analysis::CfgSpecialType::ErrorHandling) {
        add_style << ", style=filled, fillcolor=red";
    } else if (N.Id_) {
        add_style << ", style=filled, fillcolor=green";
    }
    if (as_record) {
        OS << "shape=record, ";
        OS << "label=\"{";
        endLine = "|";
    } else {
        OS << "label=\"";
        endLine = "\n";
    }
    OS << N.node_id_ << "# ";
    if (auto sdfg = N.sdfg_) {
        OS << "[SDFG] ";
    }
    if (!N.Name_.empty()) {
        OS << N.Name_ << "()" << endLine;
    }
    if (auto sdfgBlock = N.sdfg_node_) {
        OS << "N " << sdfgBlock->element_id();
        if (auto* mapLoop = dynamic_cast<const sdfg::structured_control_flow::Map*>(sdfgBlock)) {
            OS << " Map ";
            if (!sparing) {
                OS << GlobalCFGPrinterPass::escapeDot(mapLoop->indvar()->get_name()) << ": "
                   << GlobalCFGPrinterPass::escapeDot(mapLoop->init()->__str__()) << "; "
                   << GlobalCFGPrinterPass::escapeDot(mapLoop->condition()->__str__()) << "; "
                   << GlobalCFGPrinterPass::escapeDot(mapLoop->update()->__str__());
            }
            OS << endLine;
            add_style << ", style=filled, fillcolor=";
            auto& schedType = mapLoop->schedule_type();
            if (schedType.value() != sdfg::structured_control_flow::ScheduleType_Sequential::value()) {
                add_style << "hotpink2";
                OS << GlobalCFGPrinterPass::escapeDot(schedType.value());
                if (!schedType.properties().empty()) {
                    OS << " (";
                    for (const auto& [key, value] : schedType.properties()) {
                        OS << key << ":" << GlobalCFGPrinterPass::escapeDot(value) << ", ";
                    }
                    OS << ")";
                }
                OS << endLine;
            } else {
                add_style << "lightcyan";
            }
        } else {
            add_style << ", style=filled, fillcolor=" << (N.Name_.empty() ? "lightcyan" : "skyblue");
            if (auto* strucLoop = dynamic_cast<const sdfg::structured_control_flow::StructuredLoop*>(sdfgBlock)) {
                OS << " Loop ";
                if (!sparing) {
                    OS << GlobalCFGPrinterPass::escapeDot(strucLoop->indvar()->get_name()) << ": "
                       << GlobalCFGPrinterPass::escapeDot(strucLoop->init()->__str__()) << "; "
                       << GlobalCFGPrinterPass::escapeDot(strucLoop->condition()->__str__()) << "; "
                       << GlobalCFGPrinterPass::escapeDot(strucLoop->update()->__str__());
                }
                OS << endLine;
            } else if (auto* whileLoop = dynamic_cast<const sdfg::structured_control_flow::While*>(sdfgBlock)) {
                OS << " While " << endLine;
            } else {
                OS << endLine;
            }
        }
        for (int i = 0; i < N.evtSteps_; ++i) {
            OS << "<e" << i << "> \\ |";
        }
        OS << "<e" << N.evtSteps_ << "> [end]";
    } else if (auto bb = N.BB_) {
        OS << "BB ";
        std::string Buffer;
        llvm::raw_string_ostream TmpOS(Buffer);
        auto last_idx = N.last_bb_insn_idx_;
        if (!sparing) {
            bb->printAsOperand(TmpOS, false);


            if (N.first_bb_insn_idx_ >= 0) {
                TmpOS << " cont. [" << N.first_bb_insn_idx_;
            }

            llvm::BasicBlock* end_bb = N.fused_bb_last_;
            if (end_bb) {
                TmpOS << endLine;
                end_bb->printAsOperand(TmpOS, false);
            }

            if (last_idx >= 0) {
                TmpOS << " ..." << last_idx << "]";
            }
        }

        if (N.evtSteps_ > 0) {
            TmpOS << endLine;
            for (int i = 0; i < N.evtSteps_; ++i) {
                TmpOS << "<e" << i << "> \\ |";
            }
        }
        if (!sparing && N.last_insn_interesting_) {
            auto* end_bb = bb;
            llvm::Instruction* insn = (last_idx < 0) ? &*end_bb->rbegin() : &*std::next(end_bb->begin(), last_idx);
            if (N.evtSteps_ == 0) {
                TmpOS << endLine;
            } else {
                TmpOS << "<e" << N.evtSteps_ << "> ";
            }
            std::string limited;
            llvm::raw_string_ostream tos(limited);
            insn->print(tos, false);
            TmpOS << limitColumns(tos.str(), 40, 2);
        } else if (N.evtSteps_ > 0) {
            TmpOS << "<e" << N.evtSteps_ << "> [end]";
        }

        OS << GlobalCFGPrinterPass::escapeDot(TmpOS.str());
    } else if (N.Inst_ != nullptr) {
        std::string Buffer;
        llvm::raw_string_ostream TmpOS(Buffer);
        if (!sparing) {
            N.Inst_->print(TmpOS, false);
        } else {
            TmpOS << "Inst";
        }
        OS << GlobalCFGPrinterPass::escapeDot(TmpOS.str());
    }
    if (as_record) {
        OS << "}\"";
    } else {
        OS << "\"";
    }
    OS << add_style.str();
    OS << "];\n";
}

static void printSubgraphDot(
    std::ostream& OS,
    const std::vector<const analysis::GlobalCFGNode*>& Nodes,
    const std::vector<const analysis::GlobalCFGEdge*>& Edges,
    bool sparing = false
) {
    OS << "digraph G {\n";

    unsigned int Id = 0;
    std::unordered_map<const analysis::GlobalCFGNode*, unsigned int> Ids;
    for (auto* N : Nodes) {
        Id = N->node_id_;
        Ids[N] = Id;
        OS << Id << " ";
        std::string Buffer;
        llvm::raw_string_ostream TmpOS(Buffer);
        printNode(TmpOS, *N, sparing);
        OS << TmpOS.str();
    }

    for (auto* E : Edges) {
        auto Name = getEdgeTypeName(E->Type_);

        // FIXME: It seems like we miss source nodes in some cases
        auto FromIt = Ids.find(E->From_);
        if (FromIt == Ids.end()) {
            auto& Errs = warning() << "Cannot find source node for " << Name << " edge from " << E->From_->node_id_
                                   << " to " << E->To_->node_id_ << "\n";
            printNode(Errs, *E->From_);
        }
        auto ToIt = Ids.find(E->To_);
        if (ToIt == Ids.end()) {
            auto& Errs = warning() << "Cannot find target node for " << Name << " edge from " << E->From_->node_id_
                                   << " to " << E->To_->node_id_ << "\n";
            printNode(Errs, *E->From_);
        }
        if (FromIt == Ids.end() || ToIt == Ids.end()) continue;

        uint32_t fromNodeId = Ids.at(E->From_);
        std::string FromId = E->fromEvtIdx_ < 0 ? std::to_string(fromNodeId)
                                                : std::to_string(fromNodeId) + ":e" + std::to_string(E->fromEvtIdx_);
        uint32_t toNodeId = Ids.at(E->To_);
        std::string ToId = E->toEvtIdx_ < 0 ? std::to_string(toNodeId)
                                            : std::to_string(toNodeId) + ":e" + std::to_string(E->toEvtIdx_);
        OS << FromId << "->" << ToId << " [label=\"" << Name << "\"";
        if (E->incRet_) {
            OS << ", dir=both";
        }
        OS << "];\n";
    }

    OS << "}\n";
}

std::vector<std::string> get_entry_points(const std::string& path) {
    if (LibEntrypoints == "") {
        return {};
    }

    bool start_parsing = false;
    std::vector<std::string> entry_points;
    std::ifstream infile(path);
    std::string line;
    while (std::getline(infile, line)) {
        entry_points.push_back(line);
    }

    return entry_points;
}

llvm::PreservedAnalyses GlobalCFGPrinterPass::
    run(llvm::Module& Mod, llvm::ModuleAnalysisManager& MAM, analysis::AnalysisManager& GAM) {
    llvm::dbgs() << "Running GlobalCFGPrinterPass on module: " << Mod.getName() << "\n";

    auto& glb = GAM.get<analysis::GlobalCFGAnalysis>();

    if (out_path_) {
        dumpToDotFile(*out_path_, Mod, glb);
    } else {
        dumpToConsole(Mod, glb);
    }

    return llvm::PreservedAnalyses::all();
}

void GlobalCFGPrinterPass::dumpToDotFile(const std::string& path, llvm::Module& Mod, analysis::GlobalCFGAnalysis& cfg) {
    std::vector<std::string> EPs = get_entry_points(LibEntrypoints);
    EPs.push_back("main");

    std::vector<llvm::StringRef> local_names;
    for (auto& Func : Mod.functions()) {
        if (Func.isDeclaration()) continue;
        local_names.push_back(Func.getName());
    }

    std::regex modNameStripper(".*?([^/]+)$");

    for (llvm::StringRef entry_name : EPs) {
        auto* EP = cfg.getEntryPoint(entry_name);
        if (EP == nullptr) {
            warning() << "Cannot find entry-point node for " << entry_name << "() function\n";
            continue;
        }

        bool local = false;
        for (auto name : local_names) {
            if (name.str() == entry_name.str()) {
                local = true;
                break;
            }
        }
        if (!local) {
            continue;
        }

        auto full_mod_name = Mod.getName().str();
        auto mod_name = std::regex_replace(full_mod_name, modNameStripper, "$1");

        auto DotFileName = std::filesystem::path(path) / (mod_name + "." + entry_name + ".cfg.dot").str();
        auto [Nodes, Edges] = populateEntryPointSubgraph(cfg, EP);

        create_directories(DotFileName.parent_path());

        std::ofstream OS(DotFileName);
        printSubgraphDot(OS, Nodes, Edges, false);
        OS.close();
    }
}

} // namespace docc::passes
