#pragma once

#include <llvm/ADT/StringRef.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/ModuleSummaryIndex.h>
#include <llvm/Support/raw_ostream.h>

#include <boost/graph/adjacency_list.hpp>

#include "analysis.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg_registry.h"

namespace docc::analysis {

using Graph = boost::adjacency_list<boost::listS, boost::listS, boost::bidirectionalS>;
using Vertex = boost::graph_traits<Graph>::vertex_descriptor;
using Edge = boost::graph_traits<Graph>::edge_descriptor;

static uint32_t next_node_id = 0;

class SDFGCfgVisitor;

enum class CfgSpecialType { None = 0, H2D, D2H, ErrorHandling, Return };

struct GlobalCFGNode {
    GlobalCFGNode(Vertex V, uint32_t modId, int64_t funcId)
        : Vertex_(V), node_id_(next_node_id++), modId_(modId), funcId_(funcId) {}
    void print(llvm::raw_ostream &OS);

    Vertex Vertex_;
    uint32_t node_id_;
    uint32_t modId_;
    int64_t funcId_;
    llvm::GlobalValue::GUID Id_ = 0;
    std::string Name_;

    docc::analysis::SDFGHolder *sdfg_ = nullptr;
    const sdfg::structured_control_flow::ControlFlowNode *sdfg_node_ = nullptr;
    llvm::BasicBlock *BB_ = nullptr;
    llvm::BasicBlock *fused_bb_last_ = nullptr;
    int32_t first_bb_insn_idx_ = -1;
    int32_t last_bb_insn_idx_ = -1;
    bool last_insn_interesting_ = false;

    llvm::Instruction *Inst_ = nullptr;
    uint16_t evtSteps_ = 0;
    bool suppress_ = false;
    CfgSpecialType specialType_ = CfgSpecialType::None;
};

enum class EdgeType {
    Branch,
    Sequence,
    CallInternal,
    CallExternal,
    CallCrossModule,
    //    CallSDFGInternal,
    //    CallSDFGCrossModule,
    //    CallSDFGCrossLibrary,
    CallIndirect,
    Return
};

inline const char *getEdgeTypeName(EdgeType Type) {
    switch (Type) {
        case EdgeType::Branch:
            return "Branch";
        case EdgeType::Sequence:
            return "Sequence";
        case EdgeType::CallInternal:
            return "CallInternal";
        case EdgeType::CallExternal:
            return "CallExternal";
        case EdgeType::CallCrossModule:
            return "CallCrossModule";
            //        case EdgeType::CallSDFGInternal:
            //            return "CallSDFGInternal";
            //        case EdgeType::CallSDFGCrossModule:
            //            return "CallSDFGCrossModule";
            //        case EdgeType::CallSDFGCrossLibrary:
            //            return "CallSDFGCrossLibrary";
        case EdgeType::CallIndirect:
            return "CallIndirect";
        case EdgeType::Return:
            return "Return";
    }
    return "Unknown";
}

struct GlobalCFGEdge {
    GlobalCFGEdge(
        Edge E,
        EdgeType type,
        GlobalCFGNode *from,
        GlobalCFGNode *to,
        int32_t fromEvtIdx = -1,
        bool incRet = false,
        int32_t toEvtIdx = -1
    )
        : Edge_(E), Type_(type), From_(from), To_(to), fromEvtIdx_(fromEvtIdx), toEvtIdx_(toEvtIdx), incRet_(incRet) {}

    Edge Edge_;
    EdgeType Type_;
    GlobalCFGNode *From_;
    GlobalCFGNode *To_;
    int32_t fromEvtIdx_;
    int32_t toEvtIdx_;
    bool incRet_;
};

class GlobalCFGAnalysis : public Analysis {
    friend class GlobalCFGBuilder;
    friend class SDFGCfgVisitor;

public:
    static bool available(AnalysisManager &am);

    void run(AnalysisManager &AM) override;

    const Graph &getGraph() const { return Graph_; }
    const GlobalCFGEdge &getEdge(Edge E) const { return *Edges_.at(E); }
    const GlobalCFGNode &getNode(Vertex V) const { return *Nodes_.at(V); }
    const GlobalCFGNode *getEntryPoint(llvm::StringRef Name) const { return findNodeExternallyVisible(Name); }
    const std::vector<GlobalCFGNode *> *getExitPoints(llvm::GlobalValue::GUID id) const;
    const std::vector<GlobalCFGNode *> *getExitPoints(llvm::StringRef name) const;


private:
    Graph Graph_;
    std::map<Vertex, std::unique_ptr<GlobalCFGNode>> Nodes_;
    std::map<Edge, std::unique_ptr<GlobalCFGEdge>> Edges_;
    std::unordered_map<llvm::GlobalValue::GUID, GlobalCFGNode *> EntryPoints_;
    std::unordered_map<llvm::GlobalValue::GUID, std::vector<GlobalCFGNode *>> ExitPoints_;
    llvm::LLVMContext Ctx_;
    std::vector<std::unique_ptr<llvm::Module>> Modules_;
    llvm::StringMap<uint32_t> moduleIds_;
    int64_t next_func_id_ = 0;

    void addEntryPoint(llvm::GlobalValue::GUID Id, GlobalCFGNode *Node);
    GlobalCFGNode *findNodeExternallyVisible(llvm::StringRef Name) const;

    uint32_t getModuleId(llvm::StringRef modPath);
};

} // namespace docc::analysis
