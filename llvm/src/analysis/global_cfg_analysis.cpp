#include "docc/analysis/global_cfg_analysis.h"

#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/IR/CFG.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/ModuleSummaryIndex.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>

#include <cassert>
#include <memory>
#include <stack>
#include <unordered_map>

#include "docc/analysis/sdfg_registry.h"
#include "sdfg/data_flow/library_nodes/call_node.h"
#include "sdfg/targets/offloading/data_offloading_node.h"
#include "sdfg/targets/offloading/external_offloading_node.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

using namespace docc;

static llvm::ExitOnError ExitOnErr{"[docc_llvm_plugin] error: "};

llvm::cl::opt<bool> docc_debug_glbl("docc-debug-glbl");

#define DOCC_DEBUG(X) \
    if (docc_debug_glbl) X

static llvm::cl::opt<bool> IncludeAllIntrinsics(
    "docc-include-all-intrinsics",
    llvm::cl::init(false),
    llvm::cl::desc("Add external call nodes for all intrinsics, e.g. llvm.dbg.value")
);

namespace docc::analysis {

bool GlobalCFGAnalysis::available(AnalysisManager &am) { return SDFGRegistry::is_link_time(am); }

const std::vector<GlobalCFGNode *> *GlobalCFGAnalysis::getExitPoints(llvm::GlobalValue::GUID id) const {
    auto it = ExitPoints_.find(id);
    if (it != ExitPoints_.end()) {
        return &(it->second);
    } else {
        return nullptr;
    }
}

const std::vector<GlobalCFGNode *> *GlobalCFGAnalysis::getExitPoints(llvm::StringRef Name) const {
    return getExitPoints(llvm::GlobalValue::getGUID(Name));
}

class GlobalCFGBuilder {
public:
    GlobalCFGAnalysis &CFG_;
    llvm::ModuleSummaryIndex &CombinedIndex_;
    llvm::StringMap<std::vector<std::tuple<GlobalCFGNode *, GlobalCFGNode *, int32_t>>> ReturnEdgeTargets_;
    llvm::StringMap<std::vector<std::tuple<GlobalCFGNode *>>> ReturnEdgeOrigins_;
    SDFGRegistry &registry_;

    GlobalCFGBuilder(GlobalCFGAnalysis &CFG, llvm::ModuleSummaryIndex &CombinedIndex, SDFGRegistry &registry)
        : CFG_(CFG), CombinedIndex_(CombinedIndex), registry_(registry) {}

    GlobalCFGNode &addNode(uint32_t modId, int64_t funcId = -1L);
    void addModule(uint32_t modId, llvm::Module &Mod);
    void insertReturnEdges();

    void registerFunctionExit(llvm::StringRef func, GlobalCFGNode &node, bool global = true) {
        ReturnEdgeOrigins_[func].emplace_back(&node);
        node.specialType_ = CfgSpecialType::Return;
        if (global) {
            CFG_.ExitPoints_[llvm::GlobalValue::getGUID(func)].push_back(&node);
        }
    }

    void registerCallReturnSite(
        llvm::StringRef func, GlobalCFGNode &callNode, GlobalCFGNode &callReturnNode, int32_t step = -1
    ) {
        ReturnEdgeTargets_[func].push_back({&callReturnNode, &callNode, step});
    }

    GlobalCFGNode &addNodeGlobal(uint32_t modId, int64_t funcId, llvm::GlobalValue::GUID Id) {
        auto &N = addNode(modId, funcId);
        N.Id_ = Id;
        return N;
    }

    GlobalCFGNode &addNodeInternal(uint32_t modId, int64_t funcId, llvm::BasicBlock &BB) {
        auto &N = addNode(modId, funcId);
        N.BB_ = &BB;
        return N;
    };
    GlobalCFGNode &addNodeVirtual(uint32_t modId, llvm::Instruction &Inst) {
        auto &N = addNode(modId);
        N.Inst_ = &Inst;
        return N;
    };
    GlobalCFGNode &addNodeAuxiliary(uint32_t modId, int64_t funcId) {
        auto &N = addNode(modId, funcId);
        // TODO: Do we need any info, e.g. artificial return point?
        return N;
    };
    GlobalCFGNode &addNodeExternal(uint32_t modId, const std::string &Name) {
        auto &N = addNode(modId);
        N.Name_ = Name;
        return N;
    };
    GlobalCFGEdge &addEdge(
        GlobalCFGNode &From,
        GlobalCFGNode &To,
        EdgeType Type,
        int32_t fromEvtIdx = -1,
        bool incRet = false,
        int32_t toEvtIdx = -1
    ) {
        auto [E, Added] = boost::add_edge(From.Vertex_, To.Vertex_, CFG_.Graph_);
        assert(Added && "Duplicate edge");
        auto It = CFG_.Edges_
                      .insert({E, std::make_unique<GlobalCFGEdge>(E, Type, &From, &To, fromEvtIdx, incRet, toEvtIdx)})
                      .first;
        return *It->second;
    }

private:
    void addBasicBlockEdges(
        uint32_t modId,
        int64_t funcId,
        llvm::BasicBlock *BB,
        GlobalCFGNode *BBNode,
        const llvm::DenseMap<llvm::BasicBlock *, GlobalCFGNode *> &ModuleNodes
    );

    void *addFromSdfg(
        uint32_t modId,
        llvm::Function &func,
        llvm::BasicBlock &bb,
        GlobalCFGNode &node,
        SDFGHolder *sdfg,
        llvm::DenseMap<llvm::BasicBlock *, GlobalCFGNode *> &ModuleNodes
    );

    SDFGHolder *find_sdfg(llvm::Module &module, llvm::StringRef name);
};

SDFGHolder *GlobalCFGBuilder::find_sdfg(llvm::Module &module, llvm::StringRef name) {
    auto &sdfgs = registry_.at(module);

    auto it = sdfgs.find(name.str());

    if (it != sdfgs.end()) {
        return it->second.get();
    } else {
        return nullptr;
    }
}

struct SdfgVisState {
    const ControlFlowNode &current_scope;
    GlobalCFGNode &node;
    std::vector<GlobalCFGNode *> calls;
};

class SDFGCfgVisitor : public sdfg::visitor::ActualStructuredSDFGVisitor {
private:
    uint32_t modId_;
    int64_t funcId_;
    const GlobalCFGNode &sdfg_root_;
    GlobalCFGBuilder &builder_;

    std::stack<SdfgVisState, std::list<SdfgVisState>> stack_;

    void add_transfer(
        sdfg::data_flow::LibraryNode *transfer_call, bool h2d, const std::string &func_id, const std::string &label
    ) {
        auto &state = stack_.top();
        auto &prev_node = state.node;
        auto &scope = state.current_scope;

        commit(state);

        stack_.pop();

        auto &transfer_node = builder_.addNode(modId_);
        transfer_node.Name_ = label;
        transfer_node.Id_ = llvm::GlobalValue::getGUID(func_id);
        transfer_node.specialType_ = h2d ? CfgSpecialType::H2D : CfgSpecialType::D2H;
        builder_.addEdge(prev_node, transfer_node, EdgeType::Sequence, prev_node.evtSteps_);

        auto &succ_node = builder_.addNode(modId_, funcId_);
        succ_node.sdfg_node_ = prev_node.sdfg_node_;

        stack_.emplace(scope, succ_node);
        builder_.addEdge(transfer_node, succ_node, EdgeType::Sequence);
    }

    void add_call(sdfg::data_flow::CallNode *call) {
        auto &state = stack_.top();
        auto step = state.node.evtSteps_++;
        auto &call_target_name = call->callee_name();
        auto *known_call = builder_.CFG_.findNodeExternallyVisible(call_target_name);
        if (known_call) {
            auto edgeType = known_call->modId_ != sdfg_root_.modId_ ? EdgeType::CallCrossModule
                                                                    : EdgeType::CallInternal;
            builder_.addEdge(state.node, *known_call, edgeType, step, true);
            //            builder_.registerCallReturnSite(call_target_name, *known_call, state.node, step);
        } else {
            auto &callee_node = builder_.addNodeExternal(modId_, call_target_name);
            callee_node.Id_ = llvm::GlobalValue::getGUID(call_target_name);
            builder_.addEdge(state.node, callee_node, EdgeType::CallExternal, step, true);
        }
    }

    void enterScope(sdfg::structured_control_flow::ControlFlowNode &scope) {
        auto &current = stack_.top();
        auto &parent_node = current.node;

        auto &new_node = builder_.addNode(modId_, funcId_);
        new_node.sdfg_node_ = &scope;

        builder_.addEdge(parent_node, new_node, EdgeType::Sequence, parent_node.evtSteps_++, true);

        stack_.emplace(scope, new_node);
    }

    void commit(SdfgVisState &state) {}

    void leaveScope(sdfg::structured_control_flow::ControlFlowNode &scope) {
        assert(&stack_.top().current_scope == &scope);

        auto &leaving_state = stack_.top();
        auto &leaving_node = leaving_state.node;

        commit(leaving_state);

        stack_.pop();
    }

public:
    using sdfg::visitor::ActualStructuredSDFGVisitor::visit;

    SDFGCfgVisitor(
        GlobalCFGBuilder &builder,
        GlobalCFGNode &root_node,
        const sdfg::structured_control_flow::ControlFlowNode &root_scope,
        uint32_t modId
    )
        : builder_(builder), sdfg_root_(root_node), modId_(modId), funcId_(root_node.funcId_) {
        root_node.sdfg_node_ = &root_scope;
        stack_.push(SdfgVisState{.current_scope = root_scope, .node = root_node});
    }

    bool handleStructuredLoop(StructuredLoop &node) override {
        enterScope(node);

        dispatch(node.root());

        leaveScope(node);

        return true;
    }

    bool visit(While &node) override {
        enterScope(node);

        dispatch(node.root());

        leaveScope(node);

        return true;
    }

    bool visit(Return &node) override {
        auto &state = stack_.top();
        builder_.registerFunctionExit(sdfg_root_.Name_, state.node);

        return true;
    }

    bool visit(Block &node) override {
        for (auto &n : node.dataflow().nodes()) {
            if (auto *external_offload = dynamic_cast<sdfg::offloading::ExternalDataOffloadingNode *>(&n)) {
                if (!external_offload->has_transfer()) {
                    continue;
                }
                add_transfer(
                    external_offload,
                    external_offload->is_h2d(),
                    external_offload->callee_name(),
                    external_offload->callee_name() + ":" + std::to_string(external_offload->transfer_index())
                );
            } else if (auto *data_transfer = dynamic_cast<sdfg::offloading::DataOffloadingNode *>(&n)) {
                if (!data_transfer->has_transfer()) {
                    continue;
                }
                add_transfer(
                    data_transfer, data_transfer->is_h2d(), data_transfer->code().value(), data_transfer->code().value()
                );
            } else if (auto *call_node = dynamic_cast<sdfg::data_flow::CallNode *>(&n)) {
                add_call(call_node);
            }
        }

        return true;
    }

    bool visit(Sequence &node) override {
        ActualStructuredSDFGVisitor::visit(node);

        auto &state = stack_.top();
        if (stack_.size() == 1 && &state.current_scope == &node) { // we are at the end of root_scope
            builder_.registerFunctionExit(sdfg_root_.Name_, state.node);
            // TODO this is only for implicit returns and will be double if
            // there is also an explicit one at the end (for a retVal).
            // Future SDFGS should not allow implicit returns!
        }

        return true;
    }
};

void *GlobalCFGBuilder::addFromSdfg(
    uint32_t modId,
    llvm::Function &func,
    llvm::BasicBlock &bb,
    GlobalCFGNode &node,
    SDFGHolder *sdfg_holder,
    llvm::DenseMap<llvm::BasicBlock *, GlobalCFGNode *> &ModuleNodes
) {
    node.sdfg_ = sdfg_holder;
    auto [lock, sdfg] = sdfg_holder->get_for_read();
    node.Name_ = sdfg->name();

    auto &rootSeq = sdfg->root();

    auto visitor = SDFGCfgVisitor(*this, node, rootSeq, modId);

    visitor.visit(const_cast<sdfg::structured_control_flow::Sequence &>(rootSeq));

    return &node;
}

void GlobalCFGBuilder::addModule(uint32_t modId, llvm::Module &Mod) {
    // Create nodes for all basic blocks in the module
    llvm::DenseMap<llvm::BasicBlock *, GlobalCFGNode *> Nodes;
    for (llvm::Function &Func : Mod) {
        if (Func.isDeclaration()) continue;

        auto func_name = Func.getName();

        auto already_known_entry_node = CFG_.findNodeExternallyVisible(func_name);
        auto funcId = already_known_entry_node ? already_known_entry_node->funcId_ : CFG_.next_func_id_++;

        // Handle definitions that might be replaced at link-time
        assert(!Func.isInterposable() && "TODO: interposable linkage");
        // Ignore thin-link duplicates for cross-module inlining
        if (Func.hasAvailableExternallyLinkage()) {
            DOCC_DEBUG(llvm::dbgs() << "  Ignore thin-link duplicate: " << func_name << "\n");
            continue;
        }

        DOCC_DEBUG(llvm::dbgs() << " Processing function " << func_name << " in " << Mod.getName() << "\n");

        auto sdfg = find_sdfg(Mod, func_name);

        for (llvm::BasicBlock &BB : Func) {
            GlobalCFGNode *node = nullptr;

            // Public functions generate symbols, which are resolved by name.
            // Symbols with external visibility are known from thin-LTO module summaries.
            if (BB.isEntryBlock()) {
                if (already_known_entry_node) { // add the name & BB to already known cross-module impls
                    node = already_known_entry_node;
                    assert(already_known_entry_node->Id_ && "External symbols have global value summaries");
                    assert(
                        already_known_entry_node->funcId_ >= 0 && "External symbols should already have funcId assigned"
                    );
                    if (already_known_entry_node->BB_) { // some other module has already parsed this function, so abort
                        break;
                    }
                } else {
                    node = &addNodeInternal(modId, funcId, BB);
                }
                node->Name_ = func_name;
                node->BB_ = &BB;

                if (sdfg) { // override with SDFG details
                    addFromSdfg(modId, Func, BB, *node, sdfg, Nodes);
                }
            } else if (sdfg) {
                //                node = &addNodeInternal(modId, funcId, BB); // skipping. Should never be a branch
                //                target from outside. rest handled in addFromSdfg node->sdfg_ = sdfg;
            } else {
                node = &addNodeInternal(modId, funcId, BB);
            }
            if (node) {
                Nodes.insert({&BB, node});
            }
        }
    }

    for (auto &&[BB, N] : Nodes) {
        if (!N->sdfg_) {
            addBasicBlockEdges(modId, N->funcId_, BB, N, Nodes);
        }
    }
}

void GlobalCFGBuilder::addBasicBlockEdges(
    uint32_t modId,
    int64_t funcId,
    llvm::BasicBlock *BB,
    GlobalCFGNode *BBNode,
    const llvm::DenseMap<llvm::BasicBlock *, GlobalCFGNode *> &ModuleNodes
) {
    GlobalCFGNode *PrevNode = BBNode;
    bool PrevWasNoReturn = false;
    int32_t insnId = -1;

    auto endCurrentNode = [&](llvm::BasicBlock *BB, llvm::Instruction &From) -> GlobalCFGNode & {
        if (PrevNode->BB_ == BB) { // we are terminating a BBNode early
            PrevNode->last_bb_insn_idx_ = insnId;
            PrevNode->last_insn_interesting_ = true;
        }
        return *PrevNode;
    };
    auto addReturnNode = [&](llvm::StringRef FnName, GlobalCFGNode *Fallback) -> GlobalCFGNode * {
        GlobalCFGNode *ReturnNode = &addNodeAuxiliary(modId, funcId);
        ReturnNode->BB_ = BB;
        ReturnNode->first_bb_insn_idx_ = insnId + 1;
        // Second pass injects return edges
        registerCallReturnSite(FnName, *Fallback, *ReturnNode);
        return ReturnNode;
    };

    for (llvm::Instruction &I : *BB) {
        ++insnId;
        if (PrevWasNoReturn) break;

        if (auto *Call = llvm::dyn_cast<llvm::CallBase>(&I)) {
            // Function-pointer call
            llvm::Function *Callee = Call->getCalledFunction();
            if (Callee == nullptr) {
                GlobalCFGNode &callNode = addNodeVirtual(modId, I);
                addEdge(*PrevNode, callNode, EdgeType::CallIndirect, PrevNode->evtSteps_++, true);
                continue;
            }

            llvm::StringRef FnName = Callee->getName();

            // Call to function inside the current module
            if (!Callee->isDeclaration()) {
                auto *target_bb = &Callee->getEntryBlock();

                auto it = ModuleNodes.find(target_bb);
                if (it == ModuleNodes.end()) {
                    llvm::dbgs() << "found no node for entry of " << FnName;
                    llvm::dbgs() << " even though it is a same-module call target!\n";

                    auto &errorNode = addNode(modId);
                    errorNode.Name_ = "Broken: " + FnName.str();
                    addEdge(*PrevNode, errorNode, EdgeType::CallInternal, PrevNode->evtSteps_++, true);
                    continue;
                } else {
                    auto *CalleeNode = it->second;
                    EdgeType edge_type;
                    edge_type = EdgeType::CallInternal;

                    addEdge(endCurrentNode(BB, I), *CalleeNode, edge_type);
                    PrevNode = addReturnNode(FnName, CalleeNode);
                    continue;
                }
            }

            // Call to any known entry point
            if (GlobalCFGNode *TargetNode = CFG_.findNodeExternallyVisible(FnName)) {
                EdgeType edge_type;
                edge_type = EdgeType::CallCrossModule;
                addEdge(endCurrentNode(BB, I), *TargetNode, edge_type);
                PrevNode = addReturnNode(FnName, TargetNode);
                continue;
            }

            // Call to intrinsic without side-effects
            if (!IncludeAllIntrinsics && Callee->isIntrinsic()) {
                if (I.mayHaveSideEffects()) {
                    DOCC_DEBUG(llvm::dbgs() << "llvm intrinsic " << FnName << " suppressed from GCFG\n");
                }
                continue;
            }

            GlobalCFGNode &CalleeNode = addNodeExternal(modId, FnName.str());
            addEdge(*PrevNode, CalleeNode, EdgeType::CallExternal, PrevNode->evtSteps_++, true);
        }
    }

    llvm::Instruction *TI = BB->getTerminator();

    if (auto *Ret = llvm::dyn_cast<llvm::ReturnInst>(TI)) {
        llvm::Function *OwnerFunc = BB->getParent();
        registerFunctionExit(OwnerFunc->getName(), *PrevNode);
        PrevNode->last_insn_interesting_ = true;
    } else {
        bool important = true;
        auto succs = llvm::succ_size(BB);
        if (llvm::dyn_cast<llvm::BranchInst>(TI)) {
            important = false;
        }
        if (important) {
            PrevNode->last_insn_interesting_ = true;
        }

        if (succs == 0) {
            PrevNode->specialType_ = CfgSpecialType::ErrorHandling;
        }

        auto edge_type = succs > 1 ? EdgeType::Branch : EdgeType::Sequence;

        auto sourceStep = PrevNode->evtSteps_ > 0 ? PrevNode->evtSteps_ : -1;
        for (llvm::BasicBlock *SuccBB : llvm::successors(BB)) {
            addEdge(*PrevNode, *ModuleNodes.at(SuccBB), edge_type, sourceStep);
        }
    }
}

void GlobalCFGBuilder::insertReturnEdges() {
    // Add edges from all collected origins (basic-blocks with ret terminator)
    for (const auto &KV : ReturnEdgeOrigins_) {
        llvm::StringRef FnName = KV.first();
        auto It = ReturnEdgeTargets_.find(FnName);
        if (It == ReturnEdgeTargets_.end()) {
            DOCC_DEBUG(llvm::dbgs() << "  No calls to " << FnName << "\n");
            continue;
        }
        for (auto [Origin] : KV.second) {
            int32_t retSourceStep = Origin->evtSteps_ > 0 ? Origin->evtSteps_ : -1;
            for (auto [Target, Fallback, retTargetStep] : It->second) {
                addEdge(*Origin, *Target, EdgeType::Return, retSourceStep, false, retTargetStep);
            }
        }
        ReturnEdgeTargets_.erase(It);
    }
    // For remaining targets, add edges from fallbacks (callee nodes)
    for (const auto &KV : ReturnEdgeTargets_) {
        llvm::StringRef FnName = KV.first();
        DOCC_DEBUG(llvm::dbgs() << "  No returns from " << FnName << "\n");
        for (auto [Target, Fallback, step] : KV.second) {
            addEdge(*Fallback, *Target, EdgeType::Return);
        }
    }
}

GlobalCFGNode &GlobalCFGBuilder::addNode(uint32_t modId, int64_t funcId) {
    auto V = boost::add_vertex(CFG_.Graph_);
    auto [It, Added] = CFG_.Nodes_.insert({V, std::make_unique<GlobalCFGNode>(V, modId, funcId)});
    assert(Added && "Vertex key is unique");
    auto &nd = *It->second;
    return nd;
}

void GlobalCFGAnalysis::addEntryPoint(llvm::GlobalValue::GUID Id, GlobalCFGNode *Node) {
    auto [ItEntrypoint, AddedEntrypoint] = EntryPoints_.insert({Id, Node});
    assert(AddedEntrypoint && "Duplicate entry-point");
}

GlobalCFGNode *GlobalCFGAnalysis::findNodeExternallyVisible(llvm::StringRef Name) const {
    auto It = EntryPoints_.find(llvm::GlobalValue::getGUID(Name));
    if (It == EntryPoints_.end()) return nullptr;
    return It->second;
}

// llvm::ErrorOr messages are useless, because they lack context:
// [docc_llvm_plugin] error: No such file or directory
static std::unique_ptr<llvm::MemoryBuffer> loadFile(llvm::StringRef Path) {
    auto FileOrErr = llvm::MemoryBuffer::getFile(Path);
    if (std::error_code EC = FileOrErr.getError()) {
        std::string File = Path.str();
        ExitOnErr(llvm::createStringError(llvm::inconvertibleErrorCode(), "No such file or directory: %s", File.c_str())
        );
        llvm_unreachable("Fatal error");
    }
    return std::move(*FileOrErr);
}

static bool isNullHash(const llvm::ModuleHash &Hash) {
    for (auto Component : Hash)
        if (Component != 0) return false;
    return true;
}

void GlobalCFGAnalysis::run(AnalysisManager &am) {
    auto &registry = am.get<analysis::SDFGRegistry>();

    auto CombinedIndex = registry.combined_index_.get();
    DOCC_DEBUG(CombinedIndex->print(llvm::dbgs()));

    DOCC_DEBUG(llvm::dbgs() << "\nBuilding global CFG:\n");
    GlobalCFGBuilder Builder(*this, *CombinedIndex, registry);

    for (auto &Entry : *CombinedIndex) {
        llvm::GlobalValue::GUID Id = Entry.first;
        llvm::GlobalValueSummary *S = CombinedIndex->getGlobalValueSummary(Id);
        if (S->getSummaryKind() == llvm::GlobalValueSummary::FunctionKind) {
            auto func_id = next_func_id_++;
            GlobalCFGNode &N = Builder.addNodeGlobal(getModuleId(S->modulePath()), func_id, Id);

            addEntryPoint(Id, &N);
        }
    }

    // attach known, global SDFGs to EntryNodes
    for (const auto &Entry : CombinedIndex->modulePaths()) {
        auto path = Entry.first();

        for (auto &[name, holder] : registry.at(path)) {
            auto guid = llvm::GlobalValue::getGUID(name);
            auto it = EntryPoints_.find(guid);
            if (it != EntryPoints_.end()) {
                it->second->sdfg_ = holder.get();
                it->second->Name_ = name;
            }
        }
    }

    //    DOCC_WAIT_FOR_DEBUGGER("GlobalCFGBuilder");

    for (const auto &Entry : CombinedIndex->modulePaths()) {
        if (isNullHash(Entry.second)) continue; // [Regular LTO] pseudo-module
        llvm::StringRef Path = Entry.first();
        std::unique_ptr<llvm::Module> Mod = registry.get_module(Path.str(), Ctx_);

        DOCC_DEBUG(llvm::dbgs() << "  Add module to global CFG: " << Mod->getName() << "\n");
        Builder.addModule(getModuleId(Path), *Mod);
        DOCC_DEBUG(llvm::dbgs() << "  Global nodes: " << Nodes_.size() << "\n");

        Modules_.push_back(std::move(Mod));
    }

    DOCC_DEBUG(llvm::dbgs() << "\nAdding return edges in global CFG:\n");
    Builder.insertReturnEdges();
}

uint32_t GlobalCFGAnalysis::getModuleId(llvm::StringRef modPath) {
    auto it = moduleIds_.find(modPath);
    if (it != moduleIds_.end()) {
        return it->second;
    } else {
        auto next = moduleIds_.size();
        moduleIds_[modPath] = next;
        return next;
    }
}

} // namespace docc::analysis
