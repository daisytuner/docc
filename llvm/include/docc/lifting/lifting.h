#pragma once

#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>

#include <sdfg/builder/sdfg_builder.h>

#include "docc/lifting/exceptions.h"
#include "docc/lifting/utils.h"
#include "docc/utils.h"

namespace docc {
namespace lifting {

/**
 * Lifting is the process of converting a LLVM function into a SDFG.
 */
class Lifting {
private:
    llvm::TargetLibraryInfo& TLI_;
    llvm::Function& function_;
    const llvm::DataLayout& DL_;

    sdfg::FunctionType target_type_;
    sdfg::builder::SDFGBuilder builder_;

    std::map<const llvm::BasicBlock*, std::set<const sdfg::control_flow::State*>> state_mapping_;
    std::map<const sdfg::control_flow::State*, std::set<const llvm::BasicBlock*>> pred_mapping_;

    std::unordered_map<const llvm::Value*, std::string> constants_mapping_;
    std::unordered_map<const llvm::Type*, std::string> anonymous_types_mapping_;

    static void collect_globals(llvm::Function& function, llvm::Value* V, std::unordered_set<llvm::GlobalObject*>& visited);

    void visit_globals();

    void visit_arguments();

    void visit_cfg();

    void visit_block(const llvm::BasicBlock* block, sdfg::control_flow::State& state);

    sdfg::control_flow::State& visit_instruction(
        const llvm::BasicBlock* block, const llvm::Instruction* instruction, sdfg::control_flow::State& current_state
    );

    sdfg::control_flow::State& visit_AllocaInst(
        const llvm::BasicBlock* block, const llvm::AllocaInst* instruction, sdfg::control_flow::State& current_state
    );

    sdfg::control_flow::State& visit_GetElementPtrInst(
        const llvm::BasicBlock* block,
        const llvm::GetElementPtrInst* instruction,
        sdfg::control_flow::State& current_state
    );

    sdfg::control_flow::State& visit_LoadInst(
        const llvm::BasicBlock* block, const llvm::LoadInst* instruction, sdfg::control_flow::State& current_state
    );

    sdfg::control_flow::State& visit_StoreInst(
        const llvm::BasicBlock* block, const llvm::StoreInst* instruction, sdfg::control_flow::State& current_state
    );

    sdfg::control_flow::State& visit_PHINode(
        const llvm::BasicBlock* block, const llvm::PHINode* instruction, sdfg::control_flow::State& current_state
    );

    sdfg::control_flow::State& visit_ICmpInst(
        const llvm::BasicBlock* block, const llvm::ICmpInst* instruction, sdfg::control_flow::State& current_state
    );

    sdfg::control_flow::State& visit_ICmpInst_symbolic(
        const llvm::BasicBlock* block, const llvm::ICmpInst* instruction, sdfg::control_flow::State& current_state
    );

    sdfg::control_flow::State& visit_ICmpInst_dataflow(
        const llvm::BasicBlock* block, const llvm::ICmpInst* instruction, sdfg::control_flow::State& current_state
    );

    sdfg::control_flow::State& visit_FCmpInst(
        const llvm::BasicBlock* block, const llvm::FCmpInst* instruction, sdfg::control_flow::State& current_state
    );

    sdfg::control_flow::State& visit_UnaryOperator(
        const llvm::BasicBlock* block, const llvm::UnaryOperator* instruction, sdfg::control_flow::State& current_state
    );

    sdfg::control_flow::State& visit_BinaryOperator(
        const llvm::BasicBlock* block, const llvm::BinaryOperator* instruction, sdfg::control_flow::State& current_state
    );

    sdfg::control_flow::State& visit_CastInst(
        const llvm::BasicBlock* block, const llvm::CastInst* instruction, sdfg::control_flow::State& current_state
    );

    sdfg::control_flow::State& visit_ReturnInst(
        const llvm::BasicBlock* block, const llvm::ReturnInst* instruction, sdfg::control_flow::State& current_state
    );

    sdfg::control_flow::State& visit_SelectInst(
        const llvm::BasicBlock* block, const llvm::SelectInst* instruction, sdfg::control_flow::State& current_state
    );

    sdfg::control_flow::State& visit_UnreachableInst(
        const llvm::BasicBlock* block,
        const llvm::UnreachableInst* instruction,
        sdfg::control_flow::State& current_state
    );

public:
    Lifting(
        llvm::TargetLibraryInfo& TLI,
        llvm::Function& function,
        sdfg::FunctionType target_type,
        const std::string& sdfg_name
    )
        : TLI_(TLI), function_(function), DL_(function.getParent()->getDataLayout()), target_type_(target_type),
          builder_(sdfg_name, target_type) {
        if (this->function_.isVarArg()) {
            throw NotImplementedException(
                "FunctionLifting: Vararg functions are not supported",
                docc::utils::get_debug_info(function),
                function.getName().str()
            );
        }

        auto return_type = lifting::utils::get_type(
            this->builder_,
            this->anonymous_types_mapping_,
            this->DL_,
            this->function_.getReturnType(),
            lifting::utils::get_storage_type(this->target_type_, 0)
        );
        this->builder_.set_return_type(*return_type);
    };

    Lifting(llvm::TargetLibraryInfo& TLI, llvm::Function& function, sdfg::FunctionType target_type)
        : Lifting(TLI, function, target_type, function.getName().str()) {

          };

    std::unique_ptr<sdfg::SDFG> run();

    static void collect_globals(llvm::Function& function, std::unordered_set<llvm::GlobalObject*>& globals);
};

} // namespace lifting
} // namespace docc
