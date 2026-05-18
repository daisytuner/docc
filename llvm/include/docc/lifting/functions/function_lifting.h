#pragma once

#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>

#include <sdfg/builder/sdfg_builder.h>

#include "docc/lifting/exceptions.h"
#include "docc/lifting/utils.h"
#include "docc/utils.h"

namespace docc {
namespace lifting {

class FunctionLifting {
protected:
    llvm::TargetLibraryInfo& TLI_;
    const llvm::DataLayout& DL_;

    const llvm::Function& function_;

    sdfg::FunctionType target_type_;
    sdfg::builder::SDFGBuilder& builder_;

    std::map<const llvm::BasicBlock*, std::set<const sdfg::control_flow::State*>>& state_mapping_;
    std::map<const sdfg::control_flow::State*, std::set<const llvm::BasicBlock*>>& pred_mapping_;

    std::unordered_map<const llvm::Value*, std::string>& constants_mapping_;
    std::unordered_map<const llvm::Type*, std::string>& anonymous_types_mapping_;

    sdfg::control_flow::State& visit_call(
        const llvm::BasicBlock* block, const llvm::CallInst* instruction, sdfg::control_flow::State& current_state
    );

    sdfg::control_flow::State& visit_invoke(
        const llvm::BasicBlock* block, const llvm::InvokeInst* instruction, sdfg::control_flow::State& current_state
    );

public:
    FunctionLifting(
        llvm::TargetLibraryInfo& TLI,
        const llvm::DataLayout& DL,
        const llvm::Function& function,
        sdfg::FunctionType target_type,
        sdfg::builder::SDFGBuilder& builder,
        std::map<const llvm::BasicBlock*, std::set<const sdfg::control_flow::State*>>& state_mapping,
        std::map<const sdfg::control_flow::State*, std::set<const llvm::BasicBlock*>>& pred_mapping,
        std::unordered_map<const llvm::Value*, std::string>& constants_mapping,
        std::unordered_map<const llvm::Type*, std::string>& anonymous_types_mapping
    )
        : TLI_(TLI), DL_(DL), function_(function), target_type_(target_type), builder_(builder),
          state_mapping_(state_mapping), pred_mapping_(pred_mapping), constants_mapping_(constants_mapping),
          anonymous_types_mapping_(anonymous_types_mapping) {};

    virtual ~FunctionLifting() = default;

    virtual sdfg::control_flow::State&
    visit(const llvm::BasicBlock* block, const llvm::CallBase* instruction, sdfg::control_flow::State& current_state);

    static bool is_supported(llvm::TargetLibraryInfo& TLI, const llvm::CallBase* instruction);
};

} // namespace lifting
} // namespace docc
