#include "docc/lifting/lifting.h"

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/CFG.h>
#include <llvm/IR/Constant.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/GlobalObject.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/Support/Casting.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>

#include <cassert>
#include <memory>

#include "docc/lifting/functions/function_lifting.h"
#include "docc/utils.h"

#include <sdfg/data_flow/library_nodes/stdlib/stdlib.h>

namespace docc {
namespace lifting {

std::unique_ptr<sdfg::SDFG> Lifting::run() {
    // Add globals to SDFG
    this->visit_globals();

    // Add arguments to SDFG
    this->visit_arguments();

    // Add CFG to SDFG
    this->visit_cfg();

    // Visit blocks to add instructions to SDFG
    std::list<sdfg::control_flow::State*> visited;
    std::list<sdfg::control_flow::State*> queue = {
        this->builder_.get_non_const_state(&this->builder_.subject().start_state())
    };
    while (!queue.empty()) {
        sdfg::control_flow::State* state = queue.front();
        queue.pop_front();
        if (std::find(visited.begin(), visited.end(), state) != visited.end()) {
            continue;
        }
        visited.push_back(state);

        const llvm::BasicBlock* block = nullptr;
        for (auto& state_mapping_kv : this->state_mapping_) {
            if (state_mapping_kv.second.find(state) != state_mapping_kv.second.end()) {
                assert(block == nullptr);
                block = state_mapping_kv.first;
            }
        }
        if (block != nullptr) {
            this->visit_block(block, *state);
        }

        for (auto& neighbor : this->builder_.subject().out_edges(*state)) {
            queue.push_back(this->builder_.get_non_const_state(&neighbor.dst()));
        }
    }

    return builder_.move();
};

void Lifting::visit_globals() {
    std::unordered_set<llvm::GlobalObject*> visited;
    Lifting::collect_globals(this->function_, visited);

    for (llvm::GlobalObject* GV : visited) {
        if (auto function = llvm::dyn_cast<llvm::Function>(GV)) {
            if (function->isIntrinsic()) {
                continue;
            }
        }

        // Add the global to the SDFG
        std::string global = GV->getName().str();
        if (global == "llvm.used") {
            continue;
        }

        // Make internal/private globals unique by hashing module name
        // additionally, sanitize name for C/C++ compatibility
        if (GV->getLinkage() == llvm::GlobalValue::InternalLinkage) {
            if (!GV->getName().starts_with("__daisy_int_")) {
                // Hash module name and append to global name
                std::string module_path = GV->getParent()->getName().str();
                std::string module_name = std::filesystem::path(module_path).stem().string();
                global = "__daisy_int_" + utils::normalize_name(module_name) + "_" + utils::normalize_name(global);
                GV->setName(global);
                GV->setLinkage(llvm::GlobalValue::ExternalLinkage);
            }
        } else if (GV->getLinkage() == llvm::GlobalValue::PrivateLinkage) {
            if (!GV->getName().starts_with("__daisy_priv_")) {
                // Hash module name and append to global name
                std::string module_path = GV->getParent()->getName().str();
                std::string module_name = std::filesystem::path(module_path).stem().string();
                global = "__daisy_priv_" + utils::normalize_name(module_name) + "_" + utils::normalize_name(global);
                GV->setName(global);
                GV->setLinkage(llvm::GlobalValue::ExternalLinkage);
            }
        }

        // Globals are pointers but their type is the base type
        std::unique_ptr<sdfg::types::IType> global_type;
        if (llvm::GlobalVariable* GVar = llvm::dyn_cast<llvm::GlobalVariable>(GV)) {
            auto base_type = utils::get_type(
                this->builder_,
                this->anonymous_types_mapping_,
                this->DL_,
                GV->getValueType(),
                utils::get_storage_type(this->target_type_, 0),
                ""
            );
            global_type = std::make_unique<sdfg::types::Pointer>(
                base_type->storage_type(),
                0,
                base_type->initializer(),
                static_cast<const sdfg::types::IType&>(*base_type)
            );
        } else {
            global_type = utils::get_type(
                this->builder_,
                this->anonymous_types_mapping_,
                this->DL_,
                GV->getValueType(),
                utils::get_storage_type(this->target_type_, 0)
            );
        }

        switch (GV->getLinkage()) {
            case llvm::GlobalValue::LinkageTypes::ExternalLinkage:
            case llvm::GlobalValue::LinkageTypes::AvailableExternallyLinkage:
            case llvm::GlobalValue::LinkageTypes::LinkOnceAnyLinkage:
            case llvm::GlobalValue::LinkageTypes::LinkOnceODRLinkage:
            case llvm::GlobalValue::LinkageTypes::WeakAnyLinkage:
            case llvm::GlobalValue::LinkageTypes::WeakODRLinkage:
            case llvm::GlobalValue::LinkageTypes::InternalLinkage:
            case llvm::GlobalValue::LinkageTypes::PrivateLinkage: {
                this->builder_.add_external(global, *global_type, sdfg::LinkageType_External);
                break;
            }
            default: {
                throw NotImplementedException(
                    "Unsupported linkage type: " + std::to_string(GV->getLinkage()),
                    docc::utils::bestEffortLoc(*GV),
                    ::docc::utils::toIRString(*GV)
                );
            }
        }

        // Make sure the global survives later LLVM optimizations
        llvm::appendToUsed(*this->function_.getParent(), {GV});
    }
}

void Lifting::collect_globals(llvm::Function& function, std::unordered_set<llvm::GlobalObject*>& globals) {
    for (llvm::Instruction& I : llvm::instructions(function)) {
        for (llvm::Use& U : I.operands()) Lifting::collect_globals(function, U.get(), globals);
    }
}

void Lifting::collect_globals(llvm::Function& function, llvm::Value* V, std::unordered_set<llvm::GlobalObject*>& visited) {
    // Drop pointer-casts / addrspace-casts / zero-GEPs.
    V = V->stripPointerCasts();

    // If it is a ConstantExpr (GEP, bitcast, inttoptr, …) look at its operands.
    if (auto* CE = llvm::dyn_cast<llvm::ConstantExpr>(V)) {
        for (llvm::Value* Op : CE->operands()) Lifting::collect_globals(function, Op, visited);
        return;
    }

    // Aggregate constants (arrays, structs) can also hide ConstantExprs.
    if (auto* CA = llvm::dyn_cast<llvm::ConstantAggregate>(V)) {
        for (llvm::Value* Op : CA->operands()) Lifting::collect_globals(function, Op, visited);
        return;
    }

    // Finally, if we have hit a global, remember it (deduplicated with visited).
    if (auto* GV = llvm::dyn_cast<llvm::GlobalObject>(V)) {
        if (visited.find(GV) == visited.end()) {
            visited.insert(GV);
        }
    }
}

void Lifting::visit_arguments() {
    for (auto& llvm_arg : this->function_.args()) {
        std::string arg = utils::get_name(&llvm_arg);
        auto arg_type = utils::get_type(
            this->builder_,
            this->anonymous_types_mapping_,
            this->DL_,
            llvm_arg.getType(),
            utils::get_storage_type(this->target_type_, 0)
        );
        this->builder_.add_container(arg, *arg_type, true);
    }
};

void Lifting::visit_cfg() {
    // Add states for each block and PHI transitions
    for (const llvm::BasicBlock& block : this->function_) {
        const llvm::Instruction& initial_inst = *block.begin();
        this->state_mapping_.insert({&block, {}});

        if (&block == &this->function_.getEntryBlock()) {
            assert(!block.hasNPredecessorsOrMore(1));

            auto& state = this->builder_.add_state(true, ::docc::utils::get_debug_info(initial_inst));
            this->state_mapping_.at(&block).insert(&state);
            this->pred_mapping_.insert({&state, {}});
        } else if (block.phis().empty()) {
            auto& state = this->builder_.add_state(false, ::docc::utils::get_debug_info(initial_inst));
            auto preds = llvm::predecessors(&block);
            std::set<const llvm::BasicBlock*> pred_set(preds.begin(), preds.end());
            this->state_mapping_.at(&block).insert(&state);
            this->pred_mapping_.insert({&state, pred_set});
        } else {
            for (auto PI = llvm::pred_begin(&block), E = llvm::pred_end(&block); PI != E; ++PI) {
                const llvm::BasicBlock* pred = *PI;

                auto& state = this->builder_.add_state(false, ::docc::utils::get_debug_info(initial_inst));

                this->state_mapping_.at(&block).insert(&state);
                this->pred_mapping_.insert({&state, {pred}});
            }
        }
    }

    // Add edges for each <pred, succ> pair
    for (const llvm::BasicBlock& pred : this->function_) {
        const llvm::Instruction* terminator = pred.getTerminator();
        if (llvm::isa<llvm::UnreachableInst>(terminator) || llvm::isa<llvm::ReturnInst>(terminator)) {
            continue;
        }

        if (auto br_inst = llvm::dyn_cast<const llvm::BranchInst>(terminator)) {
            auto dbg_info = ::docc::utils::get_debug_info(*br_inst);

            if (br_inst->isUnconditional()) {
                auto succ = br_inst->getSuccessor(0);
                auto dst_states = this->state_mapping_.at(succ);

                const sdfg::control_flow::State* dst_state = nullptr;
                for (auto dst_state_cand : dst_states) {
                    if (this->pred_mapping_.at(dst_state_cand).contains(&pred)) {
                        assert(dst_state == nullptr);
                        dst_state = dst_state_cand;
                    }
                }
                assert(dst_state != nullptr);

                for (auto src_state : this->state_mapping_.at(&pred)) {
                    this->builder_.add_edge(*src_state, *dst_state, dbg_info);
                }
            } else {
                auto condition = br_inst->getCondition();

                sdfg::symbolic::Expression expression;
                if (utils::is_literal(condition)) {
                    expression = utils::as_symbol(llvm::dyn_cast<llvm::Constant>(condition));
                } else {
                    auto symbol = utils::find_const_name_to_sdfg_name(this->constants_mapping_, condition);
                    if (!this->builder_.subject().exists(symbol)) {
                        auto condition_type = utils::get_type(
                            this->builder_,
                            this->anonymous_types_mapping_,
                            this->DL_,
                            condition->getType(),
                            utils::get_storage_type(this->target_type_, 0)
                        );
                        this->builder_.add_container(symbol, *condition_type);

                        auto& cond_type = this->builder_.subject().type(symbol);
                        assert(
                            cond_type.type_id() == sdfg::types::TypeID::Scalar &&
                            "BranchInst: Expected scalar type as condition"
                        );
                        auto& cond_scalar_type = static_cast<const sdfg::types::Scalar&>(cond_type);
                        assert(
                            cond_scalar_type.primitive_type() == sdfg::types::PrimitiveType::Bool &&
                            "BranchInst: Expected bool type as condition"
                        );
                    }
                    expression = sdfg::symbolic::symbol(symbol);
                }

                auto if_block = br_inst->getSuccessor(0);
                auto if_dst_states = this->state_mapping_.at(if_block);

                const sdfg::control_flow::State* if_dst_state = nullptr;
                for (auto if_dst_state_cand : if_dst_states) {
                    if (this->pred_mapping_.at(if_dst_state_cand).contains(&pred)) {
                        assert(if_dst_state == nullptr);
                        if_dst_state = if_dst_state_cand;
                    }
                }
                assert(if_dst_state != nullptr);

                auto else_block = br_inst->getSuccessor(1);
                auto else_dst_states = this->state_mapping_.at(else_block);

                const sdfg::control_flow::State* else_dst_state = nullptr;
                for (auto else_dst_state_cand : else_dst_states) {
                    if (this->pred_mapping_.at(else_dst_state_cand).contains(&pred)) {
                        assert(else_dst_state == nullptr);
                        else_dst_state = else_dst_state_cand;
                    }
                }
                assert(else_dst_state != nullptr);

                for (auto src_state : this->state_mapping_.at(&pred)) {
                    auto sdfg_cond = sdfg::symbolic::Ne(expression, sdfg::symbolic::__false__());
                    this->builder_.add_edge(*src_state, *if_dst_state, sdfg_cond, dbg_info);
                    this->builder_.add_edge(*src_state, *else_dst_state, sdfg::symbolic::Not(sdfg_cond), dbg_info);
                }
            }
        } else if (auto invoke_inst = llvm::dyn_cast<const llvm::InvokeInst>(terminator)) {
            auto dbg_info = ::docc::utils::get_debug_info(*invoke_inst);

            auto normal_block = invoke_inst->getNormalDest();
            auto normal_dst_states = this->state_mapping_.at(normal_block);

            const sdfg::control_flow::State* normal_dst_state = nullptr;
            for (auto normal_dst_state_cand : normal_dst_states) {
                if (this->pred_mapping_.at(normal_dst_state_cand).contains(&pred)) {
                    assert(normal_dst_state == nullptr);
                    normal_dst_state = normal_dst_state_cand;
                }
            }
            assert(normal_dst_state != nullptr);

            auto unwind_block = invoke_inst->getUnwindDest();
            auto unwind_dst_states = this->state_mapping_.at(unwind_block);

            const sdfg::control_flow::State* unwind_dst_state = nullptr;
            for (auto unwind_dst_state_cand : unwind_dst_states) {
                if (this->pred_mapping_.at(unwind_dst_state_cand).contains(&pred)) {
                    assert(unwind_dst_state == nullptr);
                    unwind_dst_state = unwind_dst_state_cand;
                }
            }
            assert(unwind_dst_state != nullptr);

            // Add unwind symbol
            sdfg::types::Scalar unwind_type(sdfg::types::PrimitiveType::Bool);
            std::string unwind_symbol = "__unwind_" + utils::get_name(invoke_inst);
            this->builder_.add_container(unwind_symbol, unwind_type);

            auto unwind_cond = sdfg::symbolic::Ne(sdfg::symbolic::symbol(unwind_symbol), sdfg::symbolic::__false__());
            for (auto src_state : this->state_mapping_.at(&pred)) {
                this->builder_.add_edge(*src_state, *normal_dst_state, sdfg::symbolic::Not(unwind_cond), dbg_info);
                this->builder_.add_edge(*src_state, *unwind_dst_state, unwind_cond, dbg_info);
            }
        } else {
            throw NotImplementedException(
                "Unsupported terminator instruction in CFG construction",
                docc::utils::get_debug_info(*terminator),
                docc::utils::toIRString(*terminator)
            );
        }
    }
}

void Lifting::visit_block(const llvm::BasicBlock* block, sdfg::control_flow::State& state) {
    std::set<const llvm::BasicBlock*> preds = this->pred_mapping_.at(&state);

    // Phi Nodes perform double buffering. We need to create per-transition containers.
    std::string transition_suffix = utils::get_name(block);
    for (auto& pred : preds) {
        transition_suffix += "_" + utils::get_name(pred);
    }

    // We add states before the actual instruction to ensure that the correct phi values are available for the
    // instruction.
    sdfg::control_flow::State* next_state = &state;
    for (const llvm::Instruction& inst : *block) {
        auto phi_inst = llvm::dyn_cast<const llvm::PHINode>(&inst);
        if (!phi_inst) {
            break;
        }

        // Determine phi value
        llvm::Value* phi_value = nullptr;
        for (size_t i = 0; i < phi_inst->getNumIncomingValues(); ++i) {
            if (preds.contains(phi_inst->getIncomingBlock(i))) {
                assert(phi_value == nullptr && "Failed to determine phi value");
                phi_value = phi_inst->getIncomingValue(i);
            }
        }
        assert(phi_value != nullptr && "Failed to determine phi value");

        // Constants can be handled without a buffer
        if (llvm::isa<llvm::Constant>(phi_value) || llvm::isa<llvm::Argument>(phi_value) ||
            llvm::isa<llvm::GlobalValue>(phi_value)) {
            continue;
        }

        // Add state
        auto dbg_info = ::docc::utils::get_debug_info(*phi_inst);
        next_state = &this->builder_.add_state_after(*next_state, true, dbg_info);
        this->pred_mapping_.insert({next_state, preds});

        // Map Phi value to transition-specific value
        auto phi_value_type = utils::get_type(
            this->builder_,
            this->anonymous_types_mapping_,
            this->DL_,
            phi_value->getType(),
            utils::get_storage_type(this->target_type_, 0)
        );
        std::string phi_input = utils::get_name(phi_value);
        if (!this->builder_.subject().exists(phi_input)) {
            this->builder_.add_container(phi_input, *phi_value_type);
        }
        auto& input_node = this->builder_.add_access(*next_state, phi_input, dbg_info);

        std::string phi_transition = phi_input + "_" + transition_suffix;
        if (!this->builder_.subject().exists(phi_transition)) {
            this->builder_.add_container(phi_transition, *phi_value_type);
        }
        auto& output_node = this->builder_.add_access(*next_state, phi_transition, dbg_info);

        switch (phi_value_type->type_id()) {
            case sdfg::types::TypeID::Pointer: {
                sdfg::types::Pointer base_ptr_type;
                sdfg::types::Pointer ptr_type(static_cast<const sdfg::types::IType&>(base_ptr_type));
                this->builder_.add_reference_memlet(
                    *next_state, input_node, output_node, {sdfg::symbolic::zero()}, ptr_type, dbg_info
                );
                break;
            }
            case sdfg::types::TypeID::Scalar: {
                auto& tasklet =
                    this->builder_
                        .add_tasklet(*next_state, sdfg::data_flow::TaskletCode::assign, "__out", {"_in"}, dbg_info);
                this->builder_.add_computational_memlet(*next_state, input_node, tasklet, "_in", {}, dbg_info);
                this->builder_.add_computational_memlet(*next_state, tasklet, "__out", output_node, {}, dbg_info);
                break;
            }
            default: {
                throw NotImplementedException(
                    "Unsupported phi value type",
                    ::docc::utils::get_debug_info(*phi_inst),
                    ::docc::utils::toIRString(*phi_inst)
                );
            }
        }
    }

    // Visit instructions
    for (const llvm::Instruction& inst : *block) {
        if (llvm::isa<llvm::BranchInst>(inst)) {
            // Branch instructions are handled in the CFG construction.
            continue;
        }

        auto dbg_info = ::docc::utils::get_debug_info(inst);
        next_state = &this->builder_.add_state_after(*next_state, true, dbg_info);
        this->pred_mapping_.insert({next_state, preds});
        next_state = &this->visit_instruction(block, &inst, *next_state);
    }
};

sdfg::control_flow::State& Lifting::visit_instruction(
    const llvm::BasicBlock* block, const llvm::Instruction* instruction, sdfg::control_flow::State& current_state
) {
    // Constant expressions are converted into states before the instruction is visited.
    // ConstantExprVisitor constantexpr_visitor(this->TLI_, this->function_, this->DL_, this->builder_,
    //                                          this->target_type_, this->state_mapping_,
    //                                          this->pred_mapping_, this->constants_mapping_);
    // for (auto& op : instruction->operands()) {
    //     if (llvm::ConstantExpr* CE = llvm::dyn_cast<llvm::ConstantExpr>(op)) {
    //         constantexpr_visitor.visit(block, CE, current_state, instruction);
    //     }
    // }

    // Special instructions
    if (auto inst = llvm::dyn_cast<const llvm::AllocaInst>(instruction)) {
        return this->visit_AllocaInst(block, inst, current_state);
    } else if (auto inst = llvm::dyn_cast<const llvm::GetElementPtrInst>(instruction)) {
        return this->visit_GetElementPtrInst(block, inst, current_state);
    } else if (auto inst = llvm::dyn_cast<const llvm::StoreInst>(instruction)) {
        return this->visit_StoreInst(block, inst, current_state);
    } else if (auto inst = llvm::dyn_cast<const llvm::LoadInst>(instruction)) {
        return this->visit_LoadInst(block, inst, current_state);
    } else if (auto inst = llvm::dyn_cast<const llvm::PHINode>(instruction)) {
        return this->visit_PHINode(block, inst, current_state);
    } else if (auto inst = llvm::dyn_cast<const llvm::ICmpInst>(instruction)) {
        return this->visit_ICmpInst(block, inst, current_state);
    } else if (auto inst = llvm::dyn_cast<const llvm::ReturnInst>(instruction)) {
        return this->visit_ReturnInst(block, inst, current_state);
    } else if (auto inst = llvm::dyn_cast<const llvm::SelectInst>(instruction)) {
        return this->visit_SelectInst(block, inst, current_state);
    } else if (auto inst = llvm::dyn_cast<const llvm::UnreachableInst>(instruction)) {
        return this->visit_UnreachableInst(block, inst, current_state);
    }

    // Operations
    if (auto inst = llvm::dyn_cast<const llvm::UnaryOperator>(instruction)) {
        return this->visit_UnaryOperator(block, inst, current_state);
    } else if (auto inst = llvm::dyn_cast<const llvm::BinaryOperator>(instruction)) {
        return this->visit_BinaryOperator(block, inst, current_state);
    } else if (auto inst = llvm::dyn_cast<const llvm::CastInst>(instruction)) {
        return this->visit_CastInst(block, inst, current_state);
    } else if (auto inst = llvm::dyn_cast<const llvm::FCmpInst>(instruction)) {
        return this->visit_FCmpInst(block, inst, current_state);
    } else if (auto inst = llvm::dyn_cast<const llvm::CallBase>(instruction)) {
        FunctionLifting lifter(
            this->TLI_,
            this->DL_,
            this->function_,
            this->target_type_,
            this->builder_,
            this->state_mapping_,
            this->pred_mapping_,
            this->constants_mapping_,
            this->anonymous_types_mapping_
        );
        return lifter.visit(block, inst, current_state);
    }

    std::string instruction_str;
    llvm::raw_string_ostream OS(instruction_str);
    instruction->print(OS);
    throw NotImplementedException(
        "Unsupported instruction: " + instruction_str,
        ::docc::utils::get_debug_info(*instruction),
        ::docc::utils::toIRString(*instruction)
    );
};

sdfg::control_flow::State& Lifting::visit_ReturnInst(
    const llvm::BasicBlock* block, const llvm::ReturnInst* instruction, sdfg::control_flow::State& current_state
) {
    auto dbg_info = ::docc::utils::get_debug_info(*instruction);
    if (instruction->getNumOperands() == 0) {
        // Do nothing for void return
        assert(this->builder_.subject().out_degree(current_state) == 0);
        return this->builder_.add_return_state_after(current_state, "", dbg_info);
    }

    assert(instruction->getNumOperands() == 1);

    // Non-void return
    auto return_value = instruction->getReturnValue();
    assert(return_value != nullptr);

    assert(this->builder_.subject().out_degree(current_state) == 0);

    std::string data;
    if (utils::is_literal(return_value)) {
        data = utils::as_initializer(llvm::dyn_cast<llvm::Constant>(return_value));
        auto return_type = utils::get_type(
            this->builder_,
            this->anonymous_types_mapping_,
            this->DL_,
            return_value->getType(),
            utils::get_storage_type(this->target_type_, 0)
        );
        return this->builder_.add_constant_return_state_after(current_state, data, *return_type, dbg_info);
    } else {
        data = utils::find_const_name_to_sdfg_name(this->constants_mapping_, return_value);
        return this->builder_.add_return_state_after(current_state, data, dbg_info);
    }
};

sdfg::control_flow::State& Lifting::visit_UnreachableInst(
    const llvm::BasicBlock* block, const llvm::UnreachableInst* instruction, sdfg::control_flow::State& current_state
) {
    // Unreachable instructions do not have any effect on the SDFG.
    // We simply return the current state.
    auto dbg_info = ::docc::utils::get_debug_info(*instruction);
    assert(this->builder_.subject().out_degree(current_state) == 0);

    std::string data;
    llvm::Type* RetTy = this->function_.getReturnType();
    if (RetTy->isVoidTy()) {
        data = "";
    } else {
        auto undef = llvm::UndefValue::get(RetTy);
        data = utils::as_initializer(undef);
    }

    auto& ret_state =
        this->builder_
            .add_constant_return_state_after(current_state, data, this->builder_.subject().return_type(), dbg_info);
    this->builder_.add_library_node<sdfg::stdlib::UnreachableNode>(ret_state, dbg_info);

    return ret_state;
};

sdfg::control_flow::State& Lifting::visit_AllocaInst(
    const llvm::BasicBlock* block, const llvm::AllocaInst* instruction, sdfg::control_flow::State& current_state
) {
    // Define Debug
    auto dbg_info = ::docc::utils::get_debug_info(*instruction);

    // Define Output
    std::string output = utils::get_name(instruction);
    if (!this->builder_.subject().exists(output)) {
        auto output_type = utils::get_type(
            this->builder_,
            this->anonymous_types_mapping_,
            this->DL_,
            instruction->getType(),
            utils::get_storage_type(this->target_type_, 0)
        );
        this->builder_.add_container(output, *output_type);
    }
    auto& output_type = this->builder_.subject().type(output);
    assert(output_type.type_id() == sdfg::types::TypeID::Pointer && "AllocaInst: Expected pointer type as output");

    auto alloca_type = utils::get_type(
        this->builder_,
        this->anonymous_types_mapping_,
        this->DL_,
        instruction->getAllocatedType(),
        utils::get_storage_type(this->target_type_, 0)
    );

    // Define array size
    sdfg::symbolic::Expression size_sym = sdfg::symbolic::one();
    if (instruction->isArrayAllocation()) {
        const llvm::Value* arg_size = instruction->getArraySize();
        if (utils::is_literal(arg_size)) {
            size_sym = utils::as_symbol(llvm::dyn_cast<const llvm::ConstantData>(arg_size));
        } else {
            std::string arg_name = utils::find_const_name_to_sdfg_name(this->constants_mapping_, arg_size);
            size_sym = sdfg::symbolic::symbol(arg_name);
        }
    }
    size_t type_size = this->DL_.getTypeAllocSize(instruction->getAllocatedType());
    sdfg::symbolic::Expression alloca_size = sdfg::symbolic::mul(size_sym, sdfg::symbolic::integer(type_size));

    auto& output_node = this->builder_.add_access(current_state, output, dbg_info);
    auto& lib_node = this->builder_.add_library_node<sdfg::stdlib::AllocaNode>(current_state, dbg_info, alloca_size);

    sdfg::types::Pointer opaque_ptr;
    this->builder_.add_computational_memlet(current_state, lib_node, "_ret", output_node, {}, opaque_ptr, dbg_info);

    return current_state;
};

sdfg::control_flow::State& Lifting::visit_GetElementPtrInst(
    const llvm::BasicBlock* block, const llvm::GetElementPtrInst* instruction, sdfg::control_flow::State& current_state
) {
    // Define Debug
    auto dbg_info = ::docc::utils::get_debug_info(*instruction);

    // Define Output
    std::string output = utils::get_name(instruction);
    if (!this->builder_.subject().exists(output)) {
        auto output_type = utils::get_type(
            this->builder_,
            this->anonymous_types_mapping_,
            this->DL_,
            instruction->getType(),
            utils::get_storage_type(this->target_type_, 0)
        );
        this->builder_.add_container(output, *output_type);
    }
    auto& output_type = this->builder_.subject().type(output);
    assert(output_type.type_id() == sdfg::types::TypeID::Pointer && "GetElementPtrInst: Expected pointer type as output");

    // Define Input
    auto pointer_operand = instruction->getPointerOperand();
    std::string input;
    if (utils::is_null_pointer(pointer_operand)) {
        input = sdfg::symbolic::__nullptr__()->get_name();
    } else {
        input = utils::find_const_name_to_sdfg_name(this->constants_mapping_, pointer_operand);
        auto& input_type = this->builder_.subject().type(input);
        assert(
            input_type.type_id() == sdfg::types::TypeID::Pointer && "GetElementPtrInst: Expected pointer type as input"
        );
    }

    // Subset
    sdfg::data_flow::Subset subset;
    for (auto it = instruction->idx_begin(); it != instruction->idx_end(); ++it) {
        if (auto const_int = llvm::dyn_cast<llvm::ConstantInt>(*it)) {
            auto start = sdfg::symbolic::integer(const_int->getZExtValue());
            subset.push_back(start);
        } else {
            std::string name = utils::get_name(*it);
            auto start = sdfg::symbolic::symbol(name);
            assert(
                this->builder_.subject().exists(name) &&
                "GetElementPtrInst: Expected index to be a constant or a symbol"
            );
            subset.push_back(start);
        }
    }

    // Define Source Element Type
    auto source_element_type = utils::get_type(
        this->builder_,
        this->anonymous_types_mapping_,
        this->DL_,
        instruction->getSourceElementType(),
        utils::get_storage_type(this->target_type_, 0)
    );
    sdfg::types::Pointer base_ptr_type(*source_element_type);

    auto& output_node = this->builder_.add_access(current_state, output, dbg_info);
    if (utils::is_null_pointer(pointer_operand)) {
        auto& input_node = this->builder_.add_constant(current_state, input, sdfg::types::Pointer(), dbg_info);
        this->builder_.add_reference_memlet(current_state, input_node, output_node, subset, base_ptr_type, dbg_info);
    } else {
        auto& input_node = this->builder_.add_access(current_state, input, dbg_info);
        this->builder_.add_reference_memlet(current_state, input_node, output_node, subset, base_ptr_type, dbg_info);
    }

    return current_state;
};

sdfg::control_flow::State& Lifting::visit_LoadInst(
    const llvm::BasicBlock* block, const llvm::LoadInst* instruction, sdfg::control_flow::State& current_state
) {
    // Define Debug
    auto dbg_info = ::docc::utils::get_debug_info(*instruction);

    // Define Output
    std::string output = utils::get_name(instruction);
    if (!this->builder_.subject().exists(output)) {
        auto output_type = utils::get_type(
            this->builder_,
            this->anonymous_types_mapping_,
            this->DL_,
            instruction->getType(),
            utils::get_storage_type(this->target_type_, 0)
        );
        this->builder_.add_container(output, *output_type);
    }
    auto& output_type = this->builder_.subject().type(output);

    // Define Input
    auto pointer_operand = instruction->getPointerOperand();
    assert(!utils::is_null_pointer(pointer_operand) && "LoadInst: Expected non-null pointer as source");
    std::string input = utils::find_const_name_to_sdfg_name(this->constants_mapping_, pointer_operand);
    assert(this->builder_.subject().exists(input));
    auto& input_type = this->builder_.subject().type(input);
    assert(input_type.type_id() == sdfg::types::TypeID::Pointer && "LoadInst: Expected pointer type as input");

    // Define Operation
    switch (output_type.type_id()) {
        case sdfg::types::TypeID::Scalar: {
            auto& output_node = this->builder_.add_access(current_state, output, dbg_info);
            auto& input_node = this->builder_.add_access(current_state, input, dbg_info);
            auto& tasklet =
                this->builder_
                    .add_tasklet(current_state, sdfg::data_flow::TaskletCode::assign, "__out", {"_in"}, dbg_info);

            sdfg::types::Pointer base_ptr_type(output_type);
            this->builder_.add_computational_memlet(
                current_state, input_node, tasklet, "_in", {sdfg::symbolic::zero()}, base_ptr_type, dbg_info
            );
            this->builder_.add_computational_memlet(current_state, tasklet, "__out", output_node, {}, dbg_info);

            return current_state;
        }
        case sdfg::types::TypeID::Pointer:
        case sdfg::types::TypeID::Array:
        case sdfg::types::TypeID::Structure: {
            auto& output_node = this->builder_.add_access(current_state, output, dbg_info);
            auto& input_node = this->builder_.add_access(current_state, input, dbg_info);
            sdfg::types::Pointer input_ptr_type(output_type);
            this->builder_
                .add_dereference_memlet(current_state, input_node, output_node, true, input_ptr_type, dbg_info);

            return current_state;
        }
        default:
            throw NotImplementedException(
                "Unsupported load instruction",
                ::docc::utils::get_debug_info(*instruction),
                ::docc::utils::toIRString(*instruction)
            );
    }
};

sdfg::control_flow::State& Lifting::visit_StoreInst(
    const llvm::BasicBlock* block, const llvm::StoreInst* instruction, sdfg::control_flow::State& current_state
) {
    // Define Debug
    auto dbg_info = ::docc::utils::get_debug_info(*instruction);

    // Define Output
    auto output_operand = instruction->getPointerOperand();
    assert(!utils::is_null_pointer(output_operand) && "StoreInst: Expected non-null pointer as source");
    std::string output = utils::find_const_name_to_sdfg_name(this->constants_mapping_, output_operand);
    assert(this->builder_.subject().exists(output));
    auto& output_type = this->builder_.subject().type(output);
    assert(output_type.type_id() == sdfg::types::TypeID::Pointer && "StoreInst: Expected pointer type as output");

    // Define Input
    auto input_operand = instruction->getValueOperand();
    auto input_type = utils::get_type(
        this->builder_,
        this->anonymous_types_mapping_,
        this->DL_,
        input_operand->getType(),
        utils::get_storage_type(this->target_type_, 0)
    );
    // Define Operation
    switch (input_type->type_id()) {
        case sdfg::types::TypeID::Scalar: {
            auto& output_node = this->builder_.add_access(current_state, output, dbg_info);

            sdfg::data_flow::AccessNode* input_node;
            if (utils::is_literal(input_operand)) {
                std::string arg = utils::as_literal(llvm::dyn_cast<llvm::ConstantData>(input_operand));
                input_node = &this->builder_.add_constant(current_state, arg, *input_type, dbg_info);
            } else {
                std::string input = utils::find_const_name_to_sdfg_name(this->constants_mapping_, input_operand);
                input_node = &this->builder_.add_access(current_state, input, dbg_info);
            }
            auto& tasklet =
                this->builder_
                    .add_tasklet(current_state, sdfg::data_flow::TaskletCode::assign, "__out", {"_in"}, dbg_info);

            sdfg::types::Pointer base_ptr_type(*input_type);
            this->builder_.add_computational_memlet(current_state, *input_node, tasklet, "_in", {}, dbg_info);
            this->builder_.add_computational_memlet(
                current_state, tasklet, "__out", output_node, {sdfg::symbolic::zero()}, base_ptr_type, dbg_info
            );

            return current_state;
        }
        case sdfg::types::TypeID::Array:
        case sdfg::types::TypeID::Structure:
        case sdfg::types::TypeID::Pointer: {
            auto& output_node = this->builder_.add_access(current_state, output, dbg_info);

            sdfg::data_flow::AccessNode* input_node;
            if (utils::is_literal(input_operand)) {
                std::string input = utils::as_initializer(llvm::dyn_cast<llvm::Constant>(input_operand));
                input_node = &this->builder_.add_constant(current_state, input, *input_type, dbg_info);
            } else {
                std::string input = utils::find_const_name_to_sdfg_name(this->constants_mapping_, input_operand);
                input_node = &this->builder_.add_access(current_state, input, dbg_info);
            }

            sdfg::types::Pointer base_ptr_type;
            sdfg::types::Pointer output_ptr_type(*input_type);
            this->builder_
                .add_dereference_memlet(current_state, *input_node, output_node, false, output_ptr_type, dbg_info);

            return current_state;
        }
        default:
            throw NotImplementedException(
                "Unsupported store instruction",
                ::docc::utils::get_debug_info(*instruction),
                ::docc::utils::toIRString(*instruction)
            );
    }
};

sdfg::control_flow::State& Lifting::visit_PHINode(
    const llvm::BasicBlock* block, const llvm::PHINode* instruction, sdfg::control_flow::State& current_state
) {
    // Define Debug
    auto dbg_info = ::docc::utils::get_debug_info(*instruction);

    // Define Output
    std::string output = utils::get_name(instruction);
    if (!this->builder_.subject().exists(output)) {
        auto output_type = utils::get_type(
            this->builder_,
            this->anonymous_types_mapping_,
            this->DL_,
            instruction->getType(),
            utils::get_storage_type(this->target_type_, 0)
        );
        assert(
            (output_type->type_id() == sdfg::types::TypeID::Pointer ||
             output_type->type_id() == sdfg::types::TypeID::Scalar) &&
            "PHINode: Expected pointer or scalar type as output"
        );
        this->builder_.add_container(output, *output_type);
    }
    auto& output_type = this->builder_.subject().type(output);

    // Determine Input
    llvm::Value* phi_value = nullptr;
    auto preds = this->pred_mapping_.at(&current_state);
    for (size_t i = 0; i < instruction->getNumIncomingValues(); ++i) {
        if (preds.contains(instruction->getIncomingBlock(i))) {
            phi_value = instruction->getIncomingValue(i);
            break;
        }
    }
    assert(phi_value != nullptr && "PHINode: Failed to determine phi value");

    // Handle UndefValue
    if (llvm::dyn_cast<llvm::UndefValue>(phi_value)) {
        return current_state;
    }

    // Define unsupported phi value types
    if (llvm::isa<llvm::ConstantExpr>(phi_value)) {
        throw NotImplementedException(
            "PHINode: Unsupported phi value type",
            ::docc::utils::get_debug_info(*instruction),
            ::docc::utils::toIRString(*instruction)
        );
    }

    // Define Operation
    switch (output_type.type_id()) {
        case sdfg::types::TypeID::Pointer: {
            auto& output_node = this->builder_.add_access(current_state, output, dbg_info);

            // Handle nullptr
            if (utils::is_literal(phi_value)) {
                sdfg::types::Pointer ptr_type;

                auto arg = utils::as_literal(llvm::dyn_cast<llvm::ConstantData>(phi_value));
                auto& input_node = this->builder_.add_constant(current_state, arg, ptr_type, dbg_info);
                this->builder_.add_reference_memlet(current_state, input_node, output_node, {}, ptr_type, dbg_info);

                return current_state;
            } else if (auto GV = llvm::dyn_cast<llvm::GlobalValue>(phi_value)) {
                std::string global_name = utils::get_name(GV);
                auto& input_node = this->builder_.add_access(current_state, global_name, dbg_info);

                sdfg::types::Pointer base_ptr_type;
                sdfg::types::Pointer ptr_type(static_cast<const sdfg::types::IType&>(base_ptr_type));
                this->builder_.add_reference_memlet(
                    current_state, input_node, output_node, {sdfg::symbolic::zero()}, ptr_type, dbg_info
                );

                return current_state;
            } else {
                // Handle local variable
                std::string input = utils::get_name(phi_value);
                if (!llvm::isa<llvm::Argument>(phi_value)) {
                    std::string transition_suffix = utils::get_name(block);
                    for (auto& pred : preds) {
                        transition_suffix += "_" + utils::get_name(pred);
                    }
                    input += "_" + transition_suffix;
                }

                auto& input_node = this->builder_.add_access(current_state, input, dbg_info);

                sdfg::types::Pointer base_ptr_type;
                sdfg::types::Pointer ptr_type(static_cast<const sdfg::types::IType&>(base_ptr_type));
                this->builder_.add_reference_memlet(
                    current_state, input_node, output_node, {sdfg::symbolic::zero()}, ptr_type, dbg_info
                );

                return current_state;
            }
        }
        case sdfg::types::TypeID::Scalar: {
            auto& output_node = this->builder_.add_access(current_state, output, dbg_info);

            // Handle literal
            if (utils::is_literal(phi_value)) {
                auto arg = utils::as_literal(llvm::dyn_cast<llvm::ConstantData>(phi_value));
                auto& input_node = this->builder_.add_constant(current_state, arg, output_type, dbg_info);
                auto& tasklet =
                    this->builder_
                        .add_tasklet(current_state, sdfg::data_flow::TaskletCode::assign, "__out", {"_in"}, dbg_info);
                this->builder_.add_computational_memlet(current_state, input_node, tasklet, "_in", {}, dbg_info);
                this->builder_.add_computational_memlet(current_state, tasklet, "__out", output_node, {}, dbg_info);

                return current_state;
            } else {
                // Handle local variable
                std::string phi_transition = utils::get_name(phi_value);
                if (!llvm::isa<llvm::Argument>(phi_value)) {
                    std::string transition_suffix = utils::get_name(block);
                    for (auto& pred : preds) {
                        transition_suffix += "_" + utils::get_name(pred);
                    }
                    phi_transition += "_" + transition_suffix;
                }

                auto& input_node = this->builder_.add_access(current_state, phi_transition, dbg_info);
                auto& tasklet =
                    this->builder_
                        .add_tasklet(current_state, sdfg::data_flow::TaskletCode::assign, "__out", {"_in"}, dbg_info);
                this->builder_.add_computational_memlet(current_state, input_node, tasklet, "_in", {}, dbg_info);
                this->builder_.add_computational_memlet(current_state, tasklet, "__out", output_node, {}, dbg_info);

                return current_state;
            }
        }
        default:
            throw NotImplementedException(
                "PHINode: Unsupported output type",
                ::docc::utils::get_debug_info(*instruction),
                ::docc::utils::toIRString(*instruction)
            );
    }

    return current_state;
};

sdfg::control_flow::State& Lifting::visit_ICmpInst(
    const llvm::BasicBlock* block, const llvm::ICmpInst* instruction, sdfg::control_flow::State& current_state
) {
    auto predicate = instruction->getPredicate();

    // Unsigned comparisions cannot be handled symbolically
    if ((predicate == llvm::CmpInst::Predicate::ICMP_UGT || predicate == llvm::CmpInst::Predicate::ICMP_UGE ||
         predicate == llvm::CmpInst::Predicate::ICMP_ULT || predicate == llvm::CmpInst::Predicate::ICMP_ULE) &&
        !instruction->getOperand(0)->getType()->isPointerTy()) {
        return this->visit_ICmpInst_dataflow(block, instruction, current_state);
    } else {
        return this->visit_ICmpInst_symbolic(block, instruction, current_state);
    }
}

sdfg::control_flow::State& Lifting::visit_ICmpInst_symbolic(
    const llvm::BasicBlock* block, const llvm::ICmpInst* instruction, sdfg::control_flow::State& current_state
) {
    // Define Debug
    auto dbg_info = ::docc::utils::get_debug_info(*instruction);

    // Define Output
    std::string output_str = utils::get_name(instruction);
    if (!this->builder_.subject().exists(output_str)) {
        auto output_type = utils::get_type(
            this->builder_,
            this->anonymous_types_mapping_,
            this->DL_,
            instruction->getType(),
            utils::get_storage_type(this->target_type_, 0)
        );
        this->builder_.add_container(output_str, *output_type);
    }
    auto& output_type = this->builder_.subject().type(output_str);
    assert(
        output_type.type_id() == sdfg::types::TypeID::Scalar &&
        output_type.primitive_type() == sdfg::types::PrimitiveType::Bool && "ICmpInst: Expected boolean type as output"
    );
    sdfg::symbolic::Symbol output = sdfg::symbolic::symbol(output_str);

    auto left_operand = instruction->getOperand(0);
    auto left_operand_type = utils::get_type(
        this->builder_,
        this->anonymous_types_mapping_,
        this->DL_,
        left_operand->getType(),
        utils::get_storage_type(this->target_type_, 0)
    );
    assert(
        (left_operand_type->type_id() == sdfg::types::TypeID::Scalar ||
         left_operand_type->type_id() == sdfg::types::TypeID::Pointer) &&
        "ICmpInst: Expected scalar or pointer type as left operand"
    );

    sdfg::symbolic::Expression left_operand_expr;
    if (llvm::isa<llvm::ConstantInt>(left_operand)) {
        left_operand_expr = sdfg::symbolic::integer(llvm::dyn_cast<llvm::ConstantInt>(left_operand)->getSExtValue());
    } else if (llvm::isa<llvm::ConstantPointerNull>(left_operand)) {
        left_operand_expr = sdfg::symbolic::__nullptr__();
    } else {
        assert(!utils::is_literal(left_operand) && "ICmpInst: Expected constant or symbol as left operand");
        std::string left_operand_expr_str = utils::find_const_name_to_sdfg_name(this->constants_mapping_, left_operand);
        assert(
            this->builder_.subject().exists(left_operand_expr_str) &&
            "ICmpInst: Expected input to be a constant or a symbol"
        );
        left_operand_expr = sdfg::symbolic::symbol(left_operand_expr_str);
    }

    auto right_operand = instruction->getOperand(1);
    auto right_operand_type = utils::get_type(
        this->builder_,
        this->anonymous_types_mapping_,
        this->DL_,
        right_operand->getType(),
        utils::get_storage_type(this->target_type_, 0)
    );
    assert(
        (right_operand_type->type_id() == sdfg::types::TypeID::Scalar ||
         right_operand_type->type_id() == sdfg::types::TypeID::Pointer) &&
        "ICmpInst: Expected scalar or pointer type as right operand"
    );

    sdfg::symbolic::Expression right_operand_expr;
    if (llvm::isa<llvm::ConstantInt>(right_operand)) {
        right_operand_expr = sdfg::symbolic::integer(llvm::dyn_cast<llvm::ConstantInt>(right_operand)->getSExtValue());
    } else if (llvm::isa<llvm::ConstantPointerNull>(right_operand)) {
        right_operand_expr = sdfg::symbolic::__nullptr__();
    } else {
        assert(!utils::is_literal(right_operand) && "ICmpInst: Expected constant or symbol as right operand");
        std::string right_operand_expr_str =
            utils::find_const_name_to_sdfg_name(this->constants_mapping_, right_operand);
        assert(
            this->builder_.subject().exists(right_operand_expr_str) &&
            "ICmpInst: Expected input to be a constant or a symbol"
        );
        right_operand_expr = sdfg::symbolic::symbol(right_operand_expr_str);
    }

    // SDFGs support symbolic expressions for signed integers
    // and pointer comparisons only
    sdfg::symbolic::Condition condition;
    auto predicate = instruction->getSignedPredicate();
    switch (predicate) {
        case llvm::CmpInst::Predicate::ICMP_EQ: {
            condition = sdfg::symbolic::Eq(left_operand_expr, right_operand_expr);
            break;
        }
        case llvm::CmpInst::Predicate::ICMP_NE: {
            condition = sdfg::symbolic::Ne(left_operand_expr, right_operand_expr);
            break;
        }
        case llvm::CmpInst::Predicate::ICMP_SGT: {
            condition = sdfg::symbolic::Gt(left_operand_expr, right_operand_expr);
            break;
        }
        case llvm::CmpInst::Predicate::ICMP_SGE: {
            condition = sdfg::symbolic::Ge(left_operand_expr, right_operand_expr);
            break;
        }
        case llvm::CmpInst::Predicate::ICMP_SLT: {
            condition = sdfg::symbolic::Lt(left_operand_expr, right_operand_expr);
            break;
        }
        case llvm::CmpInst::Predicate::ICMP_SLE: {
            condition = sdfg::symbolic::Le(left_operand_expr, right_operand_expr);
            break;
        }
        default: {
            throw NotImplementedException(
                "ICmpInst: Unsupported predicate",
                ::docc::utils::get_debug_info(*instruction),
                ::docc::utils::toIRString(*instruction)
            );
        }
    }

    auto& state_end = this->builder_.add_state_after(current_state, false, dbg_info);
    auto& state_cond = this->builder_.add_state(false, dbg_info);
    this->builder_.add_edge(current_state, state_cond, {{output, condition}}, dbg_info);
    this->builder_.add_edge(state_cond, state_end, dbg_info);

    return state_end;
};

sdfg::control_flow::State& Lifting::visit_ICmpInst_dataflow(
    const llvm::BasicBlock* block, const llvm::ICmpInst* instruction, sdfg::control_flow::State& current_state
) {
    // Define Debug
    auto dbg_info = ::docc::utils::get_debug_info(*instruction);

    // Define Output
    std::string output = utils::get_name(instruction);
    if (!this->builder_.subject().exists(output)) {
        auto output_type = utils::get_type(
            this->builder_,
            this->anonymous_types_mapping_,
            this->DL_,
            instruction->getType(),
            utils::get_storage_type(this->target_type_, 0)
        );
        this->builder_.add_container(output, *output_type);
    }
    auto& output_type = this->builder_.subject().type(output);
    assert(output_type.type_id() == sdfg::types::TypeID::Scalar && "ICmpInst: Expected scalar type as output");

    // Define Operation
    sdfg::data_flow::TaskletCode operation;
    switch (instruction->getPredicate()) {
        case llvm::CmpInst::Predicate::ICMP_UGT: {
            operation = sdfg::data_flow::TaskletCode::int_ugt;
            break;
        }
        case llvm::CmpInst::Predicate::ICMP_UGE: {
            operation = sdfg::data_flow::TaskletCode::int_uge;
            break;
        }
        case llvm::CmpInst::Predicate::ICMP_ULT: {
            operation = sdfg::data_flow::TaskletCode::int_ult;
            break;
        }
        case llvm::CmpInst::Predicate::ICMP_ULE: {
            operation = sdfg::data_flow::TaskletCode::int_ule;
            break;
        }
        default: {
            throw NotImplementedException(
                "FCmpInst: Unsupported predicate",
                ::docc::utils::get_debug_info(*instruction),
                ::docc::utils::toIRString(*instruction)
            );
        }
    }

    auto left_operand = instruction->getOperand(0);
    auto right_operand = instruction->getOperand(1);
    assert(
        left_operand->getType()->isIntegerTy() && right_operand->getType()->isIntegerTy() &&
        "ICmpInst: Expected integer types as operands"
    );
    switch (output_type.type_id()) {
        case sdfg::types::TypeID::Scalar: {
            auto& output_node = this->builder_.add_access(current_state, output, dbg_info);
            auto& tasklet = this->builder_.add_tasklet(current_state, operation, "__out", {"_in1", "_in2"}, dbg_info);
            this->builder_.add_computational_memlet(current_state, tasklet, "__out", output_node, {}, dbg_info);

            sdfg::data_flow::AccessNode* left_node;
            if (utils::is_literal(left_operand)) {
                std::string input = utils::as_literal(llvm::dyn_cast<llvm::ConstantData>(left_operand));
                auto left_operand_type = utils::get_type(
                    this->builder_,
                    this->anonymous_types_mapping_,
                    this->DL_,
                    left_operand->getType(),
                    utils::get_storage_type(this->target_type_, 0)
                );
                left_node = &this->builder_.add_constant(current_state, input, *left_operand_type, dbg_info);
            } else {
                std::string left_input = utils::find_const_name_to_sdfg_name(this->constants_mapping_, left_operand);
                left_node = &this->builder_.add_access(current_state, left_input, dbg_info);
            }

            sdfg::data_flow::AccessNode* right_node;
            if (utils::is_literal(right_operand)) {
                std::string input = utils::as_literal(llvm::dyn_cast<llvm::ConstantData>(right_operand));
                auto right_operand_type = utils::get_type(
                    this->builder_,
                    this->anonymous_types_mapping_,
                    this->DL_,
                    right_operand->getType(),
                    utils::get_storage_type(this->target_type_, 0)
                );
                right_node = &this->builder_.add_constant(current_state, input, *right_operand_type, dbg_info);
            } else {
                std::string input = utils::find_const_name_to_sdfg_name(this->constants_mapping_, right_operand);
                right_node = &this->builder_.add_access(current_state, input, dbg_info);
            }

            this->builder_.add_computational_memlet(current_state, *left_node, tasklet, "_in1", {}, dbg_info);
            this->builder_.add_computational_memlet(current_state, *right_node, tasklet, "_in2", {}, dbg_info);

            return current_state;
        }
        default: {
            throw NotImplementedException(
                "ICmpInst: Unsupported output type",
                ::docc::utils::get_debug_info(*instruction),
                ::docc::utils::toIRString(*instruction)
            );
        }
    }
};


sdfg::control_flow::State& Lifting::visit_SelectInst(
    const llvm::BasicBlock* block, const llvm::SelectInst* instruction, sdfg::control_flow::State& current_state
) {
    // Define Debug
    auto dbg_info = ::docc::utils::get_debug_info(*instruction);

    // Define Output
    std::string output = utils::get_name(instruction);
    if (!this->builder_.subject().exists(output)) {
        auto output_type = utils::get_type(
            this->builder_,
            this->anonymous_types_mapping_,
            this->DL_,
            instruction->getType(),
            utils::get_storage_type(this->target_type_, 0)
        );
        this->builder_.add_container(output, *output_type);
    }
    auto& output_type = this->builder_.subject().type(output);
    assert(
        (output_type.type_id() == sdfg::types::TypeID::Scalar || output_type.type_id() == sdfg::types::TypeID::Pointer
        ) &&
        "SelectInst: Expected scalar, array, or pointer type as output"
    );

    // Define Condition
    auto condition = instruction->getCondition();
    auto condition_type = utils::get_type(
        this->builder_,
        this->anonymous_types_mapping_,
        this->DL_,
        condition->getType(),
        utils::get_storage_type(this->target_type_, 0)
    );
    assert((condition_type->type_id() == sdfg::types::TypeID::Scalar) && "SelectInst: Expected scalar type as condition");

    auto value_if = instruction->getTrueValue();
    auto value_else = instruction->getFalseValue();
    switch (condition_type->type_id()) {
        case sdfg::types::TypeID::Scalar: {
            sdfg::symbolic::Expression condition_sym;
            if (utils::is_literal(condition)) {
                condition_sym = utils::as_symbol(llvm::dyn_cast<const llvm::Constant>(condition));
            } else {
                auto condition_name = utils::find_const_name_to_sdfg_name(this->constants_mapping_, condition);
                condition_sym = sdfg::symbolic::symbol(condition_name);
            }

            auto& state_if = this->builder_.add_state(false, dbg_info);
            auto& state_else = this->builder_.add_state(false, dbg_info);
            auto& state_end = this->builder_.add_state_after(current_state, false, dbg_info);

            this->builder_.add_edge(
                current_state, state_if, sdfg::symbolic::Ne(condition_sym, sdfg::symbolic::__false__()), dbg_info
            );
            this->builder_.add_edge(
                current_state, state_else, sdfg::symbolic::Eq(condition_sym, sdfg::symbolic::__false__()), dbg_info
            );
            this->builder_.add_edge(state_if, state_end, dbg_info);
            this->builder_.add_edge(state_else, state_end, dbg_info);

            // If
            {
                auto& output_node = this->builder_.add_access(state_if, output, dbg_info);

                switch (output_type.type_id()) {
                    case sdfg::types::TypeID::Scalar: {
                        auto& tasklet =
                            this->builder_
                                .add_tasklet(state_if, sdfg::data_flow::TaskletCode::assign, "__out", {"_in"}, dbg_info);
                        this->builder_.add_computational_memlet(state_if, tasklet, "__out", output_node, {}, dbg_info);

                        sdfg::data_flow::AccessNode* input_node;
                        if (utils::is_literal(value_if)) {
                            std::string input = utils::as_literal(llvm::dyn_cast<llvm::ConstantData>(value_if));
                            input_node = &this->builder_.add_constant(state_if, input, output_type, dbg_info);
                        } else {
                            std::string input = utils::find_const_name_to_sdfg_name(this->constants_mapping_, value_if);
                            input_node = &this->builder_.add_access(state_if, input, dbg_info);
                        }
                        this->builder_.add_computational_memlet(state_if, *input_node, tasklet, "_in", {}, dbg_info);
                        break;
                    }
                    case sdfg::types::TypeID::Pointer: {
                        if (utils::is_literal(value_if)) {
                            std::string input = utils::as_literal(llvm::dyn_cast<llvm::ConstantData>(value_if));
                            auto& input_node = this->builder_.add_constant(state_if, input, output_type, dbg_info);

                            sdfg::types::Pointer ptr_type;
                            this->builder_
                                .add_reference_memlet(state_if, input_node, output_node, {}, ptr_type, dbg_info);
                        } else {
                            std::string input = utils::find_const_name_to_sdfg_name(this->constants_mapping_, value_if);
                            auto& input_node = this->builder_.add_access(state_if, input, dbg_info);

                            sdfg::types::Pointer base_ptr_type;
                            sdfg::types::Pointer ptr_type(static_cast<const sdfg::types::IType&>(base_ptr_type));
                            this->builder_.add_reference_memlet(
                                state_if, input_node, output_node, {sdfg::symbolic::zero()}, ptr_type, dbg_info
                            );
                        }
                        break;
                    }
                    default: {
                        throw NotImplementedException(
                            "SelectInst: Unsupported output type",
                            ::docc::utils::get_debug_info(*instruction),
                            ::docc::utils::toIRString(*instruction)
                        );
                    }
                }
            }

            // Else
            {
                auto& output_node = this->builder_.add_access(state_else, output, dbg_info);

                switch (output_type.type_id()) {
                    case sdfg::types::TypeID::Scalar: {
                        auto& tasklet =
                            this->builder_
                                .add_tasklet(state_else, sdfg::data_flow::TaskletCode::assign, "__out", {"_in"}, dbg_info);
                        this->builder_.add_computational_memlet(state_else, tasklet, "__out", output_node, {}, dbg_info);

                        sdfg::data_flow::AccessNode* input_node;
                        if (utils::is_literal(value_else)) {
                            std::string input = utils::as_literal(llvm::dyn_cast<llvm::ConstantData>(value_else));
                            input_node = &this->builder_.add_constant(state_else, input, output_type, dbg_info);
                        } else {
                            std::string input =
                                utils::find_const_name_to_sdfg_name(this->constants_mapping_, value_else);
                            input_node = &this->builder_.add_access(state_else, input, dbg_info);
                        }
                        this->builder_.add_computational_memlet(state_else, *input_node, tasklet, "_in", {}, dbg_info);

                        break;
                    }
                    case sdfg::types::TypeID::Pointer: {
                        if (utils::is_literal(value_else)) {
                            std::string input = utils::as_literal(llvm::dyn_cast<llvm::ConstantData>(value_else));
                            auto& input_node = this->builder_.add_constant(state_else, input, output_type, dbg_info);

                            sdfg::types::Pointer ptr_type;
                            this->builder_
                                .add_reference_memlet(state_else, input_node, output_node, {}, ptr_type, dbg_info);
                        } else {
                            std::string input =
                                utils::find_const_name_to_sdfg_name(this->constants_mapping_, value_else);
                            auto& input_node = this->builder_.add_access(state_else, input, dbg_info);

                            sdfg::types::Pointer base_ptr_type;
                            sdfg::types::Pointer ptr_type(static_cast<const sdfg::types::IType&>(base_ptr_type));
                            this->builder_.add_reference_memlet(
                                state_else, input_node, output_node, {sdfg::symbolic::zero()}, ptr_type, dbg_info
                            );
                        }

                        break;
                    }
                    default: {
                        throw NotImplementedException(
                            "SelectInst: Unsupported output type",
                            ::docc::utils::get_debug_info(*instruction),
                            ::docc::utils::toIRString(*instruction)
                        );
                    }
                }
            }

            return state_end;
        }
        default: {
            throw NotImplementedException(
                "SelectInst: Unsupported condition type",
                ::docc::utils::get_debug_info(*instruction),
                ::docc::utils::toIRString(*instruction)
            );
        }
    }
};

sdfg::control_flow::State& Lifting::visit_FCmpInst(
    const llvm::BasicBlock* block, const llvm::FCmpInst* instruction, sdfg::control_flow::State& current_state
) {
    // Define Debug
    auto dbg_info = ::docc::utils::get_debug_info(*instruction);

    // Define Output
    std::string output = utils::get_name(instruction);
    if (!this->builder_.subject().exists(output)) {
        auto output_type = utils::get_type(
            this->builder_,
            this->anonymous_types_mapping_,
            this->DL_,
            instruction->getType(),
            utils::get_storage_type(this->target_type_, 0)
        );
        this->builder_.add_container(output, *output_type);
    }
    auto& output_type = this->builder_.subject().type(output);
    assert(
        (output_type.type_id() == sdfg::types::TypeID::Scalar || output_type.type_id() == sdfg::types::TypeID::Structure
        ) &&
        "FCmpInst: Expected scalar or structure type as output"
    );

    // Define Operation
    sdfg::data_flow::TaskletCode operation;
    switch (instruction->getPredicate()) {
        case llvm::CmpInst::Predicate::FCMP_FALSE: {
            operation = sdfg::data_flow::TaskletCode::assign;
            break;
        }
        case llvm::CmpInst::Predicate::FCMP_OEQ: {
            operation = sdfg::data_flow::TaskletCode::fp_oeq;
            break;
        }
        case llvm::CmpInst::Predicate::FCMP_OGT: {
            operation = sdfg::data_flow::TaskletCode::fp_ogt;
            break;
        }
        case llvm::CmpInst::Predicate::FCMP_OGE: {
            operation = sdfg::data_flow::TaskletCode::fp_oge;
            break;
        }
        case llvm::CmpInst::Predicate::FCMP_OLT: {
            operation = sdfg::data_flow::TaskletCode::fp_olt;
            break;
        }
        case llvm::CmpInst::Predicate::FCMP_OLE: {
            operation = sdfg::data_flow::TaskletCode::fp_ole;
            break;
        }
        case llvm::CmpInst::Predicate::FCMP_ONE: {
            operation = sdfg::data_flow::TaskletCode::fp_one;
            break;
        }
        case llvm::CmpInst::Predicate::FCMP_ORD: {
            operation = sdfg::data_flow::TaskletCode::fp_ord;
            break;
        }
        case llvm::CmpInst::Predicate::FCMP_UEQ: {
            operation = sdfg::data_flow::TaskletCode::fp_ueq;
            break;
        }
        case llvm::CmpInst::Predicate::FCMP_UGT: {
            operation = sdfg::data_flow::TaskletCode::fp_ugt;
            break;
        }
        case llvm::CmpInst::Predicate::FCMP_UGE: {
            operation = sdfg::data_flow::TaskletCode::fp_uge;
            break;
        }
        case llvm::CmpInst::Predicate::FCMP_ULT: {
            operation = sdfg::data_flow::TaskletCode::fp_ult;
            break;
        }
        case llvm::CmpInst::Predicate::FCMP_ULE: {
            operation = sdfg::data_flow::TaskletCode::fp_ule;
            break;
        }
        case llvm::CmpInst::Predicate::FCMP_UNE: {
            operation = sdfg::data_flow::TaskletCode::fp_une;
            break;
        }
        case llvm::CmpInst::Predicate::FCMP_UNO: {
            operation = sdfg::data_flow::TaskletCode::fp_uno;
            break;
        }
        case llvm::CmpInst::Predicate::FCMP_TRUE: {
            operation = sdfg::data_flow::TaskletCode::assign;
            break;
        }
        default: {
            throw NotImplementedException(
                "FCmpInst: Unsupported predicate",
                ::docc::utils::get_debug_info(*instruction),
                ::docc::utils::toIRString(*instruction)
            );
        }
    }

    auto left_operand = instruction->getOperand(0);
    auto right_operand = instruction->getOperand(1);
    switch (output_type.type_id()) {
        case sdfg::types::TypeID::Scalar: {
            auto& output_node = this->builder_.add_access(current_state, output, dbg_info);
            auto& tasklet = this->builder_.add_tasklet(current_state, operation, "__out", {"_in1", "_in2"}, dbg_info);
            this->builder_.add_computational_memlet(current_state, tasklet, "__out", output_node, {}, dbg_info);

            sdfg::data_flow::AccessNode* left_node;
            if (utils::is_literal(left_operand)) {
                std::string input = utils::as_literal(llvm::dyn_cast<llvm::ConstantData>(left_operand));
                auto left_operand_type = utils::get_type(
                    this->builder_,
                    this->anonymous_types_mapping_,
                    this->DL_,
                    left_operand->getType(),
                    utils::get_storage_type(this->target_type_, 0)
                );
                left_node = &this->builder_.add_constant(current_state, input, *left_operand_type, dbg_info);
            } else {
                std::string left_input = utils::find_const_name_to_sdfg_name(this->constants_mapping_, left_operand);
                left_node = &this->builder_.add_access(current_state, left_input, dbg_info);
            }

            sdfg::data_flow::AccessNode* right_node;
            if (utils::is_literal(right_operand)) {
                std::string input = utils::as_literal(llvm::dyn_cast<llvm::ConstantData>(right_operand));
                auto right_operand_type = utils::get_type(
                    this->builder_,
                    this->anonymous_types_mapping_,
                    this->DL_,
                    right_operand->getType(),
                    utils::get_storage_type(this->target_type_, 0)
                );
                right_node = &this->builder_.add_constant(current_state, input, *right_operand_type, dbg_info);
            } else {
                std::string input = utils::find_const_name_to_sdfg_name(this->constants_mapping_, right_operand);
                right_node = &this->builder_.add_access(current_state, input, dbg_info);
            }

            this->builder_.add_computational_memlet(current_state, *left_node, tasklet, "_in1", {}, dbg_info);
            this->builder_.add_computational_memlet(current_state, *right_node, tasklet, "_in2", {}, dbg_info);

            return current_state;
        }
        case sdfg::types::TypeID::Structure: {
            auto& struct_type = static_cast<const sdfg::types::Structure&>(output_type);
            auto& struct_def = this->builder_.subject().structure(struct_type.name());
            assert(struct_def.is_vector() && "FCmpInst: Expected vector structure type as output");

            // Define loop
            std::string iterator = this->builder_.find_new_name("_i");
            this->builder_.add_container(iterator, sdfg::types::Scalar(sdfg::types::PrimitiveType::Int64));
            sdfg::symbolic::Symbol iter_sym = sdfg::symbolic::symbol(iterator);
            sdfg::symbolic::Expression init = sdfg::symbolic::zero();
            sdfg::symbolic::Condition cond =
                sdfg::symbolic::Lt(iter_sym, sdfg::symbolic::integer(struct_def.vector_size()));
            sdfg::symbolic::Expression update = SymEngine::add(iter_sym, sdfg::symbolic::one());

            auto loop = this->builder_.add_loop(current_state, iter_sym, init, cond, update, dbg_info);
            auto& body = std::get<1>(loop);

            // Define body
            auto& output_node = this->builder_.add_access(body, output, dbg_info);
            auto& tasklet = this->builder_.add_tasklet(body, operation, "__out", {"_in1", "_in2"}, dbg_info);
            this->builder_.add_computational_memlet(body, tasklet, "__out", output_node, {iter_sym}, dbg_info);

            sdfg::data_flow::AccessNode* left_node;
            if (utils::is_literal(left_operand)) {
                std::string input = utils::as_initializer(llvm::dyn_cast<llvm::ConstantData>(left_operand));
                auto left_operand_type = utils::get_type(
                    this->builder_,
                    this->anonymous_types_mapping_,
                    this->DL_,
                    left_operand->getType(),
                    utils::get_storage_type(this->target_type_, 0)
                );
                left_node = &this->builder_.add_constant(body, input, *left_operand_type, dbg_info);
            } else {
                std::string left_input = utils::find_const_name_to_sdfg_name(this->constants_mapping_, left_operand);
                left_node = &this->builder_.add_access(body, left_input, dbg_info);
            }

            sdfg::data_flow::AccessNode* right_node;
            if (utils::is_literal(right_operand)) {
                std::string input = utils::as_initializer(llvm::dyn_cast<llvm::ConstantData>(right_operand));
                auto right_operand_type = utils::get_type(
                    this->builder_,
                    this->anonymous_types_mapping_,
                    this->DL_,
                    right_operand->getType(),
                    utils::get_storage_type(this->target_type_, 0)
                );
                right_node = &this->builder_.add_constant(body, input, *right_operand_type, dbg_info);
            } else {
                std::string right_input = utils::find_const_name_to_sdfg_name(this->constants_mapping_, right_operand);
                right_node = &this->builder_.add_access(body, right_input, dbg_info);
            }

            this->builder_.add_computational_memlet(body, *left_node, tasklet, "_in1", {iter_sym}, dbg_info);
            this->builder_.add_computational_memlet(body, *right_node, tasklet, "_in2", {iter_sym}, dbg_info);

            return std::get<2>(loop);
        }
        default: {
            throw NotImplementedException(
                "FCmpInst: Unsupported output type",
                ::docc::utils::get_debug_info(*instruction),
                ::docc::utils::toIRString(*instruction)
            );
        }
    }
};

sdfg::control_flow::State& Lifting::visit_UnaryOperator(
    const llvm::BasicBlock* block, const llvm::UnaryOperator* instruction, sdfg::control_flow::State& current_state
) {
    // Define Debug
    auto dbg_info = ::docc::utils::get_debug_info(*instruction);

    // Define Output
    std::string output = utils::get_name(instruction);
    if (!this->builder_.subject().exists(output)) {
        auto output_type = utils::get_type(
            this->builder_,
            this->anonymous_types_mapping_,
            this->DL_,
            instruction->getType(),
            utils::get_storage_type(this->target_type_, 0)
        );
        this->builder_.add_container(output, *output_type);
    }
    auto& output_type = this->builder_.subject().type(output);
    assert(
        (output_type.type_id() == sdfg::types::TypeID::Scalar || output_type.type_id() == sdfg::types::TypeID::Structure
        ) &&
        "UnaryOperator: Expected scalar or structure type as output"
    );

    // Define Operation
    sdfg::data_flow::TaskletCode operation;
    switch (instruction->getOpcode()) {
        case llvm::Instruction::UnaryOps::FNeg: {
            operation = sdfg::data_flow::TaskletCode::fp_neg;
            break;
        }
        default: {
            throw NotImplementedException(
                "UnaryOperator: Unsupported opcode",
                ::docc::utils::get_debug_info(*instruction),
                ::docc::utils::toIRString(*instruction)
            );
        }
    }

    auto operand = instruction->getOperand(0);
    switch (output_type.type_id()) {
        case sdfg::types::TypeID::Scalar: {
            auto& output_node = this->builder_.add_access(current_state, output, dbg_info);
            auto& tasklet = this->builder_.add_tasklet(current_state, operation, "__out", {"_in1"}, dbg_info);
            this->builder_.add_computational_memlet(current_state, tasklet, "__out", output_node, {}, dbg_info);

            if (utils::is_literal(operand)) {
                std::string input = utils::as_literal(llvm::dyn_cast<llvm::ConstantData>(operand));
                auto& input_node = this->builder_.add_constant(current_state, input, output_type, dbg_info);
                this->builder_.add_computational_memlet(current_state, input_node, tasklet, "_in1", {}, dbg_info);
            } else {
                std::string input = utils::find_const_name_to_sdfg_name(this->constants_mapping_, operand);
                auto& input_node = this->builder_.add_access(current_state, input, dbg_info);
                this->builder_.add_computational_memlet(current_state, input_node, tasklet, "_in1", {}, dbg_info);
            }

            return current_state;
        }
        case sdfg::types::TypeID::Structure: {
            auto& struct_type = static_cast<const sdfg::types::Structure&>(output_type);
            auto& struct_def = this->builder_.subject().structure(struct_type.name());
            assert(struct_def.is_vector() && "UnaryOperator: Expected vector structure type as output");

            // Define loop
            std::string iterator = this->builder_.find_new_name("_i");
            this->builder_.add_container(iterator, sdfg::types::Scalar(sdfg::types::PrimitiveType::Int64));
            sdfg::symbolic::Symbol iter_sym = sdfg::symbolic::symbol(iterator);
            sdfg::symbolic::Expression init = sdfg::symbolic::zero();
            sdfg::symbolic::Condition cond =
                sdfg::symbolic::Lt(iter_sym, sdfg::symbolic::integer(struct_def.vector_size()));
            sdfg::symbolic::Expression update = SymEngine::add(iter_sym, sdfg::symbolic::one());

            auto loop = this->builder_.add_loop(current_state, iter_sym, init, cond, update, dbg_info);
            auto& body = std::get<1>(loop);

            // Define body
            auto& output_node = this->builder_.add_access(body, output, dbg_info);
            auto& tasklet = this->builder_.add_tasklet(body, operation, "__out", {"_in1"}, dbg_info);
            this->builder_.add_computational_memlet(body, tasklet, "__out", output_node, {iter_sym}, dbg_info);

            if (utils::is_literal(operand)) {
                std::string input = utils::as_initializer(llvm::dyn_cast<llvm::ConstantData>(operand));
                auto& input_node = this->builder_.add_constant(body, input, output_type, dbg_info);
                this->builder_.add_computational_memlet(body, input_node, tasklet, "_in1", {iter_sym}, dbg_info);
            } else {
                std::string input = utils::find_const_name_to_sdfg_name(this->constants_mapping_, operand);
                auto& input_node = this->builder_.add_access(body, input, dbg_info);
                this->builder_.add_computational_memlet(body, input_node, tasklet, "_in1", {iter_sym}, dbg_info);
            }

            return std::get<2>(loop);
        }
        default: {
            throw NotImplementedException(
                "UnaryOperator: Unsupported output type",
                ::docc::utils::get_debug_info(*instruction),
                ::docc::utils::toIRString(*instruction)
            );
        }
    }
};

sdfg::control_flow::State& Lifting::visit_BinaryOperator(
    const llvm::BasicBlock* block, const llvm::BinaryOperator* instruction, sdfg::control_flow::State& current_state
) {
    // Define Debug
    auto dbg_info = ::docc::utils::get_debug_info(*instruction);

    // Define Output
    std::string output = utils::get_name(instruction);
    if (!this->builder_.subject().exists(output)) {
        auto output_type = utils::get_type(
            this->builder_,
            this->anonymous_types_mapping_,
            this->DL_,
            instruction->getType(),
            utils::get_storage_type(this->target_type_, 0)
        );
        this->builder_.add_container(output, *output_type);
    }
    auto& output_type = this->builder_.subject().type(output);
    assert(
        (output_type.type_id() == sdfg::types::TypeID::Scalar || output_type.type_id() == sdfg::types::TypeID::Structure
        ) &&
        "BinaryOperator: Expected scalar or structure type as output"
    );

    // Define Operation
    sdfg::data_flow::TaskletCode operation;
    switch (instruction->getOpcode()) {
        case llvm::Instruction::BinaryOps::FAdd: {
            operation = sdfg::data_flow::TaskletCode::fp_add;
            break;
        }
        case llvm::Instruction::BinaryOps::FSub: {
            operation = sdfg::data_flow::TaskletCode::fp_sub;
            break;
        }
        case llvm::Instruction::BinaryOps::FMul: {
            operation = sdfg::data_flow::TaskletCode::fp_mul;
            break;
        }
        case llvm::Instruction::BinaryOps::FDiv: {
            operation = sdfg::data_flow::TaskletCode::fp_div;
            break;
        }
        case llvm::Instruction::BinaryOps::FRem: {
            operation = sdfg::data_flow::TaskletCode::fp_rem;
            break;
        }
        case llvm::Instruction::BinaryOps::Add: {
            operation = sdfg::data_flow::TaskletCode::int_add;
            break;
        }
        case llvm::Instruction::BinaryOps::Sub: {
            operation = sdfg::data_flow::TaskletCode::int_sub;
            break;
        }
        case llvm::Instruction::BinaryOps::Mul: {
            operation = sdfg::data_flow::TaskletCode::int_mul;
            break;
        }
        case llvm::Instruction::BinaryOps::SDiv: {
            operation = sdfg::data_flow::TaskletCode::int_sdiv;
            break;
        }
        case llvm::Instruction::BinaryOps::SRem: {
            operation = sdfg::data_flow::TaskletCode::int_srem;
            break;
        }
        case llvm::Instruction::BinaryOps::UDiv: {
            operation = sdfg::data_flow::TaskletCode::int_udiv;
            break;
        }
        case llvm::Instruction::BinaryOps::URem: {
            operation = sdfg::data_flow::TaskletCode::int_urem;
            break;
        }
        case llvm::Instruction::BinaryOps::And: {
            operation = sdfg::data_flow::TaskletCode::int_and;
            break;
        }
        case llvm::Instruction::BinaryOps::Or: {
            // If disjoint attribute, write as add
            if (auto disjoint_inst = llvm::dyn_cast<llvm::PossiblyDisjointInst>(instruction)) {
                if (disjoint_inst->isDisjoint()) {
                    operation = sdfg::data_flow::TaskletCode::int_add;
                    break;
                }
            }
            operation = sdfg::data_flow::TaskletCode::int_or;
            break;
        }
        case llvm::Instruction::BinaryOps::Xor: {
            // If disjoint attribute, write as add
            if (auto disjoint_inst = llvm::dyn_cast<llvm::PossiblyDisjointInst>(instruction)) {
                if (disjoint_inst->isDisjoint()) {
                    operation = sdfg::data_flow::TaskletCode::int_add;
                    break;
                }
            }
            operation = sdfg::data_flow::TaskletCode::int_xor;
            break;
        }
        case llvm::Instruction::BinaryOps::Shl: {
            operation = sdfg::data_flow::TaskletCode::int_shl;
            break;
        }
        case llvm::Instruction::BinaryOps::LShr: {
            operation = sdfg::data_flow::TaskletCode::int_lshr;
            break;
        }
        case llvm::Instruction::BinaryOps::AShr: {
            operation = sdfg::data_flow::TaskletCode::int_ashr;
            break;
        }
        default: {
            throw NotImplementedException(
                "BinaryOperator: Unsupported opcode",
                ::docc::utils::get_debug_info(*instruction),
                ::docc::utils::toIRString(*instruction)
            );
        }
    }

    auto left_operand = instruction->getOperand(0);
    auto right_operand = instruction->getOperand(1);
    switch (output_type.type_id()) {
        case sdfg::types::TypeID::Scalar: {
            auto& output_node = this->builder_.add_access(current_state, output, dbg_info);
            auto& tasklet = this->builder_.add_tasklet(current_state, operation, "__out", {"_in1", "_in2"}, dbg_info);

            sdfg::data_flow::AccessNode* left_node;
            auto left_input_type = utils::get_type(
                this->builder_,
                this->anonymous_types_mapping_,
                this->DL_,
                left_operand->getType(),
                utils::get_storage_type(this->target_type_, 0)
            );
            if (utils::is_literal(left_operand)) {
                std::string input = utils::as_literal(llvm::dyn_cast<llvm::ConstantData>(left_operand));
                left_node = &this->builder_.add_constant(current_state, input, *left_input_type, dbg_info);
            } else {
                std::string left_input = utils::find_const_name_to_sdfg_name(this->constants_mapping_, left_operand);
                left_node = &this->builder_.add_access(current_state, left_input, dbg_info);
            }

            sdfg::data_flow::AccessNode* right_node;
            auto right_input_type = utils::get_type(
                this->builder_,
                this->anonymous_types_mapping_,
                this->DL_,
                right_operand->getType(),
                utils::get_storage_type(this->target_type_, 0)
            );
            if (utils::is_literal(right_operand)) {
                std::string input = utils::as_literal(llvm::dyn_cast<llvm::ConstantData>(right_operand));
                right_node = &this->builder_.add_constant(current_state, input, *right_input_type, dbg_info);
            } else if (left_operand != right_operand) {
                std::string input = utils::find_const_name_to_sdfg_name(this->constants_mapping_, right_operand);
                right_node = &this->builder_.add_access(current_state, input, dbg_info);
            } else {
                right_node = left_node;
            }

            this->builder_
                .add_computational_memlet(current_state, tasklet, "__out", output_node, {}, output_type, dbg_info);
            this->builder_
                .add_computational_memlet(current_state, *left_node, tasklet, "_in1", {}, *left_input_type, dbg_info);
            this->builder_
                .add_computational_memlet(current_state, *right_node, tasklet, "_in2", {}, *right_input_type, dbg_info);

            return current_state;
        }
        case sdfg::types::TypeID::Structure: {
            auto& struct_type = static_cast<const sdfg::types::Structure&>(output_type);
            auto& struct_def = this->builder_.subject().structure(struct_type.name());
            assert(struct_def.is_vector() && "FCmpInst: Expected vector structure type as output");

            // Define loop
            std::string iterator = this->builder_.find_new_name("_i");
            this->builder_.add_container(iterator, sdfg::types::Scalar(sdfg::types::PrimitiveType::Int64));
            sdfg::symbolic::Symbol iter_sym = sdfg::symbolic::symbol(iterator);
            sdfg::symbolic::Expression init = sdfg::symbolic::zero();
            sdfg::symbolic::Condition cond =
                sdfg::symbolic::Lt(iter_sym, sdfg::symbolic::integer(struct_def.vector_size()));
            sdfg::symbolic::Expression update = SymEngine::add(iter_sym, sdfg::symbolic::one());

            auto loop = this->builder_.add_loop(current_state, iter_sym, init, cond, update, dbg_info);
            auto& body = std::get<1>(loop);

            // Define body
            auto& output_node = this->builder_.add_access(body, output, dbg_info);
            auto& tasklet = this->builder_.add_tasklet(body, operation, "__out", {"_in1", "_in2"}, dbg_info);

            sdfg::data_flow::AccessNode* left_node;
            auto left_input_type = utils::get_type(
                this->builder_,
                this->anonymous_types_mapping_,
                this->DL_,
                left_operand->getType(),
                utils::get_storage_type(this->target_type_, 0)
            );
            if (utils::is_literal(left_operand)) {
                std::string input = utils::as_initializer(llvm::dyn_cast<llvm::ConstantData>(left_operand));
                left_node = &this->builder_.add_constant(body, input, *left_input_type, dbg_info);
            } else {
                std::string left_input = utils::find_const_name_to_sdfg_name(this->constants_mapping_, left_operand);
                left_node = &this->builder_.add_access(body, left_input, dbg_info);
            }

            sdfg::data_flow::AccessNode* right_node;
            auto right_input_type = utils::get_type(
                this->builder_,
                this->anonymous_types_mapping_,
                this->DL_,
                right_operand->getType(),
                utils::get_storage_type(this->target_type_, 0)
            );
            if (utils::is_literal(right_operand)) {
                std::string input = utils::as_initializer(llvm::dyn_cast<llvm::ConstantData>(right_operand));
                right_node = &this->builder_.add_constant(body, input, *right_input_type, dbg_info);
            } else if (left_operand != right_operand) {
                std::string input = utils::find_const_name_to_sdfg_name(this->constants_mapping_, right_operand);
                right_node = &this->builder_.add_access(body, input, dbg_info);
            } else {
                right_node = left_node;
            }

            this->builder_
                .add_computational_memlet(body, tasklet, "__out", output_node, {iter_sym}, output_type, dbg_info);
            this->builder_
                .add_computational_memlet(body, *left_node, tasklet, "_in1", {iter_sym}, *left_input_type, dbg_info);
            this->builder_
                .add_computational_memlet(body, *right_node, tasklet, "_in2", {iter_sym}, *right_input_type, dbg_info);

            return std::get<2>(loop);
        }
        default: {
            throw NotImplementedException(
                "BinaryOperator: Unsupported output type",
                ::docc::utils::get_debug_info(*instruction),
                ::docc::utils::toIRString(*instruction)
            );
        }
    }
};

sdfg::control_flow::State& Lifting::visit_CastInst(
    const llvm::BasicBlock* block, const llvm::CastInst* instruction, sdfg::control_flow::State& current_state
) {
    // Define Debug
    auto dbg_info = ::docc::utils::get_debug_info(*instruction);

    // Define Output
    std::string output = utils::get_name(instruction);
    if (!this->builder_.subject().exists(output)) {
        auto output_type = utils::get_type(
            this->builder_,
            this->anonymous_types_mapping_,
            this->DL_,
            instruction->getType(),
            utils::get_storage_type(this->target_type_, 0)
        );
        this->builder_.add_container(output, *output_type);
    }
    auto& output_type = this->builder_.subject().type(output);
    assert(
        (output_type.type_id() == sdfg::types::TypeID::Scalar ||
         output_type.type_id() == sdfg::types::TypeID::Structure ||
         output_type.type_id() == sdfg::types::TypeID::Pointer) &&
        "CastInst: Expected scalar, structure or pointer type as output"
    );

    // Define Input
    auto input_operand = instruction->getOperand(0);
    auto input_type = utils::get_type(
        this->builder_,
        this->anonymous_types_mapping_,
        this->DL_,
        input_operand->getType(),
        utils::get_storage_type(this->target_type_, 0)
    );
    assert(
        (input_type->type_id() == sdfg::types::TypeID::Scalar ||
         input_type->type_id() == sdfg::types::TypeID::Structure ||
         input_type->type_id() == sdfg::types::TypeID::Pointer) &&
        "CastInst: Expected scalar, structure or pointer type as input"
    );

    if (llvm::isa<llvm::AddrSpaceCastInst>(instruction)) {
        throw NotImplementedException(
            "CastInst: AddrSpaceCastInst is not supported",
            ::docc::utils::get_debug_info(*instruction),
            ::docc::utils::toIRString(*instruction)
        );
    } else if (llvm::isa<llvm::BitCastInst>(instruction)) {
        throw NotImplementedException(
            "CastInst: BitCastInst is not supported",
            ::docc::utils::get_debug_info(*instruction),
            ::docc::utils::toIRString(*instruction)
        );
    } else if (llvm::isa<llvm::IntToPtrInst>(instruction)) {
        throw NotImplementedException(
            "CastInst: IntToPtrInst is not supported",
            ::docc::utils::get_debug_info(*instruction),
            ::docc::utils::toIRString(*instruction)
        );
    }

    // Handle ptr to int separately
    if (llvm::isa<llvm::PtrToIntInst>(instruction)) {
        assert(
            input_type->type_id() == sdfg::types::TypeID::Pointer &&
            "CastInst: Expected pointer type as input for PtrToIntInst"
        );
        assert(
            output_type.type_id() == sdfg::types::TypeID::Scalar &&
            "CastInst: Expected scalar type as output for PtrToIntInst"
        );

        sdfg::symbolic::Symbol output_sym = sdfg::symbolic::symbol(output);
        sdfg::symbolic::Symbol input_sym = SymEngine::null;
        if (utils::is_literal(input_operand)) {
            std::string arg = utils::as_initializer(llvm::dyn_cast<llvm::ConstantData>(input_operand));
            input_sym = sdfg::symbolic::symbol(arg);
        } else {
            std::string input = utils::find_const_name_to_sdfg_name(this->constants_mapping_, input_operand);
            input_sym = sdfg::symbolic::symbol(input);
        }

        auto& next_state = this->builder_.add_state_after(current_state, false, dbg_info);
        this->builder_.add_edge(current_state, next_state, {{output_sym, input_sym}}, dbg_info);
        return next_state;
    }

    switch (output_type.type_id()) {
        case sdfg::types::TypeID::Scalar: {
            auto& scalar_type = static_cast<const sdfg::types::Scalar&>(output_type);
            auto& output_node = this->builder_.add_access(current_state, output, dbg_info);

            sdfg::data_flow::AccessNode* input_node;
            if (utils::is_literal(input_operand)) {
                std::string arg = utils::as_literal(llvm::dyn_cast<llvm::ConstantData>(input_operand));
                input_node = &this->builder_.add_constant(current_state, arg, *input_type, dbg_info);
            } else {
                std::string input = utils::find_const_name_to_sdfg_name(this->constants_mapping_, input_operand);
                input_node = &this->builder_.add_access(current_state, input, dbg_info);
            }

            // We implement the cast by chosing the memlet types appropriately
            sdfg::types::PrimitiveType in_cast_type;
            sdfg::types::PrimitiveType out_cast_type;
            switch (instruction->getOpcode()) {
                case llvm::Instruction::CastOps::Trunc: {
                    auto* TI = llvm::cast<llvm::TruncInst>(instruction);
                    // truncated bits are all zero
                    bool nuw = TI->hasNoUnsignedWrap();
                    // truncated bits are all sign bits
                    bool nsw = TI->hasNoSignedWrap();

                    if (nuw && nsw) {
                        // truncated bits are all zero and sign bit == 0
                        // result fits into smaller signed type
                        // cast between signed types
                        // int64 _in = 42
                        // int32 _out = (int32) _in
                        in_cast_type = sdfg::types::as_signed(input_type->primitive_type());
                        out_cast_type = sdfg::types::as_signed(output_type.primitive_type());
                    } else {
                        // Truncation: cast between unsigned types (throw away high bits)
                        // int64 _in = 42
                        // int32 _out = (uint32) _in
                        in_cast_type = sdfg::types::as_unsigned(input_type->primitive_type());
                        out_cast_type = sdfg::types::as_unsigned(output_type.primitive_type());
                    }

                    auto& tasklet =
                        this->builder_
                            .add_tasklet(current_state, sdfg::data_flow::TaskletCode::assign, "__out", {"_in"}, dbg_info);

                    auto output_cast_type = std::make_unique<sdfg::types::Scalar>(out_cast_type);
                    this->builder_.add_computational_memlet(
                        current_state, tasklet, "__out", output_node, {}, *output_cast_type, dbg_info
                    );

                    auto input_cast_type = std::make_unique<sdfg::types::Scalar>(in_cast_type);
                    this->builder_.add_computational_memlet(
                        current_state, *input_node, tasklet, "_in", {}, *input_cast_type, dbg_info
                    );
                    break;
                }
                case llvm::Instruction::CastOps::ZExt: {
                    auto* ZI = llvm::dyn_cast<llvm::ZExtInst>(instruction);
                    assert(ZI && "ZExtInst expected");
                    if (!ZI->hasNonNeg()) {
                        // Zero extension: cast between unsigned types
                        // uint32 _in = 42
                        in_cast_type = sdfg::types::as_unsigned(input_type->primitive_type());
                        out_cast_type = sdfg::types::as_unsigned(output_type.primitive_type());
                    } else {
                        // Zero extension with non-neq
                        // Equivalent to cast with signed types
                        // -> aggressive optimization later
                        in_cast_type = sdfg::types::as_signed(input_type->primitive_type());
                        out_cast_type = sdfg::types::as_signed(output_type.primitive_type());
                    }

                    auto& tasklet =
                        this->builder_
                            .add_tasklet(current_state, sdfg::data_flow::TaskletCode::assign, "__out", {"_in"}, dbg_info);

                    auto output_cast_type = std::make_unique<sdfg::types::Scalar>(out_cast_type);
                    this->builder_.add_computational_memlet(
                        current_state, tasklet, "__out", output_node, {}, *output_cast_type, dbg_info
                    );

                    auto input_cast_type = std::make_unique<sdfg::types::Scalar>(in_cast_type);
                    this->builder_.add_computational_memlet(
                        current_state, *input_node, tasklet, "_in", {}, *input_cast_type, dbg_info
                    );
                    break;
                }
                case llvm::Instruction::CastOps::SExt: {
                    // Sign extension: cast between signed types
                    // int32 _in = -42
                    in_cast_type = sdfg::types::as_signed(input_type->primitive_type());
                    out_cast_type = sdfg::types::as_signed(output_type.primitive_type());

                    if (input_type->primitive_type() == sdfg::types::PrimitiveType::Bool) {
                        auto& tasklet = this->builder_.add_tasklet(
                            current_state, sdfg::data_flow::TaskletCode::int_mul, "__out", {"_in1", "_in2"}, dbg_info
                        );

                        auto& minus_one_node =
                            this->builder_
                                .add_constant(current_state, "-1", sdfg::types::Scalar(out_cast_type), dbg_info);
                        this->builder_.add_computational_memlet(
                            current_state,
                            minus_one_node,
                            tasklet,
                            "_in2",
                            {},
                            sdfg::types::Scalar(out_cast_type),
                            dbg_info
                        );

                        auto output_cast_type = std::make_unique<sdfg::types::Scalar>(out_cast_type);
                        this->builder_.add_computational_memlet(
                            current_state, tasklet, "__out", output_node, {}, *output_cast_type, dbg_info
                        );

                        auto input_cast_type = std::make_unique<sdfg::types::Scalar>(in_cast_type);
                        this->builder_.add_computational_memlet(
                            current_state, *input_node, tasklet, "_in1", {}, *input_cast_type, dbg_info
                        );
                    } else {
                        auto& tasklet = this->builder_.add_tasklet(
                            current_state, sdfg::data_flow::TaskletCode::assign, "__out", {"_in"}, dbg_info
                        );

                        auto output_cast_type = std::make_unique<sdfg::types::Scalar>(out_cast_type);
                        this->builder_.add_computational_memlet(
                            current_state, tasklet, "__out", output_node, {}, *output_cast_type, dbg_info
                        );

                        auto input_cast_type = std::make_unique<sdfg::types::Scalar>(in_cast_type);
                        this->builder_.add_computational_memlet(
                            current_state, *input_node, tasklet, "_in", {}, *input_cast_type, dbg_info
                        );
                    }
                    break;
                }
                case llvm::Instruction::CastOps::FPToUI: {
                    // Floating point to unsigned integer
                    // float32 _in = 42.0
                    in_cast_type = input_type->primitive_type();
                    out_cast_type = sdfg::types::as_unsigned(output_type.primitive_type());

                    auto& tasklet =
                        this->builder_
                            .add_tasklet(current_state, sdfg::data_flow::TaskletCode::assign, "__out", {"_in"}, dbg_info);

                    auto output_cast_type = std::make_unique<sdfg::types::Scalar>(out_cast_type);
                    this->builder_.add_computational_memlet(
                        current_state, tasklet, "__out", output_node, {}, *output_cast_type, dbg_info
                    );

                    auto input_cast_type = std::make_unique<sdfg::types::Scalar>(in_cast_type);
                    this->builder_.add_computational_memlet(
                        current_state, *input_node, tasklet, "_in", {}, *input_cast_type, dbg_info
                    );
                    break;
                }
                case llvm::Instruction::CastOps::FPToSI: {
                    // Floating point to signed integer
                    // float32 _in = 42.0
                    in_cast_type = input_type->primitive_type();
                    out_cast_type = sdfg::types::as_signed(output_type.primitive_type());

                    auto& tasklet =
                        this->builder_
                            .add_tasklet(current_state, sdfg::data_flow::TaskletCode::assign, "__out", {"_in"}, dbg_info);

                    auto output_cast_type = std::make_unique<sdfg::types::Scalar>(out_cast_type);
                    this->builder_.add_computational_memlet(
                        current_state, tasklet, "__out", output_node, {}, *output_cast_type, dbg_info
                    );

                    auto input_cast_type = std::make_unique<sdfg::types::Scalar>(in_cast_type);
                    this->builder_.add_computational_memlet(
                        current_state, *input_node, tasklet, "_in", {}, *input_cast_type, dbg_info
                    );
                    break;
                }
                case llvm::Instruction::CastOps::UIToFP: {
                    auto* UIFP = llvm::cast<llvm::UIToFPInst>(instruction);
                    assert(UIFP && "UIToFPInst expected");

                    if (UIFP->hasNonNeg()) {
                        // Unsigned integer to floating point with non-neg
                        // uint32 _in = 42
                        in_cast_type = sdfg::types::as_signed(input_type->primitive_type());
                        out_cast_type = output_type.primitive_type();
                    } else {
                        // Unsigned integer to floating point
                        // uint32 _in = 42
                        in_cast_type = sdfg::types::as_unsigned(input_type->primitive_type());
                        out_cast_type = output_type.primitive_type();
                    }

                    auto& tasklet =
                        this->builder_
                            .add_tasklet(current_state, sdfg::data_flow::TaskletCode::assign, "__out", {"_in"}, dbg_info);

                    auto output_cast_type = std::make_unique<sdfg::types::Scalar>(out_cast_type);
                    this->builder_.add_computational_memlet(
                        current_state, tasklet, "__out", output_node, {}, *output_cast_type, dbg_info
                    );

                    auto input_cast_type = std::make_unique<sdfg::types::Scalar>(in_cast_type);
                    this->builder_.add_computational_memlet(
                        current_state, *input_node, tasklet, "_in", {}, *input_cast_type, dbg_info
                    );
                    break;
                }
                case llvm::Instruction::CastOps::SIToFP: {
                    // Signed integer to floating point
                    // int32 _in = -42
                    in_cast_type = sdfg::types::as_signed(input_type->primitive_type());
                    out_cast_type = output_type.primitive_type();

                    auto& tasklet =
                        this->builder_
                            .add_tasklet(current_state, sdfg::data_flow::TaskletCode::assign, "__out", {"_in"}, dbg_info);

                    auto output_cast_type = std::make_unique<sdfg::types::Scalar>(out_cast_type);
                    this->builder_.add_computational_memlet(
                        current_state, tasklet, "__out", output_node, {}, *output_cast_type, dbg_info
                    );

                    auto input_cast_type = std::make_unique<sdfg::types::Scalar>(in_cast_type);
                    this->builder_.add_computational_memlet(
                        current_state, *input_node, tasklet, "_in", {}, *input_cast_type, dbg_info
                    );
                    break;
                }
                case llvm::Instruction::CastOps::FPTrunc: {
                    // Floating point truncation
                    // float64 _in = 42.0
                    in_cast_type = input_type->primitive_type();
                    out_cast_type = output_type.primitive_type();

                    auto& tasklet =
                        this->builder_
                            .add_tasklet(current_state, sdfg::data_flow::TaskletCode::assign, "__out", {"_in"}, dbg_info);

                    auto output_cast_type = std::make_unique<sdfg::types::Scalar>(out_cast_type);
                    this->builder_.add_computational_memlet(
                        current_state, tasklet, "__out", output_node, {}, *output_cast_type, dbg_info
                    );

                    auto input_cast_type = std::make_unique<sdfg::types::Scalar>(in_cast_type);
                    this->builder_.add_computational_memlet(
                        current_state, *input_node, tasklet, "_in", {}, *input_cast_type, dbg_info
                    );
                    break;
                }
                case llvm::Instruction::CastOps::FPExt: {
                    // Floating point extension
                    // float32 _in = 42.0
                    in_cast_type = input_type->primitive_type();
                    out_cast_type = output_type.primitive_type();

                    auto& tasklet =
                        this->builder_
                            .add_tasklet(current_state, sdfg::data_flow::TaskletCode::assign, "__out", {"_in"}, dbg_info);

                    auto output_cast_type = std::make_unique<sdfg::types::Scalar>(out_cast_type);
                    this->builder_.add_computational_memlet(
                        current_state, tasklet, "__out", output_node, {}, *output_cast_type, dbg_info
                    );

                    auto input_cast_type = std::make_unique<sdfg::types::Scalar>(in_cast_type);
                    this->builder_.add_computational_memlet(
                        current_state, *input_node, tasklet, "_in", {}, *input_cast_type, dbg_info
                    );
                    break;
                }
                default: {
                    throw NotImplementedException(
                        "CastInst: Unsupported cast operation",
                        ::docc::utils::get_debug_info(*instruction),
                        ::docc::utils::toIRString(*instruction)
                    );
                }
            }
            return current_state;
        }
        case sdfg::types::TypeID::Structure: {
            auto& struct_type = static_cast<const sdfg::types::Structure&>(output_type);
            auto& struct_def = this->builder_.subject().structure(struct_type.name());
            assert(struct_def.is_vector() && "CastInst: Expected vector structure type as output");
            auto& vec_elem_type = struct_def.vector_element_type();

            // Define loop
            std::string iterator = this->builder_.find_new_name("_i");
            this->builder_.add_container(iterator, sdfg::types::Scalar(sdfg::types::PrimitiveType::Int64));
            sdfg::symbolic::Symbol iter_sym = sdfg::symbolic::symbol(iterator);
            sdfg::symbolic::Expression init = sdfg::symbolic::zero();
            sdfg::symbolic::Condition cond =
                sdfg::symbolic::Lt(iter_sym, sdfg::symbolic::integer(struct_def.vector_size()));
            sdfg::symbolic::Expression update = SymEngine::add(iter_sym, sdfg::symbolic::one());

            auto loop = this->builder_.add_loop(current_state, iter_sym, init, cond, update, dbg_info);
            auto& body = std::get<1>(loop);

            // Define body
            auto& output_node = this->builder_.add_access(body, output, dbg_info);
            auto& tasklet =
                this->builder_.add_tasklet(body, sdfg::data_flow::TaskletCode::assign, "__out", {"_in"}, dbg_info);

            sdfg::data_flow::AccessNode* input_node;
            if (utils::is_literal(input_operand)) {
                std::string arg = utils::as_initializer(llvm::dyn_cast<llvm::ConstantData>(input_operand));
                input_node = &this->builder_.add_constant(body, arg, *input_type, dbg_info);
            } else {
                std::string input = utils::find_const_name_to_sdfg_name(this->constants_mapping_, input_operand);
                input_node = &this->builder_.add_access(body, input, dbg_info);
            }

            // We implement the cast by chosing the memlet types appropriately
            sdfg::types::PrimitiveType in_cast_type;
            sdfg::types::PrimitiveType out_cast_type;
            switch (instruction->getOpcode()) {
                case llvm::Instruction::CastOps::Trunc: {
                    // Truncation: cast between unsigned types
                    // int64 _in = 42
                    // int32 _out = (uint32) _in
                    in_cast_type = sdfg::types::as_unsigned(input_type->primitive_type());
                    out_cast_type = sdfg::types::as_unsigned(vec_elem_type.primitive_type());
                    break;
                }
                case llvm::Instruction::CastOps::ZExt: {
                    // Zero extension: cast between unsigned types
                    // uint32 _in = 42
                    in_cast_type = sdfg::types::as_unsigned(input_type->primitive_type());
                    out_cast_type = sdfg::types::as_unsigned(vec_elem_type.primitive_type());
                    break;
                }
                case llvm::Instruction::CastOps::SExt: {
                    // Sign extension: cast between signed types
                    // int32 _in = -42
                    in_cast_type = sdfg::types::as_signed(input_type->primitive_type());
                    out_cast_type = sdfg::types::as_signed(vec_elem_type.primitive_type());
                    break;
                }
                case llvm::Instruction::CastOps::FPToUI: {
                    // Floating point to unsigned integer
                    // float32 _in = 42.0
                    in_cast_type = input_type->primitive_type();
                    out_cast_type = sdfg::types::as_unsigned(vec_elem_type.primitive_type());
                    break;
                }
                case llvm::Instruction::CastOps::FPToSI: {
                    // Floating point to signed integer
                    // float32 _in = 42.0
                    in_cast_type = input_type->primitive_type();
                    out_cast_type = sdfg::types::as_signed(vec_elem_type.primitive_type());
                    break;
                }
                case llvm::Instruction::CastOps::UIToFP: {
                    // Unsigned integer to floating point
                    // uint32 _in = 42
                    in_cast_type = sdfg::types::as_unsigned(input_type->primitive_type());
                    out_cast_type = vec_elem_type.primitive_type();
                    break;
                }
                case llvm::Instruction::CastOps::SIToFP: {
                    // Signed integer to floating point
                    // int32 _in = -42
                    in_cast_type = sdfg::types::as_signed(input_type->primitive_type());
                    out_cast_type = vec_elem_type.primitive_type();
                    break;
                }
                case llvm::Instruction::CastOps::FPTrunc: {
                    // Floating point truncation
                    // float64 _in = 42.0
                    in_cast_type = input_type->primitive_type();
                    out_cast_type = vec_elem_type.primitive_type();
                    break;
                }
                case llvm::Instruction::CastOps::FPExt: {
                    // Floating point extension
                    // float32 _in = 42.0
                    in_cast_type = input_type->primitive_type();
                    out_cast_type = vec_elem_type.primitive_type();
                    break;
                }
                default: {
                    throw NotImplementedException(
                        "CastInst: Unsupported cast operation",
                        ::docc::utils::get_debug_info(*instruction),
                        ::docc::utils::toIRString(*instruction)
                    );
                }
            }

            auto output_cast_type_scalar = std::make_unique<sdfg::types::Scalar>(out_cast_type);
            auto output_cast_type =
                this->builder_.create_vector_type(*output_cast_type_scalar, struct_def.vector_size());
            this->builder_
                .add_computational_memlet(body, tasklet, "__out", output_node, {iter_sym}, *output_cast_type, dbg_info);

            auto input_cast_type_scalar = std::make_unique<sdfg::types::Scalar>(in_cast_type);
            auto input_cast_type = this->builder_.create_vector_type(*input_cast_type_scalar, struct_def.vector_size());
            this->builder_
                .add_computational_memlet(body, *input_node, tasklet, "_in", {iter_sym}, *input_cast_type, dbg_info);

            return std::get<2>(loop);
        }
        default: {
            throw NotImplementedException(
                "CastInst: Unsupported output type",
                ::docc::utils::get_debug_info(*instruction),
                ::docc::utils::toIRString(*instruction)
            );
        }
    }
};


} // namespace lifting
} // namespace docc
