#include "docc/lifting/function_to_sdfg.h"

#include <llvm/IR/Function.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/CodeExtractor.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>

#include <docc/target/tenstorrent/math_node_implementation_override_pass.h>
#include <docc/target/tenstorrent/tenstorrent_transform.h>
#include <sdfg/analysis/analysis.h>
#include <sdfg/analysis/users.h>
#include <sdfg/builder/sdfg_builder.h>
#include <sdfg/helpers/helpers.h>
#include <sdfg/passes/debug_info_propagation.h>
#include <sdfg/passes/normalization/loop_normal_form.h>
#include <sdfg/passes/opt_pipeline.h>
#include <sdfg/passes/pipeline.h>
#include <sdfg/passes/schedules/expansion_pass.h>
#include <sdfg/passes/structured_control_flow/pointer_evolution.h>
#include <sdfg/passes/structured_control_flow/unify_loop_exits.h>
#include <sdfg/passes/structured_control_flow/while_to_for_conversion.h>
#include <sdfg/passes/symbolic/symbol_promotion.h>
#include <sdfg/passes/symbolic/type_minimization.h>

#include "docc/lifting/functions/function_lifting.h"
#include "docc/lifting/lift_report.h"
#include "docc/lifting/lifting.h"
#include "docc/utils.h"

llvm::cl::opt<std::string> DOCC_expand(
    "docc-expand",
    llvm::cl::desc("Overrides when to expand library nodes"),
    llvm::cl::init("none"),
    llvm::cl::value_desc("none|all")
);

namespace docc {
namespace lifting {

std::unique_ptr<llvm::Region> FunctionToSDFG::expand_region(std::unique_ptr<llvm::Region>& R) {
    std::unique_ptr<llvm::Region> current = std::move(R);
    std::unique_ptr<llvm::Region> last_valid = nullptr;
    while (current) {
        auto res = this->can_be_applied(*current);
        if (!res.first) break;

        last_valid = std::move(current);
        current = std::unique_ptr<llvm::Region>(last_valid->getExpandedRegion());
    }

    return last_valid;
}

bool FunctionToSDFG::is_blacklisted(llvm::Function& F, bool apply_on_linkonce_odr) {
    // Variadic functions
    if (F.isVarArg()) {
        return true;
    }

    // clang
    if (F.getName().starts_with("__clang")) {
        return true;
    }
    // c++, e.g., __cxx_global_var_init
    if (F.getName().starts_with("__cxx")) {
        return true;
    }
    // daisy, previously lifted
    if (F.getName().starts_with("__daisy")) {
        return true;
    }
    // global ctors
    if (F.hasSection() && F.getSection().contains("startup")) {
        return true;
    }
    // global dtors
    if (F.hasSection() && F.getSection().contains("exit")) {
        return true;
    }

    // aliased functions cannot be modified
    llvm::Module* Mod = F.getParent();
    for (const auto& alias : Mod->aliases()) {
        if (alias.getAliasee() == &F) {
            return true;
        }
    }

    // LinkOnceODR only if explicitly allowed
    if (apply_on_linkonce_odr) {
        if (F.getLinkage() == llvm::GlobalValue::LinkOnceODRLinkage) {
            return false;
        } else {
            return true;
        }
    }

    // Supported linkage: external, internal, private
    if (F.getLinkage() != llvm::GlobalValue::ExternalLinkage && F.getLinkage() != llvm::GlobalValue::InternalLinkage &&
        F.getLinkage() != llvm::GlobalValue::PrivateLinkage) {
        return true;
    }

    // No optnone, alwaysinline
    // if (F.hasFnAttribute(llvm::Attribute::OptimizeNone) ||
    //     F.hasFnAttribute(llvm::Attribute::AlwaysInline)) {
    //     return true;
    // }

    return false;
}

FunctionToSDFG::FunctionToSDFG(llvm::Function& function, llvm::FunctionAnalysisManager& FAM, bool apply_on_linkonce_odr)
    : function_(function), FAM_(FAM), sdfg_counter(0), apply_on_linkonce_odr_(apply_on_linkonce_odr) {}

std::vector<std::unique_ptr<sdfg::StructuredSDFG>> FunctionToSDFG::run() {
    auto& TLI = this->FAM_.getResult<llvm::TargetLibraryAnalysis>(this->function_);
    llvm::LibFunc lf;
    if (TLI.getLibFunc(this->function_.getName(), lf)) {
        return {};
    }

    // Attempt lifting the entire function
    auto res = this->can_be_applied();
    if (res.first) {
        try {
            auto sdfg = this->apply();
            if (sdfg) {
                std::vector<std::unique_ptr<sdfg::StructuredSDFG>> sdfgs;
                sdfgs.push_back(std::move(sdfg));
                return sdfgs;
            }
        } catch (sdfg::UnstructuredControlFlowException& e) {
            // Fallthrough
            LLVM_DEBUG_PRINTLN("UnstructuredControlFlowException on '" << this->function_.getName() << "': " << e.what());
        } catch (NotImplementedException& e) {
            // Fallthrough
            LLVM_DEBUG_PRINTLN("NotImplementedException on '" << this->function_.getName() << "': " << e.what());
        } catch (sdfg::InvalidSDFGException& e) {
            // Fallthrough
            LLVM_DEBUG_PRINTLN("InvalidSDFGException on '" << this->function_.getName() << "': " << e.what());
        }
    } else {
        if (res.second != nullptr) {
            sdfg::DebugInfo dbg_info;
            if (auto* inst = llvm::dyn_cast<llvm::Instruction>(res.second)) {
                dbg_info = ::docc::utils::get_debug_info(*inst);
            }
            LiftingReport::add_failed_lift(dbg_info, "Unsupported instruction", ::docc::utils::toIRString(*res.second));
        }

        // For LinkOnceODR, we must give up if the entire function cannot be lifted
        if (this->function_.getLinkage() == llvm::GlobalValue::LinkOnceODRLinkage) {
            return {};
        }
    }

    // Attempt lifting of Single-Entry-Single-Exit regions
    auto& RI = this->FAM_.getResult<llvm::RegionInfoAnalysis>(this->function_);
    std::list<std::unique_ptr<llvm::Region>> canonical_regions;
    for (auto& sub : *RI.getTopLevelRegion()) {
        canonical_regions.push_back(std::move(sub));
    }

    std::vector<std::unique_ptr<sdfg::StructuredSDFG>> sdfgs;
    while (!canonical_regions.empty()) {
        auto canon_region = std::move(canonical_regions.front());
        canonical_regions.pop_front();

        auto expanded_region = this->expand_region(canon_region);
        if (!expanded_region) {
            continue;
        }

        // Collect subregions
        std::list<llvm::Region*> subregions;
        for (auto& sub : canonical_regions) {
            if (expanded_region->contains(sub.get())) {
                subregions.push_back(sub.get());
            }
        }

        if (FunctionToSDFG::loop_count(this->function_, *expanded_region, this->FAM_) < 2) {
            for (auto& subregion : subregions) {
                for (auto it = canonical_regions.begin(); it != canonical_regions.end(); ++it) {
                    if (it->get() == subregion) {
                        canonical_regions.erase(it);
                        break;
                    }
                }
            }
            continue;
        }

        try {
            auto sdfg = this->apply(*expanded_region);
            if (sdfg) {
                sdfgs.push_back(std::move(sdfg));
                for (auto& subregion : subregions) {
                    for (auto it = canonical_regions.begin(); it != canonical_regions.end(); ++it) {
                        if (it->get() == subregion) {
                            canonical_regions.erase(it);
                            break;
                        }
                    }
                }
                continue;
            }
        } catch (sdfg::UnstructuredControlFlowException& e) {
            // Fallthrough
        } catch (NotImplementedException& e) {
            // Fallthrough
        } catch (sdfg::InvalidSDFGException& e) {
            // Fallthrough
        }

        // Region failed, give up
        break;
    }

    return sdfgs;
}

std::pair<bool, llvm::Value*> FunctionToSDFG::can_be_applied(llvm::Region& region) {
    // Criterion: Regions must be extractable into functions
    llvm::SmallVector<llvm::BasicBlock*> blocks;
    for (auto block : region.blocks()) {
        blocks.push_back(block);
    }
    llvm::CodeExtractor code_extractor(
        blocks,
        nullptr, // DT
        false, // Aggregate args
        nullptr, // BFI
        nullptr, // BPI
        nullptr, // AC
        false, // Var args
        false, // Allow alloca
        nullptr, // Alloc block
        "" // Suffix
    );
    if (!code_extractor.isEligible()) {
        return {false, nullptr};
    }

    // Criterion: No unsupported instructions
    auto& TLI = this->FAM_.getResult<llvm::TargetLibraryAnalysis>(this->function_);
    for (auto block : region.blocks()) {
        for (auto& inst : *block) {
            // TODO: Switch
            if (llvm::dyn_cast<const llvm::SwitchInst>(&inst)) {
                return {false, &inst};
            }
            if (auto phi_inst = llvm::dyn_cast<const llvm::PHINode>(&inst)) {
                for (size_t i = 0; i < phi_inst->getNumIncomingValues(); ++i) {
                    auto phi_value = phi_inst->getIncomingValue(i);
                    // Must be first-class type: scalar or pointer
                    if (phi_value->getType()->isVectorTy() || phi_value->getType()->isAggregateType()) {
                        return {false, &inst};
                    }
                }
            }

            // Poison
            if (llvm::dyn_cast<const llvm::FreezeInst>(&inst)) {
                return {false, &inst};
            }

            // Unsafe casts
            if (llvm::isa<llvm::AddrSpaceCastInst>(&inst)) {
                return {false, &inst};
            } else if (llvm::isa<llvm::BitCastInst>(&inst)) {
                return {false, &inst};
            } else if (llvm::isa<llvm::IntToPtrInst>(&inst)) {
                return {false, &inst};
            }

            // Atomic operations
            if (llvm::isa<const llvm::AtomicRMWInst>(&inst)) {
                return {false, &inst};
            } else if (llvm::isa<const llvm::AtomicCmpXchgInst>(&inst)) {
                return {false, &inst};
            } else if (llvm::isa<const llvm::FenceInst>(&inst)) {
                return {false, &inst};
            }

            // Function calls
            if (auto call_base = llvm::dyn_cast<const llvm::CallBase>(&inst)) {
                if (!FunctionLifting::is_supported(TLI, call_base)) {
                    return {false, &inst};
                }
            }
            if (auto landing_pad = llvm::dyn_cast<const llvm::LandingPadInst>(&inst)) {
                return {false, &inst};
            }
            if (auto resume = llvm::dyn_cast<const llvm::ResumeInst>(&inst)) {
                return {false, &inst};
            }

            // Constant expressions
            for (const llvm::Use& U : inst.operands()) {
                if (llvm::isa<llvm::ConstantExpr>(U.get())) {
                    return {false, &inst};
                }
            }

            // Not implemented instructions
            if (llvm::isa<const llvm::ExtractValueInst>(&inst)) {
                return {false, &inst};
            }
            if (llvm::isa<const llvm::InsertValueInst>(&inst)) {
                return {false, &inst};
            }
            if (llvm::isa<const llvm::InsertElementInst>(&inst)) {
                return {false, &inst};
            }

            // Unsupported types
            auto output_type = inst.getType();
            if (output_type->isIntegerTy()) {
                switch (output_type->getIntegerBitWidth()) {
                    case 1:
                    case 8:
                    case 16:
                    case 32:
                    case 64:
                    case 128:
                        break;
                    default: {
                        return {false, &inst};
                    }
                }
            }
        }
        llvm::Instruction* terminator = block->getTerminator();
        if (!llvm::isa<llvm::UnreachableInst>(terminator) && !llvm::isa<llvm::ReturnInst>(terminator) &&
            !llvm::isa<llvm::BranchInst>(terminator) && !llvm::isa<llvm::InvokeInst>(terminator)) {
            return {false, terminator};
        }
    }

    return {true, nullptr};
}

std::pair<bool, llvm::Value*> FunctionToSDFG::can_be_applied() {
    auto& TLI = this->FAM_.getResult<llvm::TargetLibraryAnalysis>(this->function_);

    // Criterion: No unsupported globals' initializers
    std::unordered_set<llvm::GlobalObject*> globals;
    Lifting::collect_globals(this->function_, globals);
    for (llvm::GlobalObject* GV : globals) {
        switch (GV->getLinkage()) {
            case llvm::GlobalValue::LinkageTypes::ExternalLinkage:
            case llvm::GlobalValue::LinkageTypes::AvailableExternallyLinkage:
            case llvm::GlobalValue::LinkageTypes::LinkOnceAnyLinkage:
            case llvm::GlobalValue::LinkageTypes::LinkOnceODRLinkage:
            case llvm::GlobalValue::LinkageTypes::WeakAnyLinkage:
            case llvm::GlobalValue::LinkageTypes::WeakODRLinkage: {
                // Always allowed
                continue;
            }
            case llvm::GlobalValue::LinkageTypes::InternalLinkage: {
                // Renamed to globally unique and set to external linkage
                // Thus, initializer need not be lifted
                continue;
            }
            case llvm::GlobalValue::LinkageTypes::PrivateLinkage: {
                // Renamed to globally unique and set to external linkage
                // Thus, initializer need not be lifted
                continue;
            }
            default:
                return {false, GV};
        }
    }

    // Criterion: No unsupported instructions
    for (auto& block : this->function_) {
        for (auto& inst : block) {
            // TODO: Switch
            if (llvm::dyn_cast<const llvm::SwitchInst>(&inst)) {
                return {false, &inst};
            }
            if (auto phi_inst = llvm::dyn_cast<const llvm::PHINode>(&inst)) {
                for (size_t i = 0; i < phi_inst->getNumIncomingValues(); ++i) {
                    auto phi_value = phi_inst->getIncomingValue(i);
                    // Must be first-class type: scalar or pointer
                    if (phi_value->getType()->isVectorTy() || phi_value->getType()->isAggregateType()) {
                        return {false, &inst};
                    }
                }
            }

            // Poison
            if (llvm::dyn_cast<const llvm::FreezeInst>(&inst)) {
                return {false, &inst};
            }

            // Unsafe casts
            if (llvm::isa<llvm::AddrSpaceCastInst>(&inst)) {
                return {false, &inst};
            } else if (llvm::isa<llvm::BitCastInst>(&inst)) {
                return {false, &inst};
            } else if (llvm::isa<llvm::IntToPtrInst>(&inst)) {
                return {false, &inst};
            }

            // Atomic operations
            if (llvm::isa<const llvm::AtomicRMWInst>(&inst)) {
                return {false, &inst};
            } else if (llvm::isa<const llvm::AtomicCmpXchgInst>(&inst)) {
                return {false, &inst};
            } else if (llvm::isa<const llvm::FenceInst>(&inst)) {
                return {false, &inst};
            }

            // Function calls
            if (auto call_base = llvm::dyn_cast<const llvm::CallBase>(&inst)) {
                if (!FunctionLifting::is_supported(TLI, call_base)) {
                    return {false, &inst};
                }
            }
            if (auto landing_pad = llvm::dyn_cast<const llvm::LandingPadInst>(&inst)) {
                return {false, &inst};
            }
            if (auto resume = llvm::dyn_cast<const llvm::ResumeInst>(&inst)) {
                return {false, &inst};
            }

            // Constant expressions
            for (const llvm::Use& U : inst.operands()) {
                if (llvm::isa<llvm::ConstantExpr>(U.get())) {
                    return {false, &inst};
                }
            }

            // Not implemented instructions
            if (llvm::isa<const llvm::ExtractValueInst>(&inst)) {
                return {false, &inst};
            }
            if (llvm::isa<const llvm::InsertValueInst>(&inst)) {
                return {false, &inst};
            }
            if (llvm::isa<const llvm::InsertElementInst>(&inst)) {
                return {false, &inst};
            }

            // Unsupported types
            auto output_type = inst.getType();
            if (output_type->isIntegerTy()) {
                switch (output_type->getIntegerBitWidth()) {
                    case 1:
                    case 8:
                    case 16:
                    case 32:
                    case 64:
                    case 128:
                        break;
                    default: {
                        return {false, &inst};
                    }
                }
            }
        }
        llvm::Instruction* terminator = block.getTerminator();
        if (!llvm::isa<llvm::UnreachableInst>(terminator) && !llvm::isa<llvm::ReturnInst>(terminator) &&
            !llvm::isa<llvm::BranchInst>(terminator) && !llvm::isa<llvm::InvokeInst>(terminator)) {
            return {false, terminator};
        }
    }

    return {true, nullptr};
}

std::unique_ptr<sdfg::StructuredSDFG> FunctionToSDFG::apply(llvm::Region& region) {
    std::filesystem::path module_path = this->function_.getParent()->getName().str();
    std::string module_name = module_path.stem();

    // Refactor region into separate function
    llvm::SmallVector<llvm::BasicBlock*> blocks;
    for (auto block : region.blocks()) {
        blocks.push_back(block);
    }

    llvm::CodeExtractor code_extractor(
        blocks,
        nullptr, // DT
        false, // Aggregate args
        nullptr, // BFI
        nullptr, // BPI
        nullptr, // AC
        false, // Var args
        false, // Allow alloca
        nullptr, // Alloc block
        "" // Suffix
    );
    assert(code_extractor.isEligible());
    llvm::CodeExtractorAnalysisCache CEAC(this->function_);
    llvm::Function* external_function = code_extractor.extractCodeRegion(CEAC);

    // Set name of new function
    std::string new_function_name = ::docc::utils::hash_function_name(utils::get_name(external_function));
    new_function_name = utils::normalize_name(module_name) + utils::normalize_name(new_function_name);
    new_function_name = "__daisy_" + new_function_name + "_" + std::to_string(sdfg_counter++);
    external_function->setName(new_function_name);
    external_function->setLinkage(this->function_.getLinkage());

    // Criterion: No unsupported globals' initializers
    std::unordered_set<llvm::GlobalObject*> globals;
    Lifting::collect_globals(*external_function, globals);
    for (llvm::GlobalObject* GV : globals) {
        switch (GV->getLinkage()) {
            case llvm::GlobalValue::LinkageTypes::ExternalLinkage:
            case llvm::GlobalValue::LinkageTypes::AvailableExternallyLinkage:
            case llvm::GlobalValue::LinkageTypes::LinkOnceAnyLinkage:
            case llvm::GlobalValue::LinkageTypes::LinkOnceODRLinkage:
            case llvm::GlobalValue::LinkageTypes::WeakAnyLinkage:
            case llvm::GlobalValue::LinkageTypes::WeakODRLinkage: {
                // Always allowed
                continue;
            }
            case llvm::GlobalValue::LinkageTypes::InternalLinkage: {
                // Renamed to globally unique and set to external linkage
                // Thus, initializer need not be lifted
                continue;
            }
            case llvm::GlobalValue::LinkageTypes::PrivateLinkage: {
                // Renamed to globally unique and set to external linkage
                // Thus, initializer need not be lifted
                continue;
            }
            default:
                return nullptr;
        }
    }

    // Lift SDFG
    auto& TLI = this->FAM_.getResult<llvm::TargetLibraryAnalysis>(this->function_);
    Lifting lifting(TLI, *external_function, sdfg::FunctionType_CPU);
    std::unique_ptr<sdfg::SDFG> sdfg = lifting.run();
    sdfg->validate();

    sdfg::builder::SDFGBuilder builder_canon(sdfg);
    sdfg::passes::UnifyLoopExits unify_loop_exits_pass;
    unify_loop_exits_pass.run(builder_canon);
    unify_loop_exits_pass.run(builder_canon);
    unify_loop_exits_pass.run(builder_canon);
    sdfg = builder_canon.move();

    // Build StructuredSDFG
    sdfg::builder::StructuredSDFGBuilder structured_builder(*sdfg);

    // Propagate debug info
    sdfg::analysis::AnalysisManager analysis_manager(structured_builder.subject());
    sdfg::passes::DebugInfoPropagation debug_info_propagation_pass;
    debug_info_propagation_pass.run(structured_builder, analysis_manager);

    LiftingReport::add_successful_lift(::docc::utils::get_debug_info(*external_function));
    auto structured_sdfg = structured_builder.move();

    // Simplify SDFG
    auto simplified_sdfg = this->simplify(structured_sdfg);

    // Prevent further inlining
    external_function->addFnAttr(llvm::Attribute::NoInline);
    external_function->addFnAttr(llvm::Attribute::OptimizeNone);
    llvm::appendToUsed(*this->function_.getParent(), {external_function});

    bool verify_dbg = false;
    bool failed = llvm::verifyModule(*this->function_.getParent(), &llvm::errs(), &verify_dbg);
    if (failed) {
        throw sdfg::InvalidSDFGException("Module is broken after lifting region.");
    }

    return simplified_sdfg;
}

std::unique_ptr<sdfg::StructuredSDFG> FunctionToSDFG::apply() {
    // Lift SDFG
    auto& TLI = this->FAM_.getResult<llvm::TargetLibraryAnalysis>(this->function_);
    Lifting lifting(TLI, this->function_, sdfg::FunctionType_CPU);
    std::unique_ptr<sdfg::SDFG> sdfg = lifting.run();
    sdfg->validate();

    // Increase of graph complexity
    sdfg::builder::SDFGBuilder builder_canon(sdfg);
    sdfg::passes::UnifyLoopExits unify_loop_exits_pass;
    unify_loop_exits_pass.run(builder_canon);
    unify_loop_exits_pass.run(builder_canon);
    unify_loop_exits_pass.run(builder_canon);
    sdfg = builder_canon.move();

    // Build StructuredSDFG
    sdfg::builder::StructuredSDFGBuilder structured_builder(*sdfg);

    // Propagate debug info
    sdfg::analysis::AnalysisManager analysis_manager(structured_builder.subject());
    sdfg::passes::DebugInfoPropagation debug_info_propagation_pass;
    debug_info_propagation_pass.run(structured_builder, analysis_manager);

    LiftingReport::add_successful_lift(::docc::utils::get_debug_info(this->function_));
    auto structured_sdfg = structured_builder.move();

    // Simplify SDFG
    auto simplified_sdfg = this->simplify(structured_sdfg);

    // If LinkOnceODR, internalize original function
    if (this->function_.getLinkage() == llvm::GlobalValue::LinkOnceODRLinkage) {
        std::string module_path = this->function_.getParent()->getName().str();
        std::string module_name = std::filesystem::path(module_path).stem().string();

        llvm::Function* internal_function = llvm::Function::Create(
            this->function_.getFunctionType(),
            llvm::GlobalValue::ExternalLinkage,
            "__daisy_odr_" + utils::normalize_name(module_name) + "_" + this->function_.getName(),
            this->function_.getParent()
        );
        internal_function->copyAttributesFrom(&this->function_);
        internal_function->setComdat(nullptr);

        // Map arguments and the function itself for recursion
        llvm::ValueToValueMapTy VMap;
        auto dest_arg_it = internal_function->arg_begin();
        for (auto& src_arg : this->function_.args()) {
            dest_arg_it->setName(src_arg.getName());
            VMap[&src_arg] = &*dest_arg_it++;
        }
        VMap[&this->function_] = internal_function; // Handle recursive calls

        llvm::SmallVector<llvm::ReturnInst*, 8> Returns;
        // Clone the function body
        llvm::CloneFunctionInto(
            internal_function, &this->function_, VMap, llvm::CloneFunctionChangeType::LocalChangesOnly, Returns
        );

        // Redirect all uses in this module to the clone
        llvm::SmallVector<llvm::Use*, 16> Uses;
        for (llvm::Use& U : this->function_.uses()) {
            Uses.push_back(&U);
        }
        for (llvm::Use* U : Uses) {
            U->set(internal_function);
        }

        // Rename sdfg to match internal function
        simplified_sdfg->name(internal_function->getName().str());

        // Prevent further inlining
        internal_function->addFnAttr(llvm::Attribute::NoInline);
        internal_function->addFnAttr(llvm::Attribute::OptimizeNone);
        llvm::appendToUsed(*internal_function->getParent(), {internal_function});

        llvm::appendToUsed(*this->function_.getParent(), {&this->function_});
    } else {
        // Prevent further inlining
        this->function_.addFnAttr(llvm::Attribute::NoInline);
        this->function_.addFnAttr(llvm::Attribute::OptimizeNone);
        llvm::appendToUsed(*this->function_.getParent(), {&this->function_});
    }

    bool verify_dbg = false;
    bool failed = llvm::verifyModule(*this->function_.getParent(), &llvm::errs(), &verify_dbg);
    if (failed) {
        throw sdfg::InvalidSDFGException("Module is broken after lifting region.");
    }

    return simplified_sdfg;
}

std::unique_ptr<sdfg::StructuredSDFG> FunctionToSDFG::simplify(std::unique_ptr<sdfg::StructuredSDFG>& sdfg) {
    sdfg::builder::StructuredSDFGBuilder builder_opt(sdfg);
    sdfg::analysis::AnalysisManager analysis_manager(builder_opt.subject());

    // Optimization Pipelines
    sdfg::passes::Pipeline dataflow_simplification = sdfg::passes::Pipeline::dataflow_simplification();
    sdfg::passes::Pipeline symbolic_simplification = sdfg::passes::Pipeline::symbolic_simplification();
    sdfg::passes::Pipeline dce = sdfg::passes::Pipeline::dead_code_elimination();
    sdfg::passes::Pipeline memlet_combine = sdfg::passes::Pipeline::memlet_combine();
    sdfg::passes::DeadDataElimination dde;
    sdfg::passes::SymbolPropagation symbol_propagation_pass;

    // Promote tasklets into symbolic assignments
    sdfg::passes::SymbolPromotion symbol_promotion_pass;
    symbol_promotion_pass.run(builder_opt, analysis_manager);

    // Expand library nodes if requested
    if (DOCC_expand == "tenstorrent") {
        LLVM_DEBUG_PRINTLN("Overriding all library nodes to Tenstorrent");
        auto pass = sdfg::tenstorrent::MathNodeImplementationOverridePass();
        bool success = pass.run(builder_opt, analysis_manager);
    } else if (DOCC_expand != "none") {
        LLVM_DEBUG_PRINTLN("Expanding all library nodes");
        auto expansion_pass = sdfg::passes::ExpansionPass();
        bool expanded = expansion_pass.run(builder_opt, analysis_manager);
    }

    /***** SDFG Minimization *****/

    // Minimize SDFG by fusing blocks, tasklets and sequences
    dataflow_simplification.run(builder_opt, analysis_manager);
    dde.run(builder_opt, analysis_manager);
    dce.run(builder_opt, analysis_manager);

    // Minimize SDFG by fusing symbolic expressions
    symbolic_simplification.run(builder_opt, analysis_manager);
    dde.run(builder_opt, analysis_manager);
    dce.run(builder_opt, analysis_manager);

    /***** Structured Loops *****/

    // Unify continue/break inside branches
    {
        sdfg::passes::CommonAssignmentElimination common_assignment_elimination;
        bool applies = false;
        do {
            applies = false;
            applies |= common_assignment_elimination.run(builder_opt, analysis_manager);
        } while (applies);
        dde.run(builder_opt, analysis_manager);
        dce.run(builder_opt, analysis_manager);
        symbolic_simplification.run(builder_opt, analysis_manager);
    }

    // Propagate variables into constants
    {
        sdfg::passes::ConstantPropagation constant_propagation_pass;
        bool applies = false;
        do {
            applies = false;
            applies |= constant_propagation_pass.run(builder_opt, analysis_manager);
        } while (applies);
    }

    // Convert loops into structured loops
    sdfg::passes::WhileToForConversion for_conversion_pass;
    for_conversion_pass.run(builder_opt, analysis_manager);

    // Propagate for simpler indvar usage
    symbol_propagation_pass.run(builder_opt, analysis_manager);

    // Eliminate redundant branches
    {
        bool applies = false;
        sdfg::passes::ConditionEliminationPass condition_elimination_pass;
        do {
            applies = false;
            applies |= condition_elimination_pass.run(builder_opt, analysis_manager);
        } while (applies);
    }

    // Normalize loop condition and update (run twice)
    sdfg::passes::normalization::LoopNormalFormPass loop_normalization_pass;
    loop_normalization_pass.run(builder_opt, analysis_manager);
    symbol_propagation_pass.run(builder_opt, analysis_manager);
    dde.run(builder_opt, analysis_manager);
    dce.run(builder_opt, analysis_manager);

    // Eliminate symbols correlated to loop iterators
    sdfg::passes::SymbolEvolution symbol_evolution_pass;
    symbol_evolution_pass.run(builder_opt, analysis_manager);

    // Dead code elimination
    symbol_propagation_pass.run(builder_opt, analysis_manager);
    dde.run(builder_opt, analysis_manager);
    dce.run(builder_opt, analysis_manager);

    /***** Data Parallelism *****/

    // Combine address calculations in memlets
    memlet_combine.run(builder_opt, analysis_manager);

    // Move code out of loops where possible
    sdfg::passes::Pipeline code_motion = sdfg::passes::code_motion();
    code_motion.run(builder_opt, analysis_manager);

    // Convert pointer-based iterators to indvar usage
    sdfg::passes::PointerEvolution pointer_evolution_pass;
    pointer_evolution_pass.run(builder_opt, analysis_manager);
    loop_normalization_pass.run(builder_opt, analysis_manager);

    // Convert lib-calls into managed memory
    sdfg::passes::Pipeline memory = sdfg::passes::Pipeline::memory();
    memory.run(builder_opt, analysis_manager);

    sdfg::passes::TypeMinimizationPass type_minimization_pass;
    type_minimization_pass.run(builder_opt, analysis_manager);
    type_minimization_pass.run(builder_opt, analysis_manager);

    // Dead code elimination
    symbol_propagation_pass.run(builder_opt, analysis_manager);
    dce.run(builder_opt, analysis_manager);
    dde.run(builder_opt, analysis_manager);

    // Convert for loops into maps
    sdfg::passes::For2MapPass map_conversion_pass;
    map_conversion_pass.run(builder_opt, analysis_manager);

    // Move code out of maps where possible
    code_motion.run(builder_opt, analysis_manager);

    // Dead code elimination
    dde.run(builder_opt, analysis_manager);
    dce.run(builder_opt, analysis_manager);
    dataflow_simplification.run(builder_opt, analysis_manager);

    return builder_opt.move();
};

} // namespace lifting
} // namespace docc
