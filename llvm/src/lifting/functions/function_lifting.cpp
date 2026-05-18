#include "docc/lifting/functions/function_lifting.h"

#include "docc/lifting/functions/blas_lifting.h"
#include "docc/lifting/functions/intrinsic_lifting.h"
#include "docc/lifting/functions/libfunc_lifting.h"

#include <sdfg/data_flow/library_nodes/call_node.h>
#include <sdfg/data_flow/library_nodes/invoke_node.h>

namespace docc {
namespace lifting {

static const std::unordered_set<std::string> BLACKLISTED_FUNCTIONS = {};

bool FunctionLifting::is_supported(llvm::TargetLibraryInfo& TLI, const llvm::CallBase* instruction) {
    if (!llvm::isa<llvm::CallInst>(instruction) && !llvm::isa<llvm::InvokeInst>(instruction)) {
        LiftingReport::add_failed_lift(
            docc::utils::get_debug_info(*instruction),
            "Unsupported call base type",
            docc::utils::toIRString(*instruction)
        );
        LLVM_DEBUG_PRINTLN("Unsupported call base type: " << docc::utils::toIRString(*instruction));
        return false;
    }

    // Intrinsic functions
    if (auto intrinsic_inst = llvm::dyn_cast<const llvm::IntrinsicInst>(instruction)) {
        auto iid = intrinsic_inst->getIntrinsicID();
        if (IntrinsicLifting::is_supported(intrinsic_inst)) {
            return true;
        } else {
            LiftingReport::add_failed_lift(
                docc::utils::get_debug_info(*instruction),
                "Unsupported intrinsic: " + llvm::Intrinsic::getName(iid).str(),
                docc::utils::toIRString(*instruction)
            );
            LLVM_DEBUG_PRINTLN("Unsupported intrinsic: " << docc::utils::toIRString(*instruction));
            return false;
        }
    }

    // Operand must be supported
    auto operand = instruction->getCalledOperand();
    if (!operand) {
        LiftingReport::add_failed_lift(
            docc::utils::get_debug_info(*instruction),
            "Unsupported call with no called operand",
            docc::utils::toIRString(*instruction)
        );
        LLVM_DEBUG_PRINTLN("Unsupported call with no called operand: " << docc::utils::toIRString(*instruction));
        return false;
    }

    return true;
}

sdfg::control_flow::State& FunctionLifting::
    visit(const llvm::BasicBlock* block, const llvm::CallBase* instruction, sdfg::control_flow::State& current_state) {
    // Intrinsic functions
    if (llvm::isa<const llvm::IntrinsicInst>(instruction)) {
        IntrinsicLifting lifter(
            TLI_,
            DL_,
            function_,
            target_type_,
            builder_,
            state_mapping_,
            pred_mapping_,
            constants_mapping_,
            anonymous_types_mapping_
        );
        return lifter.visit(block, instruction, current_state);
    }

    // Match known library functions
    llvm::Function* called_func = instruction->getCalledFunction();
    if (called_func) {
        // LLVM Library functions
        llvm::LibFunc lf;
        if (TLI_.getLibFunc(called_func->getName(), lf) && LibFuncLifting::is_supported(lf)) {
            LibFuncLifting lifter(
                TLI_,
                DL_,
                function_,
                target_type_,
                builder_,
                state_mapping_,
                pred_mapping_,
                constants_mapping_,
                anonymous_types_mapping_
            );
            return lifter.visit(block, instruction, current_state);
        }

        // BLAS functions
        if (BLASLifting::is_supported(*called_func)) {
            BLASLifting lifter(
                TLI_,
                DL_,
                function_,
                target_type_,
                builder_,
                state_mapping_,
                pred_mapping_,
                constants_mapping_,
                anonymous_types_mapping_
            );
            return lifter.visit(block, instruction, current_state);
        }
    }

    if (auto call_inst = llvm::dyn_cast<const llvm::CallInst>(instruction)) {
        return this->visit_call(block, call_inst, current_state);
    } else if (auto invoke_inst = llvm::dyn_cast<const llvm::InvokeInst>(instruction)) {
        return this->visit_invoke(block, invoke_inst, current_state);
    } else {
        throw NotImplementedException(
            "Unsupported CallBase type in FunctionLifting::visit",
            docc::utils::get_debug_info(*instruction),
            docc::utils::toIRString(*instruction)
        );
    }
}

sdfg::control_flow::State& FunctionLifting::visit_call(
    const llvm::BasicBlock* block, const llvm::CallInst* instruction, sdfg::control_flow::State& current_state
) {
    // Define Debug
    auto dbg_info = docc::utils::get_debug_info(*instruction);

    // Define function name
    std::string callee_name;
    llvm::Function* called_func = instruction->getCalledFunction();
    if (called_func) {
        callee_name = called_func->getName().str();
    } else if (instruction->getCalledOperand()) {
        callee_name = utils::get_name(instruction->getCalledOperand());
    } else {
        throw NotImplementedException(
            "Unsupported call instruction with no function or operand",
            docc::utils::get_debug_info(*instruction),
            docc::utils::toIRString(*instruction)
        );
    }
    assert(!callee_name.empty() && "Function name is empty");

    // Define return type
    std::vector<std::string> outputs;
    std::unordered_map<std::string, sdfg::data_flow::AccessNode*> out_conns;
    std::unordered_map<std::string, sdfg::data_flow::AccessNode*> unique_nodes_out;
    std::unordered_map<std::string, std::unique_ptr<sdfg::types::IType>> types;
    if (!instruction->getType()->isVoidTy()) {
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
        auto output_node = &this->builder_.add_access(current_state, output, dbg_info);

        outputs.push_back("_ret");
        out_conns.insert({"_ret", output_node});
        unique_nodes_out.insert({output, output_node});
        types.insert({"_ret", this->builder_.subject().type(output).clone()});
    }

    // Define arguments
    std::vector<std::string> inputs;
    std::unordered_map<std::string, sdfg::data_flow::AccessNode*> in_conns;
    std::unordered_map<std::string, sdfg::data_flow::AccessNode*> unique_nodes_in;
    size_t i = 0;
    for (auto& op : instruction->args()) {
        std::string conn = "_arg" + std::to_string(i);

        auto op_type = utils::get_type(
            this->builder_,
            this->anonymous_types_mapping_,
            this->DL_,
            op->getType(),
            utils::get_storage_type(this->target_type_, 0)
        );
        if (utils::is_literal(op)) {
            std::string value = utils::as_initializer(llvm::dyn_cast<const llvm::ConstantData>(op));
            auto& const_node = this->builder_.add_constant(current_state, value, *op_type, dbg_info);

            inputs.push_back(conn);
            in_conns.insert({conn, &const_node});
            types.insert({conn, std::move(op_type)});
        } else {
            std::string data = utils::find_const_name_to_sdfg_name(this->constants_mapping_, op);

            // Add type
            types.insert({conn, op_type->clone()});

            // Add to inputs
            if (unique_nodes_in.find(data) == unique_nodes_in.end()) {
                auto& data_node_in = this->builder_.add_access(current_state, data, dbg_info);
                unique_nodes_in.insert({data, &data_node_in});
            }
            inputs.push_back(conn);
            in_conns.insert({conn, unique_nodes_in[data]});

            if (op_type->type_id() == sdfg::types::TypeID::Pointer) {
                if (called_func && i < called_func->arg_size()) {
                    if (called_func->getArg(i)->onlyReadsMemory()) {
                        // No output node needed
                        i++;
                        continue;
                    } else if (called_func->getArg(i)->hasByValAttr()) {
                        // No output node needed
                        i++;
                        continue;
                    }
                }
                if (unique_nodes_out.find(data) == unique_nodes_out.end()) {
                    auto& data_node_out = this->builder_.add_access(current_state, data, dbg_info);
                    unique_nodes_out.insert({data, &data_node_out});
                }
                outputs.push_back(conn);
                out_conns.insert({conn, unique_nodes_out[data]});
            }
        }

        i++;
    }

    // Define library node
    auto& lib_node =
        this->builder_
            .add_library_node<sdfg::data_flow::CallNode>(current_state, dbg_info, callee_name, outputs, inputs);

    // Define in connectors
    for (auto& [in_conn, in_node] : in_conns) {
        auto& conn_type = types.at(in_conn);
        this->builder_.add_computational_memlet(current_state, *in_node, lib_node, in_conn, {}, *conn_type, dbg_info);
    }

    // Define out connectors
    for (auto& [out_conn, out_node] : out_conns) {
        auto& conn_type = types.at(out_conn);
        this->builder_.add_computational_memlet(current_state, lib_node, out_conn, *out_node, {}, *conn_type, dbg_info);
    }

    return current_state;
}

sdfg::control_flow::State& FunctionLifting::visit_invoke(
    const llvm::BasicBlock* block, const llvm::InvokeInst* instruction, sdfg::control_flow::State& current_state
) {
    // Define Debug
    auto dbg_info = docc::utils::get_debug_info(*instruction);

    // Define function name
    std::string callee_name;
    llvm::Function* called_func = instruction->getCalledFunction();
    if (called_func) {
        callee_name = called_func->getName().str();
    } else if (instruction->getCalledOperand()) {
        callee_name = utils::get_name(instruction->getCalledOperand());
    } else {
        throw NotImplementedException(
            "Unsupported call instruction with no function or operand",
            docc::utils::get_debug_info(*instruction),
            docc::utils::toIRString(*instruction)
        );
    }
    assert(!callee_name.empty() && "Function name is empty");

    // Define return type
    std::vector<std::string> outputs;
    std::unordered_map<std::string, sdfg::data_flow::AccessNode*> out_conns;
    std::unordered_map<std::string, sdfg::data_flow::AccessNode*> unique_nodes_out;
    std::unordered_map<std::string, std::unique_ptr<sdfg::types::IType>> types;
    if (!instruction->getType()->isVoidTy()) {
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
        auto output_node = &this->builder_.add_access(current_state, output, dbg_info);

        outputs.push_back("_ret");
        out_conns.insert({"_ret", output_node});
        unique_nodes_out.insert({output, output_node});
        types.insert({"_ret", this->builder_.subject().type(output).clone()});
    }

    // Define arguments
    std::vector<std::string> inputs;
    std::unordered_map<std::string, sdfg::data_flow::AccessNode*> in_conns;
    std::unordered_map<std::string, sdfg::data_flow::AccessNode*> unique_nodes_in;
    size_t i = 0;
    for (auto& op : instruction->args()) {
        std::string conn = "_arg" + std::to_string(i);

        auto op_type = utils::get_type(
            this->builder_,
            this->anonymous_types_mapping_,
            this->DL_,
            op->getType(),
            utils::get_storage_type(this->target_type_, 0)
        );
        if (utils::is_literal(op)) {
            std::string value = utils::as_initializer(llvm::dyn_cast<const llvm::ConstantData>(op));
            auto& const_node = this->builder_.add_constant(current_state, value, *op_type, dbg_info);

            inputs.push_back(conn);
            in_conns.insert({conn, &const_node});
            types.insert({conn, std::move(op_type)});
        } else {
            std::string data = utils::find_const_name_to_sdfg_name(this->constants_mapping_, op);

            // Add type
            types.insert({conn, std::move(op_type)});

            if (unique_nodes_in.find(data) == unique_nodes_in.end()) {
                auto& data_node_in = this->builder_.add_access(current_state, data, dbg_info);
                unique_nodes_in.insert({data, &data_node_in});

                if (called_func && !called_func->isVarArg() && called_func->getArg(i)->onlyReadsMemory()) {
                    // No output node needed
                } else {
                    auto& data_node_out = this->builder_.add_access(current_state, data, dbg_info);
                    unique_nodes_out.insert({data, &data_node_out});
                }
            }

            inputs.push_back(conn);
            in_conns.insert({conn, unique_nodes_in[data]});

            if (called_func && !called_func->isVarArg() && called_func->getArg(i)->onlyReadsMemory()) {
                // No output connector needed
            } else {
                outputs.push_back(conn);
                out_conns.insert({conn, unique_nodes_out[data]});
            }
        }

        i++;
    }

    // Define library node
    auto& lib_node =
        this->builder_
            .add_library_node<sdfg::data_flow::InvokeNode>(current_state, dbg_info, callee_name, outputs, inputs);

    // Define in connectors
    for (auto& [in_conn, in_node] : in_conns) {
        auto& conn_type = types.at(in_conn);
        this->builder_.add_computational_memlet(current_state, *in_node, lib_node, in_conn, {}, *conn_type, dbg_info);
    }

    // Define out connectors
    for (auto& [out_conn, out_node] : out_conns) {
        auto& conn_type = types.at(out_conn);
        this->builder_.add_computational_memlet(current_state, lib_node, out_conn, *out_node, {}, *conn_type, dbg_info);
    }

    // Add unwind handling
    std::string unwind_container = "__unwind_" + utils::get_name(instruction);
    sdfg::types::Scalar unwind_type(sdfg::types::PrimitiveType::Bool);
    auto& unwind_node = this->builder_.add_access(current_state, unwind_container, dbg_info);
    this->builder_.add_computational_memlet(current_state, lib_node, "_unwind", unwind_node, {}, unwind_type, dbg_info);

    return current_state;
}

} // namespace lifting
} // namespace docc
