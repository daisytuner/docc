#include "docc/lifting/functions/libfunc_lifting.h"

#include <sdfg/data_flow/library_nodes/math/math.h>
#include "sdfg/data_flow/library_nodes/stdlib/stdlib.h"

namespace docc {
namespace lifting {

sdfg::control_flow::State& LibFuncLifting::
    visit(const llvm::BasicBlock* block, const llvm::CallBase* instruction, sdfg::control_flow::State& current_state) {
    llvm::Function* called_func = instruction->getCalledFunction();
    llvm::LibFunc lf;
    TLI_.getLibFunc(called_func->getName(), lf);

    if (is_tasklet(lf)) {
        return this->visit_tasklet(block, instruction, current_state);
    }
    if (is_math(lf)) {
        return this->visit_math(block, instruction, current_state);
    }

    switch (lf) {
        case llvm::LibFunc_calloc:
            return this->visit_calloc(block, instruction, current_state);
        case llvm::LibFunc_free:
            return this->visit_free(block, instruction, current_state);
        case llvm::LibFunc_malloc:
            return this->visit_malloc(block, instruction, current_state);
        default:
            break;
    }

    throw NotImplementedException(
        "Unsupported libfunc", docc::utils::get_debug_info(*instruction), docc::utils::toIRString(*instruction)
    );
}

sdfg::control_flow::State& LibFuncLifting::visit_tasklet(
    const llvm::BasicBlock* block, const llvm::CallBase* instruction, sdfg::control_flow::State& current_state
) {
    llvm::Function* called_func = instruction->getCalledFunction();
    llvm::LibFunc lf;
    TLI_.getLibFunc(called_func->getName(), lf);

    // Define Debug
    auto dbg_info = docc::utils::get_debug_info(*instruction);

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
        "LibFuncLifting: Expected scalar or structure type as output for taskletable intrinsic"
    );

    switch (output_type.type_id()) {
        case sdfg::types::TypeID::Scalar: {
            // Define Output
            auto& output_node = this->builder_.add_access(current_state, output, dbg_info);

            // Define Inputs
            std::vector<std::string> in_conns;
            std::unordered_map<std::string, sdfg::data_flow::AccessNode*> in_nodes;
            std::unordered_map<std::string, sdfg::data_flow::AccessNode*> arg_nodes;
            for (unsigned i = 0; i < instruction->arg_size(); ++i) {
                std::string in_conn = "_in" + std::to_string(i + 1);
                in_conns.push_back(in_conn);

                llvm::Value* arg = instruction->getArgOperand(i);
                if (utils::is_literal(arg)) {
                    std::string arg_str = utils::as_literal(llvm::dyn_cast<const llvm::ConstantData>(arg));
                    auto arg_type = utils::get_type(
                        this->builder_,
                        this->anonymous_types_mapping_,
                        this->DL_,
                        arg->getType(),
                        utils::get_storage_type(this->target_type_, 0)
                    );
                    auto& const_node = this->builder_.add_constant(current_state, arg_str, *arg_type, dbg_info);
                    in_nodes.insert({in_conn, &const_node});
                } else {
                    std::string input = utils::find_const_name_to_sdfg_name(this->constants_mapping_, arg);
                    if (arg_nodes.find(input) == arg_nodes.end()) {
                        auto& input_node = this->builder_.add_access(current_state, input, dbg_info);
                        arg_nodes.insert({input, &input_node});
                    }
                    in_nodes.insert({in_conn, arg_nodes[input]});
                }
            }

            // Define Tasklet
            auto& tasklet = this->builder_.add_tasklet(current_state, as_tasklet(lf), "_out", in_conns, dbg_info);
            this->builder_
                .add_computational_memlet(current_state, tasklet, "_out", output_node, {}, output_type, dbg_info);
            for (auto& [in_conn, in_node] : in_nodes) {
                const sdfg::types::IType* in_type = nullptr;
                if (auto const_node = dynamic_cast<sdfg::data_flow::ConstantNode*>(in_node)) {
                    in_type = &const_node->type();
                } else {
                    in_type = &this->builder_.subject().type(in_node->data());
                }

                this->builder_
                    .add_computational_memlet(current_state, *in_node, tasklet, in_conn, {}, *in_type, dbg_info);
            }


            return current_state;
        }
        case sdfg::types::TypeID::Structure: {
            throw NotImplementedException(
                "LibFuncLifting: Unsupported output type",
                docc::utils::get_debug_info(*instruction),
                docc::utils::toIRString(*instruction)
            );
            // auto& array_type = static_cast<const sdfg::types::Array&>(output_type);

            // sdfg::control_flow::State* state = &current_state;
            // size_t num_element = SymEngine::rcp_static_cast<const
            // SymEngine::Integer>(array_type.num_elements())->as_uint(); for (size_t i = 0; i < num_element; ++i) {
            //     // Define Output
            //     auto& output_node = this->builder_.add_access(*state, output, dbg_info);

            //     // Define Inputs
            //     std::vector<std::string> in_conns;
            //     std::unordered_map<std::string, sdfg::data_flow::AccessNode*> in_nodes;
            //     for (unsigned i = 0; i < instruction->arg_size(); ++i) {
            //         llvm::Value *arg = instruction->getArgOperand(i);
            //         if (utils::is_literal(arg)) {
            //             std::string arg_str = utils::as_literal(llvm::dyn_cast<const llvm::ConstantData>(arg), i);
            //             in_conns.push_back(arg_str);
            //         } else {
            //             std::string input = utils::find_const_name_to_sdfg_name(this->constants_mapping_, arg);
            //             auto& input_node = this->builder_.add_access(*state, input, dbg_info);
            //             in_conns.push_back("_in" + std::to_string(i + 1));
            //             in_nodes.insert({"_in" + std::to_string(i + 1), &input_node});
            //         }
            //     }

            //     // Define Tasklet
            //     auto& tasklet = this->builder_.add_tasklet(*state, operation, "__out", in_conns, dbg_info);
            //     this->builder_.add_computational_memlet(*state, tasklet, "__out", output_node,
            //     {sdfg::symbolic::integer(i)}, dbg_info); for (auto& [in_conn, in_node] : in_nodes) {
            //         this->builder_.add_computational_memlet(*state, *in_node, tasklet, in_conn,
            //         {sdfg::symbolic::integer(i)}, dbg_info);
            //     }

            //     state = &this->builder_.add_state_after(*state, true, dbg_info);
            // }

            // return *state;
        }
        default: {
            throw NotImplementedException(
                "BinaryOperator: Unsupported output type",
                docc::utils::get_debug_info(*instruction),
                docc::utils::toIRString(*instruction)
            );
        }
    }
}

sdfg::control_flow::State& LibFuncLifting::visit_math(
    const llvm::BasicBlock* block, const llvm::CallBase* instruction, sdfg::control_flow::State& current_state
) {
    llvm::Function* called_func = instruction->getCalledFunction();
    llvm::LibFunc lf;
    TLI_.getLibFunc(called_func->getName(), lf);

    // Define Debug
    auto dbg_info = docc::utils::get_debug_info(*instruction);

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
        "LibFuncLifting: Expected scalar or structure type as output for taskletable intrinsic"
    );

    switch (output_type.type_id()) {
        case sdfg::types::TypeID::Scalar: {
            // Define Output
            auto& output_node = this->builder_.add_access(current_state, output, dbg_info);

            // Define Inputs
            std::vector<std::string> in_conns;
            std::unordered_map<std::string, sdfg::data_flow::AccessNode*> in_nodes;
            std::unordered_map<std::string, sdfg::data_flow::AccessNode*> arg_nodes;
            for (unsigned i = 0; i < instruction->arg_size(); ++i) {
                std::string in_conn = "_in" + std::to_string(i + 1);
                in_conns.push_back(in_conn);

                llvm::Value* arg = instruction->getArgOperand(i);
                if (utils::is_literal(arg)) {
                    std::string arg_str = utils::as_literal(llvm::dyn_cast<const llvm::ConstantData>(arg));
                    auto arg_type = utils::get_type(
                        this->builder_,
                        this->anonymous_types_mapping_,
                        this->DL_,
                        arg->getType(),
                        utils::get_storage_type(this->target_type_, 0)
                    );
                    auto& const_node = this->builder_.add_constant(current_state, arg_str, *arg_type, dbg_info);
                    in_nodes.insert({in_conn, &const_node});
                } else {
                    std::string input = utils::find_const_name_to_sdfg_name(this->constants_mapping_, arg);
                    if (arg_nodes.find(input) == arg_nodes.end()) {
                        auto& input_node = this->builder_.add_access(current_state, input, dbg_info);
                        arg_nodes.insert({input, &input_node});
                    }
                    in_nodes.insert({in_conn, arg_nodes[input]});
                }
            }

            // Define Tasklet
            auto& tasklet = this->builder_.add_library_node<
                sdfg::math::cmath::CMathNode>(current_state, dbg_info, as_math_function(lf), output_type.primitive_type());
            this->builder_
                .add_computational_memlet(current_state, tasklet, "_out", output_node, {}, output_type, dbg_info);
            for (auto& [in_conn, in_node] : in_nodes) {
                const sdfg::types::IType* in_type = nullptr;
                if (auto const_node = dynamic_cast<sdfg::data_flow::ConstantNode*>(in_node)) {
                    in_type = &const_node->type();
                } else {
                    in_type = &this->builder_.subject().type(in_node->data());
                }

                this->builder_
                    .add_computational_memlet(current_state, *in_node, tasklet, in_conn, {}, *in_type, dbg_info);
            }


            return current_state;
        }
        case sdfg::types::TypeID::Structure: {
            throw NotImplementedException(
                "LibFuncLifting: Unsupported output type",
                docc::utils::get_debug_info(*instruction),
                docc::utils::toIRString(*instruction)
            );
            // auto& array_type = static_cast<const sdfg::types::Array&>(output_type);

            // sdfg::control_flow::State* state = &current_state;
            // size_t num_element = SymEngine::rcp_static_cast<const
            // SymEngine::Integer>(array_type.num_elements())->as_uint(); for (size_t i = 0; i < num_element; ++i) {
            //     // Define Output
            //     auto& output_node = this->builder_.add_access(*state, output, dbg_info);

            //     // Define Inputs
            //     std::vector<std::string> in_conns;
            //     std::unordered_map<std::string, sdfg::data_flow::AccessNode*> in_nodes;
            //     for (unsigned i = 0; i < instruction->arg_size(); ++i) {
            //         llvm::Value *arg = instruction->getArgOperand(i);
            //         if (utils::is_literal(arg)) {
            //             std::string arg_str = utils::as_literal(llvm::dyn_cast<const llvm::ConstantData>(arg), i);
            //             in_conns.push_back(arg_str);
            //         } else {
            //             std::string input = utils::find_const_name_to_sdfg_name(this->constants_mapping_, arg);
            //             auto& input_node = this->builder_.add_access(*state, input, dbg_info);
            //             in_conns.push_back("_in" + std::to_string(i + 1));
            //             in_nodes.insert({"_in" + std::to_string(i + 1), &input_node});
            //         }
            //     }

            //     // Define Tasklet
            //     auto& tasklet = this->builder_.add_tasklet(*state, operation, "__out", in_conns, dbg_info);
            //     this->builder_.add_computational_memlet(*state, tasklet, "__out", output_node,
            //     {sdfg::symbolic::integer(i)}, dbg_info); for (auto& [in_conn, in_node] : in_nodes) {
            //         this->builder_.add_computational_memlet(*state, *in_node, tasklet, in_conn,
            //         {sdfg::symbolic::integer(i)}, dbg_info);
            //     }

            //     state = &this->builder_.add_state_after(*state, true, dbg_info);
            // }

            // return *state;
        }
        default: {
            throw NotImplementedException(
                "BinaryOperator: Unsupported output type",
                docc::utils::get_debug_info(*instruction),
                docc::utils::toIRString(*instruction)
            );
        }
    }
}

sdfg::control_flow::State& LibFuncLifting::visit_calloc(
    const llvm::BasicBlock* block, const llvm::CallBase* instruction, sdfg::control_flow::State& current_state
) {
    llvm::Function* called_func = instruction->getCalledFunction();
    llvm::LibFunc lf;
    TLI_.getLibFunc(called_func->getName(), lf);

    // Define Debug
    auto dbg_info = docc::utils::get_debug_info(*instruction);

    // Define output
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
        (output_type.type_id() == sdfg::types::TypeID::Pointer) &&
        "LibFuncLifting: Expected pointer type as output for calloc intrinsic"
    );

    // define args
    llvm::Value* arg_num = instruction->getArgOperand(0);
    sdfg::symbolic::Expression num_sym;
    if (utils::is_literal(arg_num)) {
        num_sym = utils::as_symbol(llvm::dyn_cast<const llvm::ConstantData>(arg_num));
    } else {
        std::string arg_name = utils::find_const_name_to_sdfg_name(this->constants_mapping_, arg_num);
        num_sym = sdfg::symbolic::symbol(arg_name);
    }

    llvm::Value* arg_size = instruction->getArgOperand(1);
    sdfg::symbolic::Expression size_sym;
    if (utils::is_literal(arg_size)) {
        size_sym = utils::as_symbol(llvm::dyn_cast<const llvm::ConstantData>(arg_size));
    } else {
        std::string arg_name = utils::find_const_name_to_sdfg_name(this->constants_mapping_, arg_size);
        size_sym = sdfg::symbolic::symbol(arg_name);
    }

    // Define library node
    auto& output_node = this->builder_.add_access(current_state, output, dbg_info);
    auto& lib_node =
        this->builder_.add_library_node<sdfg::stdlib::CallocNode>(current_state, dbg_info, num_sym, size_sym);

    sdfg::types::Pointer opaque_ptr;
    this->builder_.add_computational_memlet(current_state, lib_node, "_ret", output_node, {}, opaque_ptr, dbg_info);

    return current_state;
}

sdfg::control_flow::State& LibFuncLifting::visit_free(
    const llvm::BasicBlock* block, const llvm::CallBase* instruction, sdfg::control_flow::State& current_state
) {
    llvm::Function* called_func = instruction->getCalledFunction();
    llvm::LibFunc lf;
    TLI_.getLibFunc(called_func->getName(), lf);

    // Define Debug
    auto dbg_info = docc::utils::get_debug_info(*instruction);

    // Define input
    llvm::Value* pointer_operand = instruction->getArgOperand(0);
    std::string input = utils::find_const_name_to_sdfg_name(this->constants_mapping_, pointer_operand);
    auto& input_node = this->builder_.add_access(current_state, input, dbg_info);
    auto& output_node = this->builder_.add_access(current_state, input, dbg_info);

    // Define library node
    auto& lib_node = this->builder_.add_library_node<sdfg::stdlib::FreeNode>(current_state, dbg_info);

    sdfg::types::Pointer opaque_ptr;
    this->builder_.add_computational_memlet(current_state, input_node, lib_node, "_ptr", {}, opaque_ptr, dbg_info);
    this->builder_.add_computational_memlet(current_state, lib_node, "_ptr", output_node, {}, opaque_ptr, dbg_info);

    return current_state;
}

sdfg::control_flow::State& LibFuncLifting::visit_malloc(
    const llvm::BasicBlock* block, const llvm::CallBase* instruction, sdfg::control_flow::State& current_state
) {
    llvm::Function* called_func = instruction->getCalledFunction();
    llvm::LibFunc lf;
    TLI_.getLibFunc(called_func->getName(), lf);

    // Define Debug
    auto dbg_info = docc::utils::get_debug_info(*instruction);

    // Define output
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
        (output_type.type_id() == sdfg::types::TypeID::Pointer) &&
        "LibFuncLifting: Expected pointer type as output for malloc intrinsic"
    );

    // define args
    llvm::Value* arg = instruction->getArgOperand(0);
    sdfg::symbolic::Expression sym;
    if (utils::is_literal(arg)) {
        sym = utils::as_symbol(llvm::dyn_cast<const llvm::ConstantData>(arg));
    } else {
        std::string arg_name = utils::find_const_name_to_sdfg_name(this->constants_mapping_, arg);
        sym = sdfg::symbolic::symbol(arg_name);
    }

    // Define library node
    auto& output_node = this->builder_.add_access(current_state, output, dbg_info);
    auto& lib_node = this->builder_.add_library_node<sdfg::stdlib::MallocNode>(current_state, dbg_info, sym);

    sdfg::types::Pointer opaque_ptr;
    this->builder_.add_computational_memlet(current_state, lib_node, "_ret", output_node, {}, opaque_ptr, dbg_info);

    return current_state;
}

} // namespace lifting
} // namespace docc
