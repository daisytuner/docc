#include "docc/lifting/functions/intrinsic_lifting.h"

#include <sdfg/data_flow/library_nodes/math/math.h>
#include <sdfg/data_flow/library_nodes/stdlib/stdlib.h>

namespace docc {
namespace lifting {

sdfg::control_flow::State& IntrinsicLifting::
    visit(const llvm::BasicBlock* block, const llvm::CallBase* instruction, sdfg::control_flow::State& current_state) {
    auto intrinsic_inst = llvm::dyn_cast<const llvm::IntrinsicInst>(instruction);
    auto iid = intrinsic_inst->getIntrinsicID();

    if (IntrinsicLifting::is_noop(iid)) {
        return current_state;
    }
    if (IntrinsicLifting::is_tasklet(iid)) {
        return this->visit_tasklet(block, intrinsic_inst, current_state);
    }
    if (IntrinsicLifting::is_math(iid)) {
        return this->visit_math(block, intrinsic_inst, current_state);
    }

    switch (iid) {
        case llvm::Intrinsic::memcpy:
            return this->visit_memcpy(block, intrinsic_inst, current_state);
        case llvm::Intrinsic::memmove:
            return this->visit_memmove(block, intrinsic_inst, current_state);
        case llvm::Intrinsic::memset:
            return this->visit_memset(block, intrinsic_inst, current_state);
        case llvm::Intrinsic::trap:
            return this->visit_trap(block, intrinsic_inst, current_state);
        default:
            break;
    }

    throw NotImplementedException(
        "Unsupported intrinsic", docc::utils::get_debug_info(*instruction), docc::utils::toIRString(*instruction)
    );
}

sdfg::control_flow::State& IntrinsicLifting::visit_tasklet(
    const llvm::BasicBlock* block, const llvm::IntrinsicInst* instruction, sdfg::control_flow::State& current_state
) {
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
        "IntrinsicLifting: Expected scalar or structure type as output for taskletable intrinsic"
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
                if (i >= 1 && (instruction->getIntrinsicID() == llvm::Intrinsic::abs ||
                               instruction->getIntrinsicID() == llvm::Intrinsic::expect ||
                               instruction->getIntrinsicID() == llvm::Intrinsic::expect_with_probability)) {
                    // Skip additional arguments for abs, expect, and expect_with_probability intrinsics
                    continue;
                }

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
            auto code = as_tasklet(instruction->getIntrinsicID());
            auto& tasklet = this->builder_.add_tasklet(current_state, code, {"_out"}, in_conns, dbg_info);
            this->builder_.add_computational_memlet(current_state, tasklet, "_out", output_node, {}, dbg_info);
            for (auto& [in_conn, in_node] : in_nodes) {
                this->builder_.add_computational_memlet(current_state, *in_node, tasklet, in_conn, {}, dbg_info);
            }
            return current_state;
        }
        case sdfg::types::TypeID::Structure: {
            auto& struct_type = static_cast<const sdfg::types::Structure&>(output_type);
            auto& struct_def = this->builder_.subject().structure(struct_type.name());
            assert(struct_def.is_vector() && "Expected vector structure type as output");
            auto& vec_elem_type = struct_def.vector_element_type();

            sdfg::control_flow::State* state = &current_state;
            size_t num_element = struct_def.vector_size();
            for (size_t i = 0; i < num_element; ++i) {
                // Define Output
                auto& output_node = this->builder_.add_access(*state, output, dbg_info);

                // Define Inputs
                std::vector<std::string> in_conns;
                std::unordered_map<std::string, sdfg::data_flow::AccessNode*> in_nodes;
                std::unordered_map<std::string, sdfg::data_flow::AccessNode*> arg_nodes;
                for (unsigned j = 0; j < instruction->arg_size(); ++j) {
                    if (j >= 1 && (instruction->getIntrinsicID() == llvm::Intrinsic::abs ||
                                   instruction->getIntrinsicID() == llvm::Intrinsic::expect ||
                                   instruction->getIntrinsicID() == llvm::Intrinsic::expect_with_probability)) {
                        // Skip strict argument for abs and expect intrinsics
                        continue;
                    }

                    std::string in_conn = "_in" + std::to_string(j + 1);
                    in_conns.push_back(in_conn);

                    llvm::Value* arg = instruction->getArgOperand(j);
                    if (utils::is_literal(arg)) {
                        std::string arg_str = utils::as_initializer(llvm::dyn_cast<const llvm::ConstantData>(arg));
                        auto arg_type = utils::get_type(
                            this->builder_,
                            this->anonymous_types_mapping_,
                            this->DL_,
                            arg->getType(),
                            utils::get_storage_type(this->target_type_, 0)
                        );
                        auto& const_node = this->builder_.add_constant(*state, arg_str, *arg_type, dbg_info);
                        in_nodes.insert({in_conn, &const_node});
                    } else {
                        std::string input = utils::find_const_name_to_sdfg_name(this->constants_mapping_, arg);
                        if (arg_nodes.find(input) == arg_nodes.end()) {
                            auto& input_node = this->builder_.add_access(*state, input, dbg_info);
                            arg_nodes.insert({input, &input_node});
                        }
                        in_nodes.insert({in_conn, arg_nodes[input]});
                    }
                }

                // Define Tasklet
                auto code = as_tasklet(instruction->getIntrinsicID());
                auto& tasklet = this->builder_.add_tasklet(*state, code, {"_out"}, in_conns, dbg_info);
                this->builder_
                    .add_computational_memlet(*state, tasklet, "_out", output_node, {sdfg::symbolic::integer(i)}, dbg_info);
                for (auto& [in_conn, in_node] : in_nodes) {
                    this->builder_.add_computational_memlet(
                        *state, *in_node, tasklet, in_conn, {sdfg::symbolic::integer(i)}, dbg_info
                    );
                }

                state = &this->builder_.add_state_after(*state, true, dbg_info);
            }

            return *state;
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


sdfg::control_flow::State& IntrinsicLifting::visit_math(
    const llvm::BasicBlock* block, const llvm::IntrinsicInst* instruction, sdfg::control_flow::State& current_state
) {
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
        "IntrinsicLifting: Expected scalar or structure type as output for taskletable intrinsic"
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
            auto intrinsic_id = instruction->getIntrinsicID();
            if (instruction->getIntrinsicID() == llvm::Intrinsic::powi) {
                intrinsic_id = llvm::Intrinsic::pow;
            }
            auto& lib_node = this->builder_.add_library_node<sdfg::math::cmath::CMathNode>(
                current_state, dbg_info, as_math_function(intrinsic_id), output_type.primitive_type()
            );
            this->builder_
                .add_computational_memlet(current_state, lib_node, "_out", output_node, {}, output_type, dbg_info);
            for (auto& [in_conn, in_node] : in_nodes) {
                const sdfg::types::IType* in_type = nullptr;
                if (auto const_node = dynamic_cast<sdfg::data_flow::ConstantNode*>(in_node)) {
                    in_type = &const_node->type();
                } else {
                    in_type = &this->builder_.subject().type(in_node->data());
                }

                if (instruction->getIntrinsicID() == llvm::Intrinsic::powi) {
                    // we're mapping powi to pow, so the exponent argument needs to be converted to the base type
                    this->builder_
                        .add_computational_memlet(current_state, *in_node, lib_node, in_conn, {}, output_type, dbg_info);
                } else {
                    this->builder_
                        .add_computational_memlet(current_state, *in_node, lib_node, in_conn, {}, *in_type, dbg_info);
                }
            }
            return current_state;
        }
        case sdfg::types::TypeID::Structure: {
            auto& struct_type = static_cast<const sdfg::types::Structure&>(output_type);
            auto& struct_def = this->builder_.subject().structure(struct_type.name());
            assert(struct_def.is_vector() && "Expected vector structure type as output");
            auto& vec_elem_type = struct_def.vector_element_type();

            sdfg::control_flow::State* state = &current_state;
            size_t num_element = struct_def.vector_size();
            for (size_t i = 0; i < num_element; ++i) {
                // Define Output
                auto& output_node = this->builder_.add_access(*state, output, dbg_info);

                // Define Inputs
                std::vector<std::string> in_conns;
                std::unordered_map<std::string, sdfg::data_flow::AccessNode*> in_nodes;
                std::unordered_map<std::string, sdfg::data_flow::AccessNode*> arg_nodes;
                for (unsigned i = 0; i < instruction->arg_size(); ++i) {
                    std::string in_conn = "_in" + std::to_string(i + 1);
                    in_conns.push_back(in_conn);

                    llvm::Value* arg = instruction->getArgOperand(i);
                    if (utils::is_literal(arg)) {
                        std::string arg_str = utils::as_initializer(llvm::dyn_cast<const llvm::ConstantData>(arg));
                        auto arg_type = utils::get_type(
                            this->builder_,
                            this->anonymous_types_mapping_,
                            this->DL_,
                            arg->getType(),
                            utils::get_storage_type(this->target_type_, 0)
                        );
                        auto& const_node = this->builder_.add_constant(*state, arg_str, *arg_type, dbg_info);
                        in_nodes.insert({in_conn, &const_node});
                    } else {
                        std::string input = utils::find_const_name_to_sdfg_name(this->constants_mapping_, arg);
                        if (arg_nodes.find(input) == arg_nodes.end()) {
                            auto& input_node = this->builder_.add_access(*state, input, dbg_info);
                            arg_nodes.insert({input, &input_node});
                        }
                        in_nodes.insert({in_conn, arg_nodes[input]});
                    }
                }

                // Define Tasklet
                auto intrinsic_id = instruction->getIntrinsicID();
                if (instruction->getIntrinsicID() == llvm::Intrinsic::powi) {
                    intrinsic_id = llvm::Intrinsic::pow;
                }
                auto& lib_node = this->builder_.add_library_node<sdfg::math::cmath::CMathNode>(
                    *state, dbg_info, as_math_function(intrinsic_id), vec_elem_type.primitive_type()
                );
                this->builder_.add_computational_memlet(
                    *state, lib_node, "_out", output_node, {sdfg::symbolic::integer(i)}, output_type, dbg_info
                );
                for (auto& [in_conn, in_node] : in_nodes) {
                    const sdfg::types::IType* in_type = nullptr;
                    if (auto const_node = dynamic_cast<sdfg::data_flow::ConstantNode*>(in_node)) {
                        in_type = &const_node->type();
                    } else {
                        in_type = &this->builder_.subject().type(in_node->data());
                    }

                    if (instruction->getIntrinsicID() == llvm::Intrinsic::powi) {
                        // we're mapping powi to pow, so the exponent argument needs to be converted to the base type
                        this->builder_.add_computational_memlet(
                            *state, *in_node, lib_node, in_conn, {sdfg::symbolic::integer(i)}, output_type, dbg_info
                        );
                    } else {
                        this->builder_.add_computational_memlet(
                            *state, *in_node, lib_node, in_conn, {sdfg::symbolic::integer(i)}, *in_type, dbg_info
                        );
                    }
                }

                state = &this->builder_.add_state_after(*state, true, dbg_info);
            }

            return *state;
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

sdfg::control_flow::State& IntrinsicLifting::visit_memcpy(
    const llvm::BasicBlock* block, const llvm::IntrinsicInst* instruction, sdfg::control_flow::State& current_state
) {
    // Define Debug
    auto dbg_info = docc::utils::get_debug_info(*instruction);

    // Add function symbol
    if (!this->builder_.subject().exists("memcpy")) {
        sdfg::types::Function function_type(sdfg::types::Pointer(), false);
        function_type.add_param(sdfg::types::Pointer());
        function_type.add_param(sdfg::types::Pointer());
        function_type.add_param(sdfg::types::Scalar(sdfg::types::PrimitiveType::UInt64));
        this->builder_.add_external("memcpy", function_type, sdfg::LinkageType_External);
    }

    // Define dst
    llvm::Value* arg_dst = instruction->getArgOperand(0);
    assert(!utils::is_literal(arg_dst) && "IntrinsicLifting: Expect non-literal argument for memcpy");
    std::string dst_name = utils::find_const_name_to_sdfg_name(this->constants_mapping_, arg_dst);
    auto& dst_node = this->builder_.add_access(current_state, dst_name, dbg_info);
    auto& dst_type = this->builder_.subject().type(dst_name);

    // Define src
    llvm::Value* arg_src = instruction->getArgOperand(1);
    assert(!utils::is_literal(arg_src) && "IntrinsicLifting: Expect non-literal argument for memcpy");
    std::string src_name = utils::find_const_name_to_sdfg_name(this->constants_mapping_, arg_src);
    auto& src_node = this->builder_.add_access(current_state, src_name, dbg_info);
    auto& src_type = this->builder_.subject().type(src_name);

    // Define count
    llvm::Value* arg_count = instruction->getArgOperand(2);
    sdfg::symbolic::Expression count_sym;
    if (utils::is_literal(arg_count)) {
        count_sym = utils::as_symbol(llvm::dyn_cast<const llvm::ConstantData>(arg_count));
    } else {
        std::string arg_name = utils::find_const_name_to_sdfg_name(this->constants_mapping_, arg_count);
        count_sym = sdfg::symbolic::symbol(arg_name);
    }

    // Define library node
    auto& lib_node = this->builder_.add_library_node<sdfg::stdlib::MemcpyNode>(current_state, dbg_info, count_sym);
    this->builder_.add_computational_memlet(current_state, src_node, lib_node, "_src", {}, src_type, dbg_info);
    this->builder_.add_computational_memlet(current_state, lib_node, "_dst", dst_node, {}, dst_type, dbg_info);

    return current_state;
}

sdfg::control_flow::State& IntrinsicLifting::visit_memmove(
    const llvm::BasicBlock* block, const llvm::IntrinsicInst* instruction, sdfg::control_flow::State& current_state
) {
    // Define Debug
    auto dbg_info = docc::utils::get_debug_info(*instruction);

    // Add function symbol
    if (!this->builder_.subject().exists("memmove")) {
        sdfg::types::Function function_type(sdfg::types::Pointer(), false);
        function_type.add_param(sdfg::types::Pointer());
        function_type.add_param(sdfg::types::Pointer());
        function_type.add_param(sdfg::types::Scalar(sdfg::types::PrimitiveType::UInt32));
        this->builder_.add_external("memmove", function_type, sdfg::LinkageType_External);
    }

    // Define dst
    llvm::Value* arg_dst = instruction->getArgOperand(0);
    assert(!utils::is_literal(arg_dst) && "IntrinsicLifting: Expect non-literal argument for memmove");
    std::string dst_name = utils::find_const_name_to_sdfg_name(this->constants_mapping_, arg_dst);
    auto& dst_node = this->builder_.add_access(current_state, dst_name, dbg_info);
    auto& dst_type = this->builder_.subject().type(dst_name);

    // Define src
    llvm::Value* arg_src = instruction->getArgOperand(1);
    assert(!utils::is_literal(arg_src) && "IntrinsicLifting: Expect non-literal argument for memmove");
    std::string src_name = utils::find_const_name_to_sdfg_name(this->constants_mapping_, arg_src);
    auto& src_node = this->builder_.add_access(current_state, src_name, dbg_info);
    auto& src_type = this->builder_.subject().type(src_name);

    // Define count
    llvm::Value* arg_count = instruction->getArgOperand(2);
    sdfg::symbolic::Expression count_sym;
    if (utils::is_literal(arg_count)) {
        count_sym = utils::as_symbol(llvm::dyn_cast<const llvm::ConstantData>(arg_count));
    } else {
        std::string arg_name = utils::find_const_name_to_sdfg_name(this->constants_mapping_, arg_count);
        count_sym = sdfg::symbolic::symbol(arg_name);
    }

    // Define library node
    auto& lib_node = this->builder_.add_library_node<sdfg::stdlib::MemmoveNode>(current_state, dbg_info, count_sym);
    this->builder_.add_computational_memlet(current_state, src_node, lib_node, "_src", {}, src_type, dbg_info);
    this->builder_.add_computational_memlet(current_state, lib_node, "_dst", dst_node, {}, dst_type, dbg_info);

    return current_state;
}

sdfg::control_flow::State& IntrinsicLifting::visit_memset(
    const llvm::BasicBlock* block, const llvm::IntrinsicInst* instruction, sdfg::control_flow::State& current_state
) {
    // Define Debug
    auto dbg_info = docc::utils::get_debug_info(*instruction);

    // Add function symbol
    if (!this->builder_.subject().exists("memset")) {
        sdfg::types::Function function_type(sdfg::types::Pointer(), false);
        function_type.add_param(sdfg::types::Pointer());
        function_type.add_param(sdfg::types::Scalar(sdfg::types::PrimitiveType::Int32));
        function_type.add_param(sdfg::types::Scalar(sdfg::types::PrimitiveType::UInt32));
        this->builder_.add_external("memset", function_type, sdfg::LinkageType_External);
    }

    // Define ptr
    llvm::Value* arg_ptr = instruction->getArgOperand(0);
    assert(!utils::is_literal(arg_ptr) && "IntrinsicLifting: Expect non-literal argument for memset");
    std::string ptr_name = utils::find_const_name_to_sdfg_name(this->constants_mapping_, arg_ptr);
    auto& ptr_node = this->builder_.add_access(current_state, ptr_name, dbg_info);
    auto& ptr_type = this->builder_.subject().type(ptr_name);

    // Define value
    llvm::Value* arg_value = instruction->getArgOperand(1);
    sdfg::symbolic::Expression value_sym;
    if (utils::is_literal(arg_value)) {
        value_sym = utils::as_symbol(llvm::dyn_cast<const llvm::ConstantData>(arg_value));
    } else {
        std::string arg_name = utils::find_const_name_to_sdfg_name(this->constants_mapping_, arg_value);
        value_sym = sdfg::symbolic::symbol(arg_name);
    }

    // Define num
    llvm::Value* arg_num = instruction->getArgOperand(2);
    sdfg::symbolic::Expression num_sym;
    if (utils::is_literal(arg_num)) {
        num_sym = utils::as_symbol(llvm::dyn_cast<const llvm::ConstantData>(arg_num));
    } else {
        std::string arg_name = utils::find_const_name_to_sdfg_name(this->constants_mapping_, arg_num);
        num_sym = sdfg::symbolic::symbol(arg_name);
    }

    // Define library node
    auto& lib_node =
        this->builder_.add_library_node<sdfg::stdlib::MemsetNode>(current_state, dbg_info, value_sym, num_sym);
    this->builder_.add_computational_memlet(current_state, lib_node, "_ptr", ptr_node, {}, ptr_type, dbg_info);

    return current_state;
}

sdfg::control_flow::State& IntrinsicLifting::visit_trap(
    const llvm::BasicBlock* block, const llvm::IntrinsicInst* instruction, sdfg::control_flow::State& current_state
) {
    // Define Debug
    auto dbg_info = docc::utils::get_debug_info(*instruction);

    // Define library node
    auto& lib_node = this->builder_.add_library_node<sdfg::stdlib::TrapNode>(current_state, dbg_info);

    return current_state;
}

} // namespace lifting
} // namespace docc
