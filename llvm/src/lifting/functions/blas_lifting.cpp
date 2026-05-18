#include "docc/lifting/functions/blas_lifting.h"

namespace docc {
namespace lifting {

sdfg::math::blas::BLAS_Layout as_blas_layout(llvm::Value* layout) {
    if (llvm::isa<llvm::ConstantInt>(layout)) {
        auto layout_int = llvm::dyn_cast<llvm::ConstantInt>(layout)->getZExtValue();
        return static_cast<sdfg::math::blas::BLAS_Layout>(layout_int);
    }
    throw NotImplementedException(
        "BLASLifting: Invalid layout", docc::utils::bestEffortLoc(*layout), docc::utils::toIRString(*layout)
    );
}

sdfg::math::blas::BLAS_Transpose as_blas_transpose(llvm::Value* transpose) {
    if (llvm::isa<llvm::ConstantInt>(transpose)) {
        auto transpose_int = llvm::dyn_cast<llvm::ConstantInt>(transpose)->getZExtValue();
        return static_cast<sdfg::math::blas::BLAS_Transpose>(transpose_int);
    }
    throw NotImplementedException(
        "BLASLifting: Invalid transpose", docc::utils::bestEffortLoc(*transpose), docc::utils::toIRString(*transpose)
    );
}

sdfg::math::blas::BLAS_Precision as_blas_precision(llvm::Function* called_func) {
    if (called_func->getName().starts_with("cblas_d")) {
        return sdfg::math::blas::BLAS_Precision::d;
    } else if (called_func->getName().starts_with("cblas_s")) {
        return sdfg::math::blas::BLAS_Precision::s;
    }

    throw NotImplementedException(
        "Unsupported BLAS precision", docc::utils::get_debug_info(*called_func), docc::utils::toIRString(*called_func)
    );
}

sdfg::symbolic::Expression as_symbolic_expression(llvm::Value* value) {
    if (utils::is_symbol(value)) {
        return utils::as_symbol(llvm::dyn_cast<llvm::ConstantData>(value));
    }
    std::string name = utils::get_name(value);
    return sdfg::symbolic::symbol(name);
}


sdfg::control_flow::State& BLASLifting::
    visit(const llvm::BasicBlock* block, const llvm::CallBase* instruction, sdfg::control_flow::State& current_state) {
    auto called_func = instruction->getCalledFunction();
    auto code = BLASLifting::library_node_code(called_func->getName());
    if (code == sdfg::math::blas::LibraryNodeType_DOT) {
        return visit_dot(block, instruction, current_state);
    } else if (code == sdfg::math::blas::LibraryNodeType_GEMM) {
        return visit_gemm(block, instruction, current_state);
    }

    throw NotImplementedException(
        "BLASLifting: Unsupported BLAS function",
        docc::utils::get_debug_info(*instruction),
        docc::utils::toIRString(*instruction)
    );
}

sdfg::control_flow::State& BLASLifting::visit_dot(
    const llvm::BasicBlock* block, const llvm::CallBase* instruction, sdfg::control_flow::State& current_state
) {
    llvm::Function* called_func = instruction->getCalledFunction();
    sdfg::math::blas::BLAS_Precision precision = as_blas_precision(called_func);

    sdfg::symbolic::Expression n = as_symbolic_expression(instruction->getArgOperand(0));
    sdfg::symbolic::Expression incx = as_symbolic_expression(instruction->getArgOperand(2));
    sdfg::symbolic::Expression incy = as_symbolic_expression(instruction->getArgOperand(4));

    auto cblas_x_sdfg = utils::get_name(instruction->getArgOperand(1));
    assert(this->builder_.subject().exists(cblas_x_sdfg));
    auto cblas_y_sdfg = utils::get_name(instruction->getArgOperand(3));
    assert(this->builder_.subject().exists(cblas_y_sdfg));

    auto dbg_info = docc::utils::get_debug_info(*instruction);
    auto& x_node_in = this->builder_.add_access(current_state, cblas_x_sdfg, dbg_info);
    auto& y_node_in = this->builder_.add_access(current_state, cblas_y_sdfg, dbg_info);

    auto& dot_node = this->builder_.add_library_node<sdfg::math::blas::DotNode>(
        current_state, dbg_info, sdfg::math::blas::ImplementationType_BLAS, precision, n, incx, incy
    );

    sdfg::types::Scalar base_type(
        precision == sdfg::math::blas::BLAS_Precision::d ? sdfg::types::PrimitiveType::Double
                                                         : sdfg::types::PrimitiveType::Float
    );
    sdfg::types::Pointer ptr_type(base_type);
    this->builder_.add_computational_memlet(current_state, x_node_in, dot_node, "__x", {}, ptr_type, dbg_info);
    this->builder_.add_computational_memlet(current_state, y_node_in, dot_node, "__y", {}, ptr_type, dbg_info);

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
    auto& output_node = this->builder_.add_access(current_state, output, dbg_info);
    auto& output_type = dynamic_cast<const sdfg::types::Scalar&>(this->builder_.subject().type(output_node.data()));

    this->builder_.add_computational_memlet(current_state, dot_node, "__out", output_node, {}, base_type, dbg_info);

    return current_state;
}

sdfg::control_flow::State& BLASLifting::visit_gemm(
    const llvm::BasicBlock* block, const llvm::CallBase* instruction, sdfg::control_flow::State& current_state
) {
    // Determine BLAS precision
    llvm::Function* called_func = instruction->getCalledFunction();
    sdfg::math::blas::BLAS_Precision precision = as_blas_precision(called_func);

    // Helper arguments
    sdfg::math::blas::BLAS_Layout layout = as_blas_layout(instruction->getArgOperand(0));
    sdfg::math::blas::BLAS_Transpose trans_a = as_blas_transpose(instruction->getArgOperand(1));
    sdfg::math::blas::BLAS_Transpose trans_b = as_blas_transpose(instruction->getArgOperand(2));
    sdfg::symbolic::Expression m = as_symbolic_expression(instruction->getArgOperand(3));
    sdfg::symbolic::Expression n = as_symbolic_expression(instruction->getArgOperand(4));
    sdfg::symbolic::Expression k = as_symbolic_expression(instruction->getArgOperand(5));
    sdfg::symbolic::Expression lda = as_symbolic_expression(instruction->getArgOperand(8));
    sdfg::symbolic::Expression ldb = as_symbolic_expression(instruction->getArgOperand(10));
    sdfg::symbolic::Expression ldc = as_symbolic_expression(instruction->getArgOperand(13));

    sdfg::types::Scalar base_type(
        precision == sdfg::math::blas::BLAS_Precision::d ? sdfg::types::PrimitiveType::Double
                                                         : sdfg::types::PrimitiveType::Float
    );

    auto cblas_alpha = instruction->getArgOperand(6);
    sdfg::data_flow::AccessNode* alpha_node = nullptr;
    if (utils::is_literal(cblas_alpha)) {
        std::string cblas_alpha_sdfg = utils::as_literal(llvm::dyn_cast<llvm::ConstantData>(cblas_alpha));
        alpha_node =
            &this->builder_
                 .add_constant(current_state, cblas_alpha_sdfg, base_type, docc::utils::get_debug_info(*instruction));
    } else {
        std::string cblas_alpha_sdfg = utils::get_name(cblas_alpha);
        assert(this->builder_.subject().exists(cblas_alpha_sdfg));
        alpha_node =
            &this->builder_.add_access(current_state, cblas_alpha_sdfg, docc::utils::get_debug_info(*instruction));
    }

    auto cblas_a_sdfg = utils::get_name(instruction->getArgOperand(7));
    assert(this->builder_.subject().exists(cblas_a_sdfg));
    auto cblas_b_sdfg = utils::get_name(instruction->getArgOperand(9));
    assert(this->builder_.subject().exists(cblas_b_sdfg));

    auto cblas_beta = instruction->getArgOperand(11);
    sdfg::data_flow::AccessNode* beta_node = nullptr;
    if (utils::is_literal(cblas_beta)) {
        std::string cblas_beta_sdfg = utils::as_literal(llvm::dyn_cast<llvm::ConstantData>(cblas_beta));
        beta_node =
            &this->builder_
                 .add_constant(current_state, cblas_beta_sdfg, base_type, docc::utils::get_debug_info(*instruction));
    } else {
        std::string cblas_beta_sdfg = utils::get_name(cblas_beta);
        assert(this->builder_.subject().exists(cblas_beta_sdfg));
        beta_node =
            &this->builder_.add_access(current_state, cblas_beta_sdfg, docc::utils::get_debug_info(*instruction));
    }

    auto cblas_c_sdfg = utils::get_name(instruction->getArgOperand(12));
    assert(this->builder_.subject().exists(cblas_c_sdfg));

    // Add GEMM node
    auto dbg_info = docc::utils::get_debug_info(*instruction);
    auto& gemm_node = this->builder_.add_library_node<sdfg::math::blas::GEMMNode>(
        current_state,
        dbg_info,
        sdfg::math::blas::ImplementationType_BLAS,
        precision,
        layout,
        trans_a,
        trans_b,
        m,
        n,
        k,
        lda,
        ldb,
        ldc
    );

    sdfg::types::Pointer ptr_type(base_type);

    // Add access nodes
    auto& a_node_in = this->builder_.add_access(current_state, cblas_a_sdfg, dbg_info);
    auto& b_node_in = this->builder_.add_access(current_state, cblas_b_sdfg, dbg_info);
    auto& c_node_in = this->builder_.add_access(current_state, cblas_c_sdfg, dbg_info);
    auto& c_node_out = this->builder_.add_access(current_state, cblas_c_sdfg, dbg_info);

    // Add edges
    this->builder_.add_computational_memlet(current_state, a_node_in, gemm_node, "__A", {}, ptr_type, dbg_info);
    this->builder_.add_computational_memlet(current_state, b_node_in, gemm_node, "__B", {}, ptr_type, dbg_info);
    this->builder_.add_computational_memlet(current_state, c_node_in, gemm_node, "__C", {}, ptr_type, dbg_info);
    this->builder_.add_computational_memlet(current_state, gemm_node, "__C", c_node_out, {}, ptr_type, dbg_info);
    this->builder_.add_computational_memlet(current_state, *alpha_node, gemm_node, "__alpha", {}, base_type, dbg_info);
    this->builder_.add_computational_memlet(current_state, *beta_node, gemm_node, "__beta", {}, base_type, dbg_info);

    return current_state;
}

} // namespace lifting
} // namespace docc
