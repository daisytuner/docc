#include "mlir/Target/SDFG/FuncToSDFGTranslator.h"

#include <llvm-19/llvm/Support/Casting.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/LogicalResult.h>
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/SDFG/SDFGTranslator.h"

namespace mlir {
namespace sdfg {

LogicalResult translateFuncFuncOp(SDFGTranslator& translator, func::FuncOp* func_op) {
    if (!translator.builder_empty()) {
        return func_op->emitOpError("Currently only one function is supported");
    }

    // Check that at most one type is returned
    if (func_op->getFunctionType().getNumResults() > 1) {
        return func_op->emitOpError("Only one result type is supported for SDFGs");
    }

    std::string sdfg_name = func_op->getSymName().data();
    translator.builder().subject().name("__docc_" + sdfg_name);
    translator.builder_empty(false);
    llvm::ScopedHashTableScope<Value, std::string> value_map_scope(translator.value_map());

    // Return type
    auto function_type = func_op->getFunctionType();
    assert(function_type.getNumResults() <= 1);
    if (function_type.getNumResults() == 1) {
        auto return_type = translator.convertType(function_type.getResult(0));
        if (!return_type) {
            return func_op->emitError("Could not convert type ") << function_type.getResult(0) << " to SDFG type";
        }
        translator.builder().set_return_type(*return_type);
    }

    // Arguments
    for (auto arg : func_op->getRegion().getArguments()) {
        translator.get_or_create_container(arg, true);
    }

    // Region
    translator.enter_sequence(translator.builder().subject().root());
    for (auto& op : func_op->getRegion().getOps()) {
        if (failed(translateOp(translator, &op))) {
            return failure();
        }
    }
    translator.exit_sequence(translator.builder().subject().root());

    return success();
}

LogicalResult translateFuncReturnOp(SDFGTranslator& translator, func::ReturnOp* return_op) {
    if (return_op->getOperands().size() == 1) {
        auto return_container = translator.get_or_create_container(return_op->getOperand(0));
        bool isa_tensor = llvm::isa<TensorType>(return_op->getOperand(0).getType());
        if (isa_tensor) {
            return_container = translator.store_in_c_order(
                return_container,
                translator.get_or_create_tensor_info(
                    return_container, llvm::dyn_cast<TensorType>(return_op->getOperand(0).getType())
                ),
                translator.convertType(return_op->getOperand(0).getType())->primitive_type()
            );
        }
        translator.handle_frees(return_container);
        translator.builder().add_return(translator.insertion_point(), return_container);
        if (isa_tensor) {
            auto result_tensor_type = llvm::dyn_cast<TensorType>(return_op->getOperand(0).getType());
            translator.builder().subject().add_metadata(
                "return_shape", translator.get_or_create_tensor_info(return_container, result_tensor_type).shape_str()
            );
        }
    } else if (return_op->getOperands().size() > 1) {
        return return_op->emitOpError("Only one result type is supported for SDFGs");
    }
    return success();
}

LogicalResult translateFuncOp(SDFGTranslator& translator, Operation* op) {
    return llvm::TypeSwitch<Operation*, LogicalResult>(op)
        .Case<func::FuncOp>([&](func::FuncOp func_op) { return translateFuncFuncOp(translator, &func_op); })
        .Case<func::ReturnOp>([&](func::ReturnOp return_op) { return translateFuncReturnOp(translator, &return_op); })
        .Default([&](Operation* op) { return op->emitError("Unknown operation from func dialect encountered"); });
}

} // namespace sdfg
} // namespace mlir
