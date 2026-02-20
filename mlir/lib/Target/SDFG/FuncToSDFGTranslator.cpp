#include "mlir/Target/SDFG/FuncToSDFGTranslator.h"

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
    translator.insertion_point(translator.builder().subject().root());
    for (auto& op : func_op->getRegion().getOps()) {
        if (failed(translateOp(translator, &op))) {
            return failure();
        }
    }

    return success();
}

LogicalResult translateFuncReturnOp(SDFGTranslator& translator, func::ReturnOp* return_op) {
    if (return_op->getOperands().size() == 1) {
        translator.builder()
            .add_return(translator.insertion_point(), translator.get_or_create_container(return_op->getOperand(0)));
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
