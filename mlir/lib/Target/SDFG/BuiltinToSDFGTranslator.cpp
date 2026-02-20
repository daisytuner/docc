#include "mlir/Target/SDFG/BuiltinToSDFGTranslator.h"

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/LogicalResult.h>
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Target/SDFG/SDFGTranslator.h"

namespace mlir {
namespace sdfg {

LogicalResult translateBuiltinModuleOp(SDFGTranslator& translator, ModuleOp* module_op) {
    for (auto& op : module_op->getRegion().getOps()) {
        LogicalResult status = translateOp(translator, &op);
        if (failed(status)) {
            return failure();
        }
    }
    return success();
}

LogicalResult translateBuiltinOp(SDFGTranslator& translator, Operation* op) {
    return llvm::TypeSwitch<Operation*, LogicalResult>(op)
        .Case<ModuleOp>([&](ModuleOp module_op) { return translateBuiltinModuleOp(translator, &module_op); })
        .Default([&](Operation* op) { return op->emitError("A module op is required here"); });
}

} // namespace sdfg
} // namespace mlir
