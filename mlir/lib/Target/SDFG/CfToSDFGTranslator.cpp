#include "mlir/Target/SDFG/CfToSDFGTranslator.h"

#include <llvm/ADT/TypeSwitch.h>
#include <string>

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/SDFG/SDFGTranslator.h"
#include "sdfg/data_flow/library_nodes/stdlib/assert.h"
#include "sdfg/element.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/type.h"

namespace mlir {
namespace sdfg {

LogicalResult translateCfAssertOp(SDFGTranslator& translator, cf::AssertOp* assert_op) {
    Value arg = assert_op->getArg();
    auto deb_info = translator.get_debug_info(assert_op->getOperationName(), assert_op->getLoc());

    auto arg_container = translator.get_or_create_container(arg);
    ::sdfg::types::Scalar bool_type(::sdfg::types::PrimitiveType::Bool);

    auto& builder = translator.builder();
    auto& block = builder.add_block(translator.insertion_point(), {}, deb_info);
    auto& arg_access = builder.add_access(block, arg_container, deb_info);
    auto& libnode = builder.add_library_node<::sdfg::stdlib::AssertNode>(block, deb_info, assert_op->getMsg().data());
    builder.add_computational_memlet(block, arg_access, libnode, "_arg", {}, bool_type, deb_info);

    return success();
}

LogicalResult translateCfOp(SDFGTranslator& translator, Operation* op) {
    return llvm::TypeSwitch<Operation*, LogicalResult>(op)
        .Case<cf::AssertOp>([&](cf::AssertOp assert_op) { return translateCfAssertOp(translator, &assert_op); })
        .Default([&](Operation* op) {
            return op->emitError("Unknown operation from linalg dialect encountered: ") << op->getName();
        });
}

} // namespace sdfg
} // namespace mlir
