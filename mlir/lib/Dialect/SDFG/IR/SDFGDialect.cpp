#include <llvm/Support/Casting.h>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SDFG/IR/SDFG.h"
#include "mlir/IR/BuiltinTypes.h"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/SDFG/IR/SDFGOpsAttributes.cpp.inc"

// SDFG Dialect
#include "mlir/Dialect/SDFG/IR/SDFGOpsDialect.cpp.inc"

// SDFG Dialect Types
#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/SDFG/IR/SDFGOpsTypes.cpp.inc"

namespace mlir {
namespace sdfg {

bool is_primitive(const Type type) {
    if (auto int_type = llvm::dyn_cast<IntegerType>(type)) {
        switch (int_type.getWidth()) {
            case 1:
            case 8:
            case 16:
            case 32:
            case 64:
            case 128:
                return true;
            default:
                return false;
        }
    }
    return type.isF16() || type.isBF16() || type.isF32() || type.isF64() || type.isF80() || type.isF128();
}

bool is_tensor_of_primitive(const Type type) {
    if (auto tensor_type = llvm::dyn_cast<TensorType>(type)) {
        return is_primitive(tensor_type.getElementType());
    }
    return false;
}

void sdfg::SDFGDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/SDFG/IR/SDFGOps.cpp.inc"
        >();
}

} // namespace sdfg
} // namespace mlir
