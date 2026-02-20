#include "mlir/Target/SDFG/helper.h"

#include <llvm/Support/Casting.h>

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace sdfg {

bool is_sdfg_primitive(Type type) {
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

bool is_tensor_of_sdfg_primitive(Type type) {
    if (auto tensor_type = llvm::dyn_cast<TensorType>(type)) {
        return is_sdfg_primitive(tensor_type.getElementType());
    }
    return false;
}

} // namespace sdfg
} // namespace mlir
