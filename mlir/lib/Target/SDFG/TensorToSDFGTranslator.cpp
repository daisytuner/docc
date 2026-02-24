#include "mlir/Target/SDFG/TensorToSDFGTranslator.h"

#include <cstdint>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/LogicalResult.h>
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Target/SDFG/SDFGTranslator.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/scalar.h"

namespace mlir {
namespace sdfg {

LogicalResult translateTensorEmptyOp(SDFGTranslator& translator, tensor::EmptyOp* empty_op) {
    Value result = empty_op->getResult();
    auto result_tensor_type = llvm::dyn_cast<TensorType>(result.getType());

    std::string container = translator.get_or_create_container(result);
    auto tensor_info = translator.get_or_create_tensor_info(container, result_tensor_type);

    auto element_type = translator.convertType(result_tensor_type.getElementType());
    auto sdfg_tensor = tensor_info.get_sdfg_tensor(static_cast<::sdfg::types::Scalar&>(*element_type));

    uint64_t size = 1;
    for (int64_t dim : tensor_info.shape()) {
        size *= dim;
    }
    translator.handle_malloc(
        container, ::sdfg::symbolic::mul(::sdfg::symbolic::integer(size), ::sdfg::symbolic::size_of_type(*element_type))
    );

    return success();
}

LogicalResult translateTensorOp(SDFGTranslator& translator, Operation* op) {
    return llvm::TypeSwitch<Operation*, LogicalResult>(op)
        .Case<tensor::EmptyOp>([&](tensor::EmptyOp empty_op) { return translateTensorEmptyOp(translator, &empty_op); })
        .Default([&](Operation* op) { return op->emitError("Unknown operation from tensor dialect encountered"); });
}

} // namespace sdfg
} // namespace mlir
