#pragma once

#include "mlir/IR/Types.h"

namespace mlir {
namespace sdfg {

bool is_sdfg_primitive(Type type);
bool is_tensor_of_sdfg_primitive(Type type);

} // namespace sdfg
} // namespace mlir
