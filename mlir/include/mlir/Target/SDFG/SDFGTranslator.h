#pragma once

#include <llvm/ADT/ScopedHashTable.h>
#include <optional>
#include <string>

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/function.h"
#include "sdfg/structured_control_flow/sequence.h"

namespace mlir {
namespace sdfg {

class SDFGTranslator {
    bool builder_empty_;
    ::sdfg::builder::StructuredSDFGBuilder builder_;

    llvm::ScopedHashTable<Value, std::string> value_map_;
    size_t value_counter_;

    ::sdfg::structured_control_flow::Sequence* insertion_point_;

public:
    SDFGTranslator();

    ::sdfg::builder::StructuredSDFGBuilder& builder();

    bool builder_empty();
    void builder_empty(bool empty);

    llvm::ScopedHashTable<Value, std::string>& value_map();

    ::sdfg::structured_control_flow::Sequence& insertion_point();
    void insertion_point(::sdfg::structured_control_flow::Sequence& sequence);

    std::string get_or_create_container(Value val, bool argument = false);

    std::unique_ptr<::sdfg::types::IType> convertType(const Type mlir_type);
};

LogicalResult translateOp(SDFGTranslator& translator, Operation* op);

LogicalResult emitJSON(SDFGTranslator& translator, raw_ostream& os);

} // namespace sdfg
} // namespace mlir
