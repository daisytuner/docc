#pragma once

#include <llvm/ADT/ScopedHashTable.h>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/function.h"
#include "sdfg/structured_control_flow/sequence.h"

namespace mlir {
namespace sdfg {

class TensorInfo {
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;
    int64_t offset_;

public:
    TensorInfo();
    TensorInfo(std::vector<int64_t> shape, std::vector<int64_t> strides, int64_t offset = 0);

    const std::vector<int64_t>& shape() const;
    const std::vector<int64_t>& strides() const;
    int64_t offset() const;

    /// Compute C-order contiguous strides from shape.
    static std::vector<int64_t> compute_strides(const std::vector<int64_t>& shape);

    /// Create TensorInfo from a tensor type (assumes C-order contiguous).
    static TensorInfo from_tensor_type(TensorType type);

    /// Create transposed view: output_strides[i] = input_strides[perm[i]].
    TensorInfo transpose(ArrayRef<int64_t> permutation) const;

    /// Create flipped view: negate stride and adjust offset for each flipped axis.
    TensorInfo flip(ArrayRef<int64_t> axes) const;

    /// Returns true iff a reshape is valid (contiguous layout, same element count).
    bool is_reshape_valid(ArrayRef<int64_t> new_shape) const;

    /// Create reshaped view (only valid for contiguous tensors).
    TensorInfo reshape(ArrayRef<int64_t> new_shape) const;
};

class SDFGTranslator {
    bool builder_empty_;
    ::sdfg::builder::StructuredSDFGBuilder builder_;

    llvm::ScopedHashTable<Value, std::string> value_map_;
    size_t value_counter_;

    ::sdfg::structured_control_flow::Sequence* insertion_point_;

    std::unordered_map<std::string, TensorInfo> tensor_info_map_;

public:
    SDFGTranslator();

    ::sdfg::builder::StructuredSDFGBuilder& builder();

    bool builder_empty();
    void builder_empty(bool empty);

    llvm::ScopedHashTable<Value, std::string>& value_map();

    ::sdfg::structured_control_flow::Sequence& insertion_point();
    void insertion_point(::sdfg::structured_control_flow::Sequence& sequence);

    std::unordered_map<std::string, TensorInfo>& tensor_info_map();

    std::string get_or_create_container(Value val, bool argument = false);

    std::unique_ptr<::sdfg::types::IType> convertType(const Type mlir_type);

    std::string convertTypedAttr(const TypedAttr attr);
};

LogicalResult translateOp(SDFGTranslator& translator, Operation* op);

LogicalResult emitJSON(SDFGTranslator& translator, raw_ostream& os);

} // namespace sdfg
} // namespace mlir
