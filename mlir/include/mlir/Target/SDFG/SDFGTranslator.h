#pragma once

#include <list>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/ScopedHashTable.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/element.h"
#include "sdfg/function.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/tensor.h"

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

    /// Returns true iff the tensor has basic C-order contiguous strides.
    static bool has_basic_strides(ArrayRef<int64_t> shape, ArrayRef<int64_t> strides);

    /// Create transposed view: output_strides[i] = input_strides[perm[i]].
    TensorInfo transpose(ArrayRef<int64_t> permutation) const;

    /// Create flipped view: negate stride and adjust offset for each flipped axis.
    TensorInfo flip(ArrayRef<int64_t> axes) const;

    /// Returns true iff a reshape is valid (contiguous layout, same element count).
    bool is_reshape_valid(ArrayRef<int64_t> new_shape) const;

    /// Returns true iff the tensor has basic C-order contiguous strides.
    bool has_basic_strides() const;

    /// Returns true iff the tensor is transposed in the last two dimensions.
    bool has_transposed_strides_last_two_dims() const;

    /// Create reshaped view (only valid for contiguous tensors).
    TensorInfo reshape(ArrayRef<int64_t> new_shape) const;

    /// return shape as string for metadata
    std::string shape_str() const;

    /// Create SDFG tensor type
    std::unique_ptr<::sdfg::types::Tensor> get_sdfg_tensor(const ::sdfg::types::Scalar& element_type) const;

    /// Create SDFG tensor layout
    ::sdfg::math::tensor::TensorLayout get_tensor_layout() const;

    /// Return string representation of tensor info (for debug purposes)
    std::string toStr() const;
};

class SDFGTranslator {
    bool builder_empty_;
    ::sdfg::builder::StructuredSDFGBuilder builder_;

    llvm::ScopedHashTable<Value, std::string> value_map_;
    size_t value_counter_;

    std::list<::sdfg::structured_control_flow::Sequence*> insertion_points_;

    std::unordered_map<std::string, TensorInfo> tensor_info_map_;

    std::unordered_map<::sdfg::structured_control_flow::Sequence*, std::list<std::string>> memory_map_;

    std::unordered_map<std::string, std::string> alias_map_;

    // Maps a linalg.fill result whose materialization was deferred to the constant scalar value it
    // fills with, so each consumer can regenerate a freshly-filled array instead of copying from a
    // single shared buffer.
    llvm::DenseMap<Value, Value> constant_fill_map_;

    std::vector<std::string> output_args_;

public:
    SDFGTranslator();

    ::sdfg::builder::StructuredSDFGBuilder& builder();

    bool builder_empty();
    void builder_empty(bool empty);

    llvm::ScopedHashTable<Value, std::string>& value_map();

    ::sdfg::structured_control_flow::Sequence& insertion_point();
    void enter_sequence(::sdfg::structured_control_flow::Sequence& sequence);
    void exit_sequence(::sdfg::structured_control_flow::Sequence& sequence);

    std::unordered_map<std::string, TensorInfo>& tensor_info_map();

    TensorInfo& get_or_create_tensor_info(const std::string& container, const TensorType& type);

    std::string get_or_create_container(Value val, bool argument = false);

    std::unique_ptr<::sdfg::types::IType> convertType(const Type mlir_type);

    std::string convertTypedAttr(const TypedAttr attr);

    void add_reference(
        const std::string& src_container,
        const std::string& dst_container,
        const ::sdfg::DebugInfo& deb_info = ::sdfg::DebugInfo()
    );

    void handle_malloc(
        std::string container,
        const ::sdfg::symbolic::Expression size,
        const ::sdfg::DebugInfo& deb_info = ::sdfg::DebugInfo()
    );
    void handle_frees(std::string return_container = "", const ::sdfg::DebugInfo& deb_info = ::sdfg::DebugInfo());

    /// If `output` is used as a DPS init by more than one linalg op, allocate a fresh
    /// copy via malloc + memcpy and return the new container name.
    /// Otherwise return the original container for `output`.
    /// If `consumer_overwrites_output` is true, the caller guarantees it fully overwrites the
    /// returned buffer before reading it (e.g. matmul with beta=0), so the init copy is skipped
    /// and a fresh uninitialized buffer is handed out (the malloc still happens).
    std::string get_or_copy_output_container(
        Value output, const ::sdfg::DebugInfo& deb_info = ::sdfg::DebugInfo(), bool consumer_overwrites_output = false
    );

    /// Record that `result` is produced by a constant `value` linalg.fill whose materialization is
    /// deferred. Each consumer regenerates a freshly-filled array on demand instead of copying from
    /// a single shared buffer (saving the source buffer and turning per-use memcpys into fills).
    void record_constant_fill(Value result, Value value);

    /// Returns true if `result` was recorded via record_constant_fill.
    bool is_constant_fill(Value result) const;

    /// Returns the container of `input` if its buffer can be safely overwritten in place by an
    /// elementwise consumer producing `result`, otherwise the empty string. Conservatively limited
    /// to inputs produced by a matmul/batch_matmul: a fresh, fully-written, owned buffer used
    /// exactly once and matching the result's shape and element type. Lets the consumer reuse that
    /// buffer instead of allocating a new output, saving a malloc (and its free).
    std::string try_inplace_reuse_container(Value input, Value result);

    std::string store_in_c_order(
        const std::string& container,
        const TensorInfo& tensor_info,
        const ::sdfg::types::Scalar& element_type,
        const ::sdfg::DebugInfo& deb_info = ::sdfg::DebugInfo()
    );

    /// Set the output argument names for multi-output support
    void set_output_args(const std::vector<std::string>& output_args);

    /// Get the output argument names
    const std::vector<std::string>& output_args() const;

    /// Copy tensor data to an output argument container in C-order
    void copy_to_output(
        const std::string& src_container,
        const TensorInfo& tensor_info,
        const ::sdfg::types::Scalar& element_type,
        const std::string& output_container,
        const ::sdfg::DebugInfo& deb_info = ::sdfg::DebugInfo()
    );

    /// Copy scalar data to an output argument container
    void copy_scalar_to_output(
        const std::string& src_container,
        const std::string& output_container,
        const ::sdfg::DebugInfo& deb_info = ::sdfg::DebugInfo()
    );

    /// Creates an SDFG debug info from an operation
    ::sdfg::DebugInfo get_debug_info(llvm::StringLiteral operation_name, Location loc);
};

LogicalResult translateOp(SDFGTranslator& translator, Operation* op);

LogicalResult emitJSON(SDFGTranslator& translator, raw_ostream& os);

} // namespace sdfg
} // namespace mlir
