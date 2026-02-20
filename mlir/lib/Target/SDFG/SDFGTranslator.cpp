#include "mlir/Target/SDFG/SDFGTranslator.h"
#include <llvm/ADT/TypeSwitch.h>
#include <stdexcept>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Target/SDFG/BuiltinToSDFGTranslator.h"
#include "mlir/Target/SDFG/FuncToSDFGTranslator.h"
#include "mlir/Target/SDFG/LinalgToSDFGTranslator.h"
#include "sdfg/builder/structured_sdfg_builder.h"

namespace mlir {
namespace sdfg {

// ===----------------------------------------------------------------------===//
// TensorInfo
// ===----------------------------------------------------------------------===//

TensorInfo::TensorInfo() : offset_(0) {}

TensorInfo::TensorInfo(std::vector<int64_t> shape, std::vector<int64_t> strides, int64_t offset)
    : shape_(std::move(shape)), strides_(std::move(strides)), offset_(offset) {}

const std::vector<int64_t>& TensorInfo::shape() const { return shape_; }

const std::vector<int64_t>& TensorInfo::strides() const { return strides_; }

int64_t TensorInfo::offset() const { return offset_; }

std::vector<int64_t> TensorInfo::compute_strides(const std::vector<int64_t>& shape) {
    if (shape.empty()) {
        return {};
    }
    std::vector<int64_t> strides(shape.size());
    int64_t stride = 1;
    for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}

TensorInfo TensorInfo::from_tensor_type(TensorType type) {
    std::vector<int64_t> shape(type.getShape().begin(), type.getShape().end());
    std::vector<int64_t> strides = compute_strides(shape);
    return TensorInfo(std::move(shape), std::move(strides), 0);
}

TensorInfo TensorInfo::transpose(ArrayRef<int64_t> permutation) const {
    std::vector<int64_t> new_shape;
    std::vector<int64_t> new_strides;
    new_shape.reserve(permutation.size());
    new_strides.reserve(permutation.size());
    for (int64_t p : permutation) {
        new_shape.push_back(shape_[p]);
        new_strides.push_back(strides_[p]);
    }
    return TensorInfo(std::move(new_shape), std::move(new_strides), offset_);
}

TensorInfo TensorInfo::flip(ArrayRef<int64_t> axes) const {
    TensorInfo result = *this;
    for (int64_t axis : axes) {
        result.offset_ += (shape_[axis] - 1) * strides_[axis];
        result.strides_[axis] = -strides_[axis];
    }
    return result;
}

bool TensorInfo::is_reshape_valid(ArrayRef<int64_t> new_shape) const {
    int64_t old_num_elements = 1;
    for (int64_t dim : shape_) {
        old_num_elements *= dim;
    }
    int64_t new_num_elements = 1;
    for (int64_t dim : new_shape) {
        new_num_elements *= dim;
    }
    if (old_num_elements != new_num_elements) {
        return false;
    }
    auto expected_strides = compute_strides(shape_);
    if (expected_strides.size() != strides_.size()) {
        return false;
    }
    for (size_t i = 0; i < expected_strides.size(); ++i) {
        if (expected_strides[i] != strides_[i]) {
            return false;
        }
    }
    return true;
}

TensorInfo TensorInfo::reshape(ArrayRef<int64_t> new_shape) const {
    std::vector<int64_t> shape(new_shape.begin(), new_shape.end());
    std::vector<int64_t> strides = compute_strides(shape);
    return TensorInfo(std::move(shape), std::move(strides), offset_);
}

// ===----------------------------------------------------------------------===//
// SDFGTranslator
// ===----------------------------------------------------------------------===//

SDFGTranslator::SDFGTranslator()
    : builder_empty_(true), builder_("empty", ::sdfg::FunctionType_CPU), value_counter_(0), insertion_point_(nullptr) {}

::sdfg::builder::StructuredSDFGBuilder& SDFGTranslator::builder() { return this->builder_; }

bool SDFGTranslator::builder_empty() { return this->builder_empty_; }

void SDFGTranslator::builder_empty(bool empty) { this->builder_empty_ = empty; }

std::string SDFGTranslator::get_or_create_container(Value val, bool argument) {
    if (!this->value_map_.count(val)) {
        this->value_map_.insert(val, "_" + std::to_string(value_counter_++));
    }
    std::string container = *this->value_map_.begin(val);
    auto type = convertType(val.getType());
    if (builder_.subject().exists(container)) {
        assert(builder_.subject().type(container) == *type);
        assert(!argument || builder_.subject().is_argument(container));
    } else {
        builder_.add_container(container, *type, argument);
    }
    return container;
}

llvm::ScopedHashTable<Value, std::string>& SDFGTranslator::value_map() { return this->value_map_; }

std::unordered_map<std::string, TensorInfo>& SDFGTranslator::tensor_info_map() { return this->tensor_info_map_; }

::sdfg::structured_control_flow::Sequence& SDFGTranslator::insertion_point() {
    if (this->insertion_point_ == nullptr) {
        std::runtime_error("Tried accessing insertion point but is nullptr");
    }
    return *this->insertion_point_;
}

void SDFGTranslator::insertion_point(::sdfg::structured_control_flow::Sequence& sequence) {
    this->insertion_point_ = &sequence;
}

std::unique_ptr<::sdfg::types::IType> SDFGTranslator::convertType(const Type mlir_type) {
    if (mlir_type.isInteger(1)) {
        return std::make_unique<::sdfg::types::Scalar>(::sdfg::types::PrimitiveType::Bool);
    } else if (mlir_type.isSignedInteger(8) || mlir_type.isSignlessInteger(8)) {
        return std::make_unique<::sdfg::types::Scalar>(::sdfg::types::PrimitiveType::Int8);
    } else if (mlir_type.isSignedInteger(16) || mlir_type.isSignlessInteger(16)) {
        return std::make_unique<::sdfg::types::Scalar>(::sdfg::types::PrimitiveType::Int16);
    } else if (mlir_type.isSignedInteger(32) || mlir_type.isSignlessInteger(32)) {
        return std::make_unique<::sdfg::types::Scalar>(::sdfg::types::PrimitiveType::Int32);
    } else if (mlir_type.isSignedInteger(64) || mlir_type.isSignlessInteger(64)) {
        return std::make_unique<::sdfg::types::Scalar>(::sdfg::types::PrimitiveType::Int64);
    } else if (mlir_type.isSignedInteger(128) || mlir_type.isSignlessInteger(128)) {
        return std::make_unique<::sdfg::types::Scalar>(::sdfg::types::PrimitiveType::Int128);
    } else if (mlir_type.isUnsignedInteger(8)) {
        return std::make_unique<::sdfg::types::Scalar>(::sdfg::types::PrimitiveType::UInt8);
    } else if (mlir_type.isUnsignedInteger(16)) {
        return std::make_unique<::sdfg::types::Scalar>(::sdfg::types::PrimitiveType::UInt16);
    } else if (mlir_type.isUnsignedInteger(32)) {
        return std::make_unique<::sdfg::types::Scalar>(::sdfg::types::PrimitiveType::UInt32);
    } else if (mlir_type.isUnsignedInteger(64)) {
        return std::make_unique<::sdfg::types::Scalar>(::sdfg::types::PrimitiveType::UInt64);
    } else if (mlir_type.isUnsignedInteger(128)) {
        return std::make_unique<::sdfg::types::Scalar>(::sdfg::types::PrimitiveType::UInt128);
    } else if (mlir_type.isF16()) {
        return std::make_unique<::sdfg::types::Scalar>(::sdfg::types::PrimitiveType::Half);
    } else if (mlir_type.isBF16()) {
        return std::make_unique<::sdfg::types::Scalar>(::sdfg::types::PrimitiveType::BFloat);
    } else if (mlir_type.isF32()) {
        return std::make_unique<::sdfg::types::Scalar>(::sdfg::types::PrimitiveType::Float);
    } else if (mlir_type.isF64()) {
        return std::make_unique<::sdfg::types::Scalar>(::sdfg::types::PrimitiveType::Double);
    } else if (mlir_type.isF80()) {
        return std::make_unique<::sdfg::types::Scalar>(::sdfg::types::PrimitiveType::X86_FP80);
    } else if (mlir_type.isF128()) {
        return std::make_unique<::sdfg::types::Scalar>(::sdfg::types::PrimitiveType::FP128);
    } else if (auto vector_type = dyn_cast_or_null<VectorType>(mlir_type)) {
        auto base_type = this->convertType(vector_type.getElementType());
        if (!base_type) {
            return nullptr;
        }
        return std::make_unique<::sdfg::types::Pointer>(*base_type);
    } else if (auto tensor_type = dyn_cast_or_null<TensorType>(mlir_type)) {
        auto base_type = this->convertType(tensor_type.getElementType());
        if (!base_type) {
            return nullptr;
        }
        return std::make_unique<::sdfg::types::Pointer>(*base_type);
    }
    return nullptr;
}

std::string SDFGTranslator::convertTypedAttr(const TypedAttr attr) {
    return llvm::TypeSwitch<TypedAttr, std::string>(attr)
        .Case<FloatAttr>([](FloatAttr attr) { return std::to_string(attr.getValue().convertToDouble()); })
        .Case<IntegerAttr>([](IntegerAttr attr) { return std::to_string(attr.getInt()); })
        .Default([](TypedAttr attr) { return ""; });
}

LogicalResult translateOp(SDFGTranslator& translator, Operation* op) {
    if (op->getDialect()->getNamespace() == BuiltinDialect::getDialectNamespace()) {
        return translateBuiltinOp(translator, op);
    } else if (op->getDialect()->getNamespace() == func::FuncDialect::getDialectNamespace()) {
        return translateFuncOp(translator, op);
    } else if (op->getDialect()->getNamespace() == linalg::LinalgDialect::getDialectNamespace()) {
        return translateLinalgOp(translator, op);
    }
    // Handle all others
    return op->emitOpError("Could not translate!");
}

LogicalResult emitJSON(SDFGTranslator& translator, raw_ostream& os) { return success(); }

} // namespace sdfg
} // namespace mlir
