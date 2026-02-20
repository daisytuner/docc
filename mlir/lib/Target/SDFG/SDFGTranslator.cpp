#include "mlir/Target/SDFG/SDFGTranslator.h"
#include <stdexcept>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Target/SDFG/BuiltinToSDFGTranslator.h"
#include "mlir/Target/SDFG/FuncToSDFGTranslator.h"
#include "sdfg/builder/structured_sdfg_builder.h"

namespace mlir {
namespace sdfg {

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

LogicalResult translateOp(SDFGTranslator& translator, Operation* op) {
    if (op->getDialect()->getNamespace() == BuiltinDialect::getDialectNamespace()) {
        return translateBuiltinOp(translator, op);
    } else if (op->getDialect()->getNamespace() == func::FuncDialect::getDialectNamespace()) {
        return translateFuncOp(translator, op);
    }
    // Handle all others
    return op->emitOpError("Could not translate!");
}

LogicalResult emitJSON(SDFGTranslator& translator, raw_ostream& os) { return success(); }

} // namespace sdfg
} // namespace mlir
