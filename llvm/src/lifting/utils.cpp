#include "docc/lifting/utils.h"

#include <llvm/IR/Constants.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/TypedPointerType.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>
#include <sdfg/codegen/language_extensions/cpp_language_extension.h>
#include <sdfg/types/type.h>

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>
#include "docc/utils.h"
#include "sdfg/element.h"

namespace docc {
namespace lifting {
namespace utils {

std::string get_name(const llvm::Value* value) {
    std::string llvm_name;

    if (!value->getName().empty()) {
        llvm_name = std::string(value->getName());
    }

    if (llvm_name.empty()) {
        llvm::raw_string_ostream OS(llvm_name);
        value->printAsOperand(OS, false);
    }

    return normalize_name(llvm_name);
}

std::string normalize_name(const std::string value) {
    std::string normalized_name = value;
    std::replace(normalized_name.begin(), normalized_name.end(), '.', '_');
    std::replace(normalized_name.begin(), normalized_name.end(), ':', '_');
    std::replace(normalized_name.begin(), normalized_name.end(), '-', '_');
    std::replace(normalized_name.begin(), normalized_name.end(), '%', '_');
    std::replace(normalized_name.begin(), normalized_name.end(), '@', '_');
    std::replace(normalized_name.begin(), normalized_name.end(), '(', '_');
    std::replace(normalized_name.begin(), normalized_name.end(), ')', '_');
    std::replace(normalized_name.begin(), normalized_name.end(), '&', '_');

    std::string tmp;
    std::remove_copy(normalized_name.begin(), normalized_name.end(), std::back_inserter(tmp), '<');
    normalized_name = tmp;
    tmp.clear();

    std::remove_copy(normalized_name.begin(), normalized_name.end(), std::back_inserter(tmp), '>');
    normalized_name = tmp;
    tmp.clear();

    std::remove_copy(normalized_name.begin(), normalized_name.end(), std::back_inserter(tmp), ' ');
    normalized_name = tmp;
    tmp.clear();

    std::remove_copy(normalized_name.begin(), normalized_name.end(), std::back_inserter(tmp), '*');
    normalized_name = tmp;
    tmp.clear();

    std::remove_copy(normalized_name.begin(), normalized_name.end(), std::back_inserter(tmp), ',');
    normalized_name = tmp;
    tmp.clear();

    if (normalized_name == "badref") {
        return "";
    }

    if (normalized_name == "this") {
        return "self";
    }

    return normalized_name;
}

bool is_literal(const llvm::Value* value) {
    return llvm::isa<llvm::ConstantData>(value) || llvm::isa<llvm::ConstantAggregate>(value);
}

bool is_null_pointer(const llvm::Value* value) { return llvm::dyn_cast<llvm::ConstantPointerNull>(value) != nullptr; }

bool is_symbol(const llvm::Value* value) {
    return llvm::isa<llvm::ConstantInt>(value) || llvm::isa<llvm::ConstantPointerNull>(value);
}

std::string as_literal(const llvm::ConstantData* value) {
    if (auto integer = llvm::dyn_cast<llvm::ConstantInt>(value)) {
        if (integer->getBitWidth() == 1) {
            return integer->isZero() ? "0" : "1";
        }

        llvm::SmallVector<char> val;
        if (!integer->getValue().isSignBitSet()) {
            integer->getValue().toStringUnsigned(val);
        } else {
            integer->getValue().toStringSigned(val);
        }

        return std::string(val.begin(), val.end());
    }

    if (auto fp = llvm::dyn_cast<llvm::ConstantFP>(value)) {
        if (fp->isInfinity()) {
            if (fp->isNegative()) {
                return "-INFINITY";
            } else {
                return "INFINITY";
            }
        } else if (fp->isNaN()) {
            return "NAN";
        }

        llvm::SmallVector<char> val;
        fp->getValue().toString(val);
        std::string tmp = std::string(val.begin(), val.end());
        if (tmp.find('.') == std::string::npos) {
            tmp += ".0";
        }

        if (fp->getType()->isFloatTy()) {
            tmp += "f";
        }

        return tmp;
    }

    if (auto undef = llvm::dyn_cast<llvm::UndefValue>(value)) {
        if (undef->getType()->isIntegerTy()) {
            return "0";
        } else if (undef->getType()->isFloatingPointTy()) {
            return "0.0";
        } else if (undef->getType()->isPointerTy()) {
            return sdfg::symbolic::__nullptr__()->get_name();
        }

        throw NotImplementedException(
            "UndefValue of unsupported type", docc::utils::bestEffortLoc(*value), docc::utils::toIRString(*value)
        );
    }

    if (llvm::dyn_cast<llvm::ConstantPointerNull>(value)) {
        return sdfg::symbolic::__nullptr__()->get_name();
    }

    throw NotImplementedException(
        "Unsupported constant data type", docc::utils::bestEffortLoc(*value), docc::utils::toIRString(*value)
    );
}

sdfg::symbolic::Expression as_symbol(const llvm::Constant* value) {
    if (auto const_int = llvm::dyn_cast<llvm::ConstantInt>(value)) {
        if (const_int->getBitWidth() == 1) {
            if (const_int->isZero()) {
                return sdfg::symbolic::__false__();
            } else {
                return sdfg::symbolic::__true__();
            }
        }

        if (!const_int->getValue().isSignBitSet()) {
            return sdfg::symbolic::integer(const_int->getZExtValue());
        } else {
            return sdfg::symbolic::integer(const_int->getSExtValue());
        }
    }

    if (llvm::dyn_cast<llvm::ConstantPointerNull>(value)) {
        return sdfg::symbolic::__nullptr__();
    }

    throw std::runtime_error("Not a symbolic expression");
}

std::string get_initializer(const llvm::GlobalVariable& global) {
    if (!has_initializer(global)) {
        return "";
    }

    return as_initializer(global.getInitializer());
}

bool has_initializer(const llvm::GlobalVariable& global) {
    if (!global.hasInitializer()) {
        return false;
    }
    auto initializer = global.getInitializer();
    if (!initializer) {
        return false;
    }
    if (initializer->isNullValue()) {
        return false;
    }
    if (llvm::dyn_cast<llvm::UndefValue>(initializer)) {
        return false;
    }

    return true;
}

std::string as_initializer(const llvm::Constant* initializer) {
    /*───────────────────
     *  Scalars
     *───────────────────*/
    if (auto* const_int = llvm::dyn_cast<llvm::ConstantInt>(initializer)) {
        return as_literal(const_int);
    } else if (auto* const_fp = llvm::dyn_cast<llvm::ConstantFP>(initializer)) {
        return as_literal(const_fp);
    } else if (auto* const_null = llvm::dyn_cast<llvm::ConstantPointerNull>(initializer)) {
        return as_literal(const_null);
    } else if (auto* const_undef = llvm::dyn_cast<llvm::UndefValue>(initializer)) {
        return as_literal(const_undef);
    }

    /*───────────────────
     *  Explicit aggregates (already contain their elements)
     *───────────────────*/
    else if (auto* const_array = llvm::dyn_cast<llvm::ConstantArray>(initializer)) {
        std::string result = "{";
        for (size_t i = 0; i < const_array->getNumOperands(); ++i) {
            result += as_initializer(const_array->getOperand(i));
            if (i + 1 < const_array->getNumOperands()) result += ", ";
        }
        return result + "}";
    } else if (auto* const_struct = llvm::dyn_cast<llvm::ConstantStruct>(initializer)) {
        std::string result = "{";
        for (size_t i = 0; i < const_struct->getNumOperands(); ++i) {
            result += as_initializer(const_struct->getOperand(i));
            if (i + 1 < const_struct->getNumOperands()) result += ", ";
        }
        return result + "}";
    } else if (auto* const_vector = llvm::dyn_cast<llvm::ConstantVector>(initializer)) {
        std::string result = "{";
        for (size_t i = 0; i < const_vector->getNumOperands(); ++i) { // FIX ①
            result += as_initializer(const_vector->getOperand(i));
            if (i + 1 < const_vector->getNumOperands()) result += ", "; // FIX ②
        }
        return result + "}";
    }

    /*───────────────────
     *  ConstantData* – raw bitcasts of scalars that LLVM keeps
     *───────────────────*/
    else if (auto* const_data_array = llvm::dyn_cast<llvm::ConstantDataArray>(initializer)) {
        std::string result = "{";
        llvm::Type* elemTy = const_data_array->getElementType();

        if (elemTy->isIntegerTy()) {
            for (unsigned i = 0; i < const_data_array->getNumElements(); ++i) {
                auto element = const_data_array->getElementAsConstant(i);
                result += as_literal(llvm::dyn_cast<llvm::ConstantData>(element));
                if (i + 1 < const_data_array->getNumElements()) result += ", ";
            }
        } else if (elemTy->isFloatingPointTy()) {
            for (unsigned i = 0; i < const_data_array->getNumElements(); ++i) {
                auto element = const_data_array->getElementAsConstant(i);
                result += as_literal(llvm::dyn_cast<llvm::ConstantData>(element));
                if (i + 1 < const_data_array->getNumElements()) result += ", ";
            }
        } else {
            throw NotImplementedException(
                "Unsupported ConstantDataArray element type",
                docc::utils::bestEffortLoc(*initializer),
                docc::utils::toIRString(*initializer)
            );
        }
        return result + "}";
    } else if (auto* const_data_vector = llvm::dyn_cast<llvm::ConstantDataVector>(initializer)) {
        std::string result = "{";
        llvm::Type* elemTy = const_data_vector->getElementType();

        if (elemTy->isIntegerTy()) {
            for (unsigned i = 0; i < const_data_vector->getNumElements(); ++i) {
                auto element = const_data_vector->getElementAsConstant(i);
                result += as_literal(llvm::dyn_cast<llvm::ConstantData>(element));
                if (i + 1 < const_data_vector->getNumElements()) result += ", ";
            }
        } else if (elemTy->isFloatingPointTy()) {
            for (unsigned i = 0; i < const_data_vector->getNumElements(); ++i) {
                auto element = const_data_vector->getElementAsConstant(i);
                result += as_literal(llvm::dyn_cast<llvm::ConstantData>(element));
                if (i + 1 < const_data_vector->getNumElements()) result += ", ";
            }
        } else {
            throw NotImplementedException(
                "Unsupported ConstantDataVector element type",
                docc::utils::bestEffortLoc(*initializer),
                docc::utils::toIRString(*initializer)
            );
        }
        return result + "}";
    }

    /*───────────────────
     *  zeroinitializer  → ConstantAggregateZero
     *───────────────────*/
    else if (auto* const_zero = llvm::dyn_cast<llvm::ConstantAggregateZero>(initializer)) {
        llvm::Type* ty = const_zero->getType();

        // Helper lambda: fabricate the same “zero” as an explicit Constant
        auto null_of = [&](llvm::Type* subTy) -> llvm::Constant* { return llvm::Constant::getNullValue(subTy); };

        if (auto* arrTy = llvm::dyn_cast<llvm::ArrayType>(ty)) {
            std::string result = "{";
            for (uint64_t i = 0; i < arrTy->getNumElements(); ++i) {
                result += as_initializer(null_of(arrTy->getElementType()));
                if (i + 1 < arrTy->getNumElements()) result += ", ";
            }
            return result + "}";
        } else if (auto* structTy = llvm::dyn_cast<llvm::StructType>(ty)) {
            std::string result = "{";
            for (unsigned i = 0; i < structTy->getNumElements(); ++i) {
                result += as_initializer(null_of(structTy->getElementType(i)));
                if (i + 1 < structTy->getNumElements()) result += ", ";
            }
            return result + "}";
        } else if (auto* vecTy = llvm::dyn_cast<llvm::VectorType>(ty)) {
            std::string result = "{";
            for (unsigned i = 0; i < vecTy->getElementCount().getFixedValue(); ++i) {
                result += as_initializer(null_of(vecTy->getElementType()));
                if (i + 1 < vecTy->getElementCount().getFixedValue()) result += ", ";
            }
            return result + "}";
        }
        // Should never get here – AggregateZero is only array/struct/vector.
    }

    /*───────────────────
     *  Other Global Variables (e.g., bitcasted, nested)
     *───────────────────*/
    else if (auto global_var = llvm::dyn_cast<llvm::GlobalVariable>(initializer)) {
        return utils::get_name(global_var);
    }


    /*───────────────────
     *  Fallback
     *───────────────────*/
    throw NotImplementedException(
        "Unsupported initializer", docc::utils::bestEffortLoc(*initializer), docc::utils::toIRString(*initializer)
    );
}

std::string create_const_name_to_sdfg_name(
    sdfg::builder::SDFGBuilder& builder,
    std::unordered_map<const llvm::Value*, std::string>& constants_mapping,
    const llvm::Value* llvm_ptr
) {
    if (constants_mapping.find(llvm_ptr) == constants_mapping.end()) {
        constants_mapping[llvm_ptr] = builder.find_new_name();
    }

    return constants_mapping[llvm_ptr];
}

std::string find_const_name_to_sdfg_name(
    std::unordered_map<const llvm::Value*, std::string>& constants_mapping, const llvm::Value* llvm_ptr
) {
    if (constants_mapping.find(llvm_ptr) != constants_mapping.end()) {
        return constants_mapping[llvm_ptr];
    }

    return get_name(llvm_ptr);
}

/**** Deprecated API ****/

std::unique_ptr<sdfg::types::IType> get_type(
    sdfg::builder::SDFGBuilder& builder,
    std::unordered_map<const llvm::Type*, std::string>& anonymous_types_mapping,
    const llvm::DataLayout& DL,
    llvm::Type* type,
    sdfg::types::StorageType storage_type,
    const std::string& initializer
) {
    size_t alignment = 0;

    if (type->isVoidTy()) {
        return std::make_unique<
            sdfg::types::Scalar>(storage_type, alignment, initializer, sdfg::types::PrimitiveType::Void);
    } else if (type->isIntegerTy()) {
        llvm::Align abi = DL.getABITypeAlign(type);
        alignment = abi.value();

        auto integer_type = llvm::dyn_cast<llvm::IntegerType>(type);
        switch (integer_type->getBitWidth()) {
            case 1:
                return std::make_unique<
                    sdfg::types::Scalar>(storage_type, alignment, initializer, sdfg::types::PrimitiveType::Bool);
            case 8:
                return std::make_unique<
                    sdfg::types::Scalar>(storage_type, alignment, initializer, sdfg::types::PrimitiveType::Int8);
            case 16:
                return std::make_unique<
                    sdfg::types::Scalar>(storage_type, alignment, initializer, sdfg::types::PrimitiveType::Int16);
            case 32:
                return std::make_unique<
                    sdfg::types::Scalar>(storage_type, alignment, initializer, sdfg::types::PrimitiveType::Int32);
            case 64:
                return std::make_unique<
                    sdfg::types::Scalar>(storage_type, alignment, initializer, sdfg::types::PrimitiveType::Int64);
            case 128:
                return std::make_unique<
                    sdfg::types::Scalar>(storage_type, alignment, initializer, sdfg::types::PrimitiveType::Int128);
            default:
                throw NotImplementedException(
                    "Unsupported integer type", sdfg::DebugInfo(), docc::utils::toIRString(*type)
                );
        }
    } else if (type->isFloatingPointTy()) {
        llvm::Align abi = DL.getABITypeAlign(type);
        alignment = abi.value();

        switch (type->getTypeID()) {
            case llvm::Type::HalfTyID:
                return std::make_unique<
                    sdfg::types::Scalar>(storage_type, alignment, initializer, sdfg::types::PrimitiveType::Half);
            case llvm::Type::BFloatTyID:
                return std::make_unique<
                    sdfg::types::Scalar>(storage_type, alignment, initializer, sdfg::types::PrimitiveType::BFloat);
            case llvm::Type::FloatTyID:
                return std::make_unique<
                    sdfg::types::Scalar>(storage_type, alignment, initializer, sdfg::types::PrimitiveType::Float);
            case llvm::Type::DoubleTyID:
                return std::make_unique<
                    sdfg::types::Scalar>(storage_type, alignment, initializer, sdfg::types::PrimitiveType::Double);
            case llvm::Type::X86_FP80TyID:
                return std::make_unique<
                    sdfg::types::Scalar>(storage_type, alignment, initializer, sdfg::types::PrimitiveType::X86_FP80);
            case llvm::Type::FP128TyID:
                return std::make_unique<
                    sdfg::types::Scalar>(storage_type, alignment, initializer, sdfg::types::PrimitiveType::FP128);
            case llvm::Type::PPC_FP128TyID:
                return std::make_unique<
                    sdfg::types::Scalar>(storage_type, alignment, initializer, sdfg::types::PrimitiveType::PPC_FP128);
            default:
                throw NotImplementedException(
                    "Unsupported floating point type", sdfg::DebugInfo(), docc::utils::toIRString(*type)
                );
        }
    } else if (type->isArrayTy()) {
        llvm::Align abi = DL.getABITypeAlign(type);
        alignment = abi.value();

        return std::make_unique<sdfg::types::Array>(
            storage_type,
            alignment,
            initializer,
            *get_type(builder, anonymous_types_mapping, DL, type->getArrayElementType(), storage_type),
            sdfg::symbolic::integer(type->getArrayNumElements())
        );
    } else if (auto vec_type = llvm::dyn_cast<llvm::FixedVectorType>(type)) {
        llvm::Align abi = DL.getABITypeAlign(type);
        alignment = abi.value();

        auto num_elements = vec_type->getNumElements();
        auto element_type = vec_type->getElementType();
        auto element_sdfg_type = get_type(builder, anonymous_types_mapping, DL, element_type, storage_type);
        assert(element_sdfg_type->type_id() == sdfg::types::TypeID::Scalar && "Vector element type must be a scalar");

        return builder.create_vector_type(static_cast<const sdfg::types::Scalar&>(*element_sdfg_type), num_elements);
    } else if (auto typed_ptr_type = llvm::dyn_cast<llvm::TypedPointerType>(type)) {
        llvm::Align abi = DL.getABITypeAlign(type);
        alignment = abi.value();

        auto pointee_type = typed_ptr_type->getElementType();
        return std::make_unique<sdfg::types::Pointer>(
            storage_type,
            alignment,
            initializer,
            *get_type(builder, anonymous_types_mapping, DL, pointee_type, storage_type)
        );
    } else if (type->isPointerTy()) {
        llvm::Align abi = DL.getABITypeAlign(type);
        alignment = abi.value();

        return std::make_unique<sdfg::types::Pointer>(storage_type, alignment, initializer);
    } else if (type->isStructTy()) {
        llvm::Align abi = DL.getABITypeAlign(type);
        alignment = abi.value();

        auto structure_type = llvm::dyn_cast<llvm::StructType>(type);
        auto structure_name = normalize_name(structure_type->getName().str());
        if (structure_name.empty()) {
            if (anonymous_types_mapping.find(type) != anonymous_types_mapping.end()) {
                structure_name = anonymous_types_mapping[type];
            } else {
                structure_name = "__daisy_anonymous_struct" + std::to_string(anonymous_types_mapping.size());
                anonymous_types_mapping[type] = structure_name;
            }
        }

        // Create the actual type.
        auto sdfg_structure_type =
            std::make_unique<sdfg::types::Structure>(storage_type, alignment, initializer, structure_name);

        // Define structure if necessary
        auto defined_structures = builder.subject().structures();
        if (std::find(defined_structures.begin(), defined_structures.end(), structure_name) ==
            defined_structures.end()) {
            auto& structure_definition = builder.add_structure(structure_name, structure_type->isPacked());
            for (auto elem : structure_type->elements()) {
                structure_definition.add_member(*get_type(builder, anonymous_types_mapping, DL, elem, storage_type));
            }
        }

        return sdfg_structure_type;
    } else if (type->isFunctionTy()) {
        auto function_type = llvm::dyn_cast<llvm::FunctionType>(type);
        auto return_type = get_type(builder, anonymous_types_mapping, DL, function_type->getReturnType(), storage_type);
        auto sdfg_function_type = std::make_unique<
            sdfg::types::Function>(storage_type, alignment, initializer, *return_type, function_type->isVarArg());
        for (auto param : function_type->params()) {
            auto param_type = get_type(builder, anonymous_types_mapping, DL, param, storage_type);
            sdfg_function_type->add_param(*param_type);
        }
        return sdfg_function_type;
    }

    throw NotImplementedException("Unsupported type", sdfg::DebugInfo(), docc::utils::toIRString(*type));
}

sdfg::types::StorageType get_storage_type(sdfg::FunctionType target_type, size_t address_space) {
    if (target_type == sdfg::FunctionType_CPU) {
        switch (address_space) {
            case 0:
                return sdfg::types::StorageType::CPU_Stack();
            default:
                throw NotImplementedException(
                    "Unsupported CPU address space: " + std::to_string(address_space), sdfg::DebugInfo(), ""
                );
        }
    } else if (target_type == sdfg::FunctionType_NV_GLOBAL) {
        switch (address_space) {
            case 0:
                return sdfg::types::StorageType::NV_Generic();
            case 1:
                return sdfg::types::StorageType::NV_Global();
            case 3:
                return sdfg::types::StorageType::NV_Shared();
            case 4:
                return sdfg::types::StorageType::NV_Constant();
            default:
                throw NotImplementedException(
                    "Unsupported GPU address space: " + std::to_string(address_space), sdfg::DebugInfo(), ""
                );
        }
    }

    throw NotImplementedException(
        "Unsupported function type: " + target_type.value(), sdfg::DebugInfo(), target_type.value()
    );
}


std::pair<std::string, sdfg::data_flow::Subset>
get_memlet(std::unordered_map<const llvm::Value*, std::string>& constants_mapping, const llvm::GEPOperator* gep) {
    auto ptr = gep->getPointerOperand();

    std::string data = find_const_name_to_sdfg_name(constants_mapping, ptr);
    if (!llvm::dyn_cast<llvm::GlobalVariable>(ptr) && llvm::dyn_cast<llvm::GlobalValue>(ptr)) {
        throw NotImplementedException(
            "Unsupported global value: " + ptr->getName().str(),
            docc::utils::bestEffortLoc(*ptr),
            docc::utils::toIRString(*ptr)
        );
    }

    if (utils::is_null_pointer(ptr)) {
        data = sdfg::symbolic::__nullptr__()->get_name();
    }

    sdfg::data_flow::Subset subset;
    for (auto it = gep->idx_begin(); it != gep->idx_end(); ++it) {
        if (auto const_int = llvm::dyn_cast<llvm::ConstantInt>(*it)) {
            auto start = sdfg::symbolic::integer(const_int->getZExtValue());
            subset.push_back(start);
        } else {
            auto start = sdfg::symbolic::symbol(utils::get_name(*it));
            subset.push_back(start);
        }
    }

    return {data, subset};
}

std::vector<llvm::GlobalVariable*> add_nested_global(const llvm::ConstantExpr* expr) {
    std::vector<llvm::GlobalVariable*> result;
    for (auto& op : expr->operands()) {
        if (auto global = llvm::dyn_cast<llvm::GlobalVariable>(op)) {
            result.push_back(global);
        } else if (auto const_expr = llvm::dyn_cast<llvm::ConstantExpr>(op)) {
            auto nested_result = add_nested_global(const_expr);
            result.insert(result.end(), nested_result.begin(), nested_result.end());
        } else {
            throw NotImplementedException(
                "Unsupported operand", docc::utils::bestEffortLoc(*expr), docc::utils::toIRString(*expr)
            );
        }
    }
    return result;
}

} // namespace utils
} // namespace lifting
} // namespace docc
