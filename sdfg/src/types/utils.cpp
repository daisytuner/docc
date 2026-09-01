#include "sdfg/types/utils.h"
#include <cmath>
#include <memory>
#include <string>

#include "sdfg/codegen/utils.h"
#include "sdfg/function.h"
#include "sdfg/symbolic/symbolic.h"

#include "sdfg/types/structure.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace types {

std::unique_ptr<types::IType>
infer_type_internal(const sdfg::Function& function, const types::IType& type, const data_flow::Subset& subset) {
    if (subset.empty()) {
        return type.clone();
    }

    if (type.type_id() == TypeID::Scalar) {
        if (!subset.empty()) {
            throw InvalidSDFGException("Scalar type must have no subset");
        }

        return type.clone();
    } else if (type.type_id() == TypeID::Array) {
        auto& array_type = static_cast<const types::Array&>(type);

        data_flow::Subset element_subset(subset.begin() + 1, subset.end());
        return infer_type_internal(function, array_type.element_type(), element_subset);
    } else if (type.type_id() == TypeID::Structure) {
        auto& structure_type = static_cast<const types::Structure&>(type);

        data_flow::Subset element_subset(subset.begin() + 1, subset.end());

        auto& definition = function.structure(structure_type.name());
        if (definition.is_vector()) {
            return infer_type_internal(function, definition.vector_element_type(), element_subset);
        }
        auto member = SymEngine::rcp_dynamic_cast<const SymEngine::Integer>(subset.at(0));
        return infer_type_internal(function, definition.member_type(member), element_subset);
    } else if (type.type_id() == TypeID::Tensor) {
        auto& tensor_type = static_cast<const types::Tensor&>(type);

        data_flow::Subset element_subset(subset.begin() + 1, subset.end());

        if (tensor_type.shape().size() == 1) {
            return infer_type_internal(function, tensor_type.element_type(), element_subset);
        } else {
            auto inner_tensor = std::make_unique<types::Tensor>(
                tensor_type.storage_type(),
                tensor_type.alignment(),
                tensor_type.initializer(),
                tensor_type.element_type(),
                symbolic::MultiExpression(tensor_type.shape().begin() + 1, tensor_type.shape().end()),
                symbolic::MultiExpression(tensor_type.strides().begin() + 1, tensor_type.strides().end()),
                tensor_type.offset()
            );
            return infer_type_internal(function, *inner_tensor, element_subset);
        }
    } else if (type.type_id() == TypeID::Pointer) {
        throw InvalidSDFGException("Subset references non-contiguous memory");
    }

    throw InvalidSDFGException("Type inference failed because of unknown type");
};

std::unique_ptr<types::IType>
infer_type(const sdfg::Function& function, const types::IType& type, const data_flow::Subset& subset) {
    if (subset.empty()) {
        return type.clone();
    }

    if (type.type_id() == TypeID::Pointer) {
        auto& pointer_type = static_cast<const types::Pointer&>(type);
        if (!pointer_type.has_pointee_type()) {
            throw InvalidSDFGException("Opaque pointer with non-empty subset");
        }

        auto& pointee_type = pointer_type.pointee_type();
        data_flow::Subset element_subset(subset.begin() + 1, subset.end());
        return infer_type_internal(function, pointee_type, element_subset);
    } else {
        return infer_type_internal(function, type, subset);
    }
};

std::unique_ptr<types::IType> recombine_array_type(const types::IType& type, uint depth, const types::IType& inner_type) {
    if (depth == 0) {
        return inner_type.clone();
    } else {
        if (auto atype = dynamic_cast<const types::Array*>(&type)) {
            return std::make_unique<types::Array>(
                atype->storage_type(),
                atype->alignment(),
                atype->initializer(),
                *recombine_array_type(atype->element_type(), depth - 1, inner_type).get(),
                atype->num_elements()
            );
        } else {
            throw std::runtime_error("construct_type: Non array types are not supported yet!");
        }
    }
};

const IType& peel_to_innermost_element(const IType& type, int follow_ptr) {
    int next_follow = follow_ptr;
    if (follow_ptr == PEEL_TO_INNERMOST_ELEMENT_FOLLOW_ONLY_OUTER_PTR) {
        next_follow = 0; // only follow an outermost pointer
    }

    switch (type.type_id()) {
        case TypeID::Array:
            return peel_to_innermost_element(dynamic_cast<const types::Array&>(type).element_type(), next_follow);
        case TypeID::Reference:
            return peel_to_innermost_element(dynamic_cast<const codegen::Reference&>(type).reference_type(), next_follow);
        case TypeID::Tensor:
            return peel_to_innermost_element(dynamic_cast<const types::Tensor&>(type).element_type(), next_follow);
        case TypeID::Pointer:
            if (follow_ptr != 0) {
                if (follow_ptr != PEEL_TO_INNERMOST_ELEMENT_FOLLOW_ONLY_OUTER_PTR) {
                    next_follow = follow_ptr - 1; // follow one less pointer
                }

                auto& pointer_type = dynamic_cast<const types::Pointer&>(type);
                if (pointer_type.has_pointee_type()) {
                    return peel_to_innermost_element(pointer_type.pointee_type(), next_follow);
                } else {
                    return type;
                }
            }
            // fall back to cut-off if we did not follow the pointer
        default:
            return type;
    }
}

symbolic::Expression get_contiguous_element_size(const types::IType& type, bool allow_comp_time_eval) {
    // need to peel explicitly, primitive_type() would follow ALL pointers, even ***, even though this is not contiguous
    auto& innermost = peel_to_innermost_element(type, PEEL_TO_INNERMOST_ELEMENT_FOLLOW_ONLY_OUTER_PTR);
    return get_type_size(innermost, allow_comp_time_eval);
}

symbolic::Expression get_type_size(const types::IType& type, bool allow_comp_time_eval) {
    bool only_symbolic = false;

    auto id = type.type_id();
    if (id == TypeID::Pointer || id == TypeID::Reference || id == TypeID::Function) {
        return symbolic::integer(8); // assume 64-bit pointers
    } else if (id == TypeID::Structure) {
        // TODO if we have the target definition, we could evaluate the StructureDefinition to a size
        only_symbolic = true;
    } else if (id == TypeID::Array) {
        auto& arr = dynamic_cast<const types::Array&>(type);
        auto inner_element_size = get_type_size(arr.element_type(), allow_comp_time_eval);
        if (!inner_element_size.is_null()) {
            return symbolic::mul(inner_element_size, arr.num_elements());
        } else {
            return {};
        }
    } else if (id == TypeID::Tensor) {
        auto& tensor = dynamic_cast<const types::Tensor&>(type);
        auto inner_element_size = get_type_size(tensor.element_type(), allow_comp_time_eval);
        if (!inner_element_size.is_null()) {
            symbolic::Expression num_elements = symbolic::one();
            for (const auto& dim : tensor.shape()) {
                num_elements = symbolic::mul(num_elements, dim);
            }
            return symbolic::mul(inner_element_size, num_elements);
        } else {
            return {};
        }
    }

    if (only_symbolic) {
        // Could not statically figure out the size
        // Could be struct we could evaluate by its definition or sth. we do not understand here
        if (allow_comp_time_eval) {
            return symbolic::size_of_type(type);
        } else { // size unknown
            return {};
        }
    } else { // should just be a primitive type
        auto prim_type = type.primitive_type();
        double size = std::ceil(static_cast<double>(types::bit_width(prim_type)) / 8.0);
        long size_of_type = static_cast<long>(size);
        if (size_of_type != 0) {
            return symbolic::integer(size_of_type);
        } else {
            return {};
        }
    }
}

const types::IType* peel_to_next_element(const types::IType& type) {
    switch (type.type_id()) {
        case TypeID::Array:
            return &dynamic_cast<const types::Array&>(type).element_type();
        case TypeID::Reference:
            return &dynamic_cast<const codegen::Reference&>(type).reference_type();
        case TypeID::Tensor:
            return &dynamic_cast<const types::Tensor&>(type).element_type();
        case TypeID::Pointer: {
            auto& pointer_type = dynamic_cast<const types::Pointer&>(type);
            if (pointer_type.has_pointee_type()) {
                return &pointer_type.pointee_type();
            } else {
                return nullptr;
            }
        }
        default:
            return &type;
    }
}

bool is_contiguous_type(const types::IType& base_type, StructuredSDFG& sdfg) {
    auto& type = types::peel_to_innermost_element(base_type);
    if (type.type_id() == types::TypeID::Pointer) {
        return false;
    }

    // Check for distant nests
    if (type != types::peel_to_innermost_element(base_type, -1)) {
        return false;
    }

    // Check for nested structures
    if (type.type_id() == types::TypeID::Structure) {
        std::list<types::Structure> structures;
        std::unordered_set<std::string> visited_structures;
        structures.push_back(dynamic_cast<const types::Structure&>(type));
        while (structures.size() > 0) {
            auto structure = structures.front();
            structures.pop_front();
            if (visited_structures.contains(structure.name())) {
                return false; // infinitely nested structures are not supported
            }

            visited_structures.insert(structure.name());
            auto& definition = sdfg.structure(structure.name());
            for (size_t i = 0; i < definition.num_members(); i++) {
                auto& member_type = definition.member_type(symbolic::integer(i));
                if (member_type.type_id() == types::TypeID::Structure) {
                    structures.push_back(dynamic_cast<const types::Structure&>(member_type));
                } else if (member_type.type_id() == types::TypeID::Pointer) {
                    return false; // pointers in structures are not supported
                }
            }
        }
    }
    return true;
}

PrimitiveType get_primitive_type_to_hold_upper_bound(const symbolic::Expression& expr) {
    static auto int32_bound = symbolic::integer(INT32_MAX);

    if (symbolic::is_true(symbolic::Le(expr, int32_bound))) {
        return Int32;
    } else {
        return Int64;
    }
}

PrimitiveType get_primitive_type_to_hold_lower_bound(const symbolic::Expression& expr) {
    static auto int32_bound = symbolic::integer(INT32_MIN);

    if (symbolic::is_true(symbolic::Ge(expr, int32_bound))) {
        return Int32;
    } else {
        return Int64;
    }
}

PrimitiveType get_primitive_type_to_hold(PrimitiveType a, PrimitiveType b) {
    // Describe each supported integer type by its bit width and signedness.
    auto bit_width = [](types::PrimitiveType t) -> int {
        switch (t) {
            case types::PrimitiveType::Int32:
            case types::PrimitiveType::UInt32:
                return 32;
            case types::PrimitiveType::Int64:
            case types::PrimitiveType::UInt64:
                return 64;
            default:
                throw std::invalid_argument("Cannot handle " + std::string(types::primitive_type_to_string(t)));
        }
    };
    auto is_signed = [](types::PrimitiveType t) -> bool {
        switch (t) {
            case types::PrimitiveType::Int32:
            case types::PrimitiveType::Int64:
                return true;
            case types::PrimitiveType::UInt32:
            case types::PrimitiveType::UInt64:
                return false;
            default:
                throw std::invalid_argument("Cannot handle " + std::string(types::primitive_type_to_string(t)));
        }
    };

    // Can every value of `src` be extended into `dst` without any loss?
    auto fits_in = [&](types::PrimitiveType src, types::PrimitiveType dst) -> bool {
        const int src_bits = bit_width(src);
        const int dst_bits = bit_width(dst);
        const bool src_signed = is_signed(src);
        const bool dst_signed = is_signed(dst);

        if (src_signed == dst_signed) {
            // Same signedness: the destination just needs to be at least as wide.
            return dst_bits >= src_bits;
        }
        // not safe in general, as we may loose overflows
        return false;
    };

    // Candidates ordered from smallest to largest capacity so the first match is
    // the tightest common type that both operands extend into without loss.
    static constexpr types::PrimitiveType candidates[] = {
        types::PrimitiveType::Int32,
        types::PrimitiveType::UInt32,
        types::PrimitiveType::Int64,
        types::PrimitiveType::UInt64,
    };
    for (auto candidate : candidates) {
        if (fits_in(a, candidate) && fits_in(b, candidate)) {
            return candidate;
        }
    }

    throw std::invalid_argument(
        "Cannot combine types " + std::string(types::primitive_type_to_string(a)) + " and " +
        types::primitive_type_to_string(b) + " without loss"
    );
}

} // namespace types
} // namespace sdfg
