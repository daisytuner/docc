#pragma once

#include <cassert>
#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>

#include "sdfg/exceptions.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {

namespace builder {
class SDFGBuilder;
class StructuredSDFGBuilder;
} // namespace builder

namespace deepcopy {
class StructuredSDFGDeepCopy;
} // namespace deepcopy

namespace serializer {
class JSONSerializer;
} // namespace serializer

class Function;

class DebugInfo {
private:
    std::string filename_;
    std::string function_;
    size_t start_line_;
    size_t start_column_;
    size_t end_line_;
    size_t end_column_;

    bool has_;

public:
    DebugInfo();

    DebugInfo(std::string filename, size_t start_line, size_t start_column, size_t end_line, size_t end_column);

    DebugInfo(
        std::string filename,
        std::string function,
        size_t start_line,
        size_t start_column,
        size_t end_line,
        size_t end_column
    );

    bool has() const;

    std::string filename() const;

    std::string function() const;

    size_t start_line() const;

    size_t start_column() const;

    size_t end_line() const;

    size_t end_column() const;

    static DebugInfo merge(const DebugInfo& left, const DebugInfo& right);
};

enum class ElementType : uint64_t {
    State = 1LL,
    InterstateEdge = 1LL << 1,
    AccessNode = 1LL << 2,
    ConstantNode = 1LL << 3,
    Tasklet = 1LL << 4,
    LibraryNode = 1LL << 5,
    Memlet = 1LL << 6,
    Return = 1LL << 7,
    Block = 1LL << 8,
    Sequence = 1LL << 9,
    Transition = 1LL << 10,
    IfElse = 1LL << 11,
    While = 1LL << 12,
    Continue = 1LL << 13,
    Break = 1LL << 14,
    For = 1LL << 15,
    Map = 1LL << 16,
    Reduce = 1LL << 17,
};

/// Bitwise combination of element types, enabling category masks.
constexpr ElementType operator|(ElementType lhs, ElementType rhs) {
    return static_cast<ElementType>(static_cast<uint64_t>(lhs) | static_cast<uint64_t>(rhs));
}

constexpr ElementType operator&(ElementType lhs, ElementType rhs) {
    return static_cast<ElementType>(static_cast<uint64_t>(lhs) & static_cast<uint64_t>(rhs));
}

constexpr ElementType operator~(ElementType type) { return static_cast<ElementType>(~static_cast<uint64_t>(type)); }

/// Tests whether \p type is contained in the category \p mask (efficient, branch-free type check).
constexpr bool is_a(ElementType type, ElementType mask) {
    return (static_cast<uint64_t>(type) & static_cast<uint64_t>(mask)) != 0;
}

/// Returns the type of an element as a human-readable string (no allocation).
constexpr std::string_view element_type_name(ElementType type) {
    switch (type) {
        case ElementType::State:
            return "state";
        case ElementType::InterstateEdge:
            return "interstate_edge";
        case ElementType::AccessNode:
            return "access_node";
        case ElementType::ConstantNode:
            return "constant_node";
        case ElementType::Tasklet:
            return "tasklet";
        case ElementType::LibraryNode:
            return "library_node";
        case ElementType::Memlet:
            return "memlet";
        case ElementType::Return:
            return "return";
        case ElementType::Block:
            return "block";
        case ElementType::Sequence:
            return "sequence";
        case ElementType::Transition:
            return "transition";
        case ElementType::IfElse:
            return "if_else";
        case ElementType::While:
            return "while";
        case ElementType::Continue:
            return "continue";
        case ElementType::Break:
            return "break";
        case ElementType::For:
            return "for";
        case ElementType::Map:
            return "map";
        case ElementType::Reduce:
            return "reduce";
    }
    throw InvalidSDFGException("Element: Unknown element type");
}

class Element {
    friend class builder::SDFGBuilder;
    friend class builder::StructuredSDFGBuilder;
    friend class serializer::JSONSerializer;
    friend class deepcopy::StructuredSDFGDeepCopy;

protected:
    size_t element_id_;
    DebugInfo debug_info_;

public:
    Element(size_t element_id, const DebugInfo& debug_info);

    virtual ~Element() = default;

    size_t element_id() const;

    const DebugInfo& debug_info() const;

    void set_debug_info(const DebugInfo& debug_info);

    /**
     * Returns the type of the element.
     */
    virtual ElementType type_id() const = 0;

    /**
     * Returns the type of the element as a human-readable string.
     */
    std::string_view element_type() const { return element_type_name(this->type_id()); }

    /**
     * Validates the element.
     *
     * @throw InvalidSDFGException if the element is invalid
     */
    virtual void validate(const Function& function) const = 0;

    virtual void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) = 0;

    virtual void replace(const symbolic::ExpressionMapping& replacements);
};

typedef size_t ElementId;

/**
 * Resolves the concrete target type for isa/dyn_cast, accepting the type itself
 * (`Block`), a pointer to it (`Block*`), or a reference to it (`Block&`).
 */
template<typename T>
using element_cast_target_t = std::remove_pointer_t<std::remove_reference_t<T>>;

/**
 * LLVM-style RTTI check: true if \p element belongs to type category \p T.
 *
 * \p T must expose a `static bool classof(const Element&)` predicate (typically
 * implemented via a power-of-two ElementType category mask). This is a
 * branch-free alternative to dynamic_cast for element categories.
 *
 * \p T may be given as the target type (`isa<While>`), a pointer (`isa<While*>`),
 * or a reference (`isa<While&>`); all are treated identically.
 */
template<typename T>
bool isa(const Element& element) {
    return element_cast_target_t<T>::classof(element);
}

template<typename T>
bool isa(const Element* element) {
    return element != nullptr && element_cast_target_t<T>::classof(*element);
}

/// Efficient down-cast to \p T, or nullptr if \p element is not a \p T.
template<typename T>
element_cast_target_t<T>* dyn_cast(Element* element) {
    using Target = element_cast_target_t<T>;
    return (element != nullptr && Target::classof(*element)) ? static_cast<Target*>(element) : nullptr;
}

template<typename T>
const element_cast_target_t<T>* dyn_cast(const Element* element) {
    using Target = element_cast_target_t<T>;
    return (element != nullptr && Target::classof(*element)) ? static_cast<const Target*>(element) : nullptr;
}

/// Checked down-cast to \p T; throws InvalidSDFGException if \p element is not a \p T.
template<typename T>
element_cast_target_t<T>& dyn_cast(Element& element) {
    using Target = element_cast_target_t<T>;
    if (!Target::classof(element)) {
        throw InvalidSDFGException("dyn_cast: element is not of the requested type");
    }
    return static_cast<Target&>(element);
}

template<typename T>
const element_cast_target_t<T>& dyn_cast(const Element& element) {
    using Target = element_cast_target_t<T>;
    if (!Target::classof(element)) {
        throw InvalidSDFGException("dyn_cast: element is not of the requested type");
    }
    return static_cast<const Target&>(element);
}

} // namespace sdfg
