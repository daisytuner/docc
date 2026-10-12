#pragma once

#include <algorithm>
#include <cctype>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string_view>
#include <type_traits>

#include "sdfg/types/type.h"

namespace sdfg {
namespace codegen {

class PrettyPrinter {
public:
    // Constructor
    PrettyPrinter(int indent = 0, bool frozen = false);
    PrettyPrinter(std::ostream& stream, int indent = 0, bool frozen = false);

    // Set the indentation level
    void setIndent(int indent);

    int indent() const;

    int changeIndent(int delta);

    // Get the underlying string
    std::string str() const;

    // Clear the stringstream content
    void clear();

    /// Monotonic count of writes that emitted non-whitespace text; compare two readings to detect an empty region.
    size_t content() const {
        return content_;
    }

    // Overload the insertion operator
    template<typename T>
    PrettyPrinter& operator<<(const T& value) {
        if (frozen_) {
            throw std::runtime_error("PrettyPrinter is frozen");
        }
        applyIndent();
        stream << value;
        if (has_content(value)) {
            ++content_;
        }
        return *this;
    }

    // Overload for manipulators (like std::endl)
    PrettyPrinter& operator<<(std::ostream& (*manip)(std::ostream&) );

private:
    std::unique_ptr<std::stringstream> owned_stream;
    std::ostream& stream;
    int indentSize;
    bool isNewLine = true;
    bool frozen_;
    size_t content_ = 0;

    // Apply indentation only at the beginning of a new line
    void applyIndent();

    template<typename T>
    static bool has_content(const T& value) {
        auto not_space = [](char c) {
            return !std::isspace(static_cast<unsigned char>(c));
        };
        if constexpr (std::is_same_v<T, char>) {
            return not_space(value);
        } else if constexpr (std::is_convertible_v<const T&, std::string_view>) {
            std::string_view text(value);
            return std::any_of(text.begin(), text.end(), not_space);
        } else {
            // Stream manipulators such as std::hex emit nothing.
            return !std::is_function_v<T>;
        }
    }
};

class Reference : public types::IType {
private:
    std::unique_ptr<types::IType> reference_;

public:
    Reference(const types::IType& reference_);

    Reference(
        types::StorageType storage_type, size_t alignment, const std::string& initializer, const types::IType& reference_
    );

    std::unique_ptr<types::IType> clone() const override;

    virtual types::TypeID type_id() const override;

    types::PrimitiveType primitive_type() const override;

    bool is_symbol() const override;

    bool is_pointer_like() const override {
        return true;
    }

    const types::IType& reference_type() const;

    bool operator==(const types::IType& other) const override;

    std::string print() const override;

    /**
     * @brief Replace symbolic expressions on this type
     * @param old_expression Expression to replace
     * @param new_expression Replacement expression
     *
     * Replaces occurrences of symbolic expressions on the type.
     */
    virtual void
    replace_symbols(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;
    virtual void replace_symbols(const symbolic::ExpressionMapping& replacements) override;
};

} // namespace codegen
} // namespace sdfg
