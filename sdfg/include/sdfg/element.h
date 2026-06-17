#pragma once

#include <cassert>
#include <string>

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
     * Validates the element.
     *
     * @throw InvalidSDFGException if the element is invalid
     */
    virtual void validate(const Function& function) const = 0;

    virtual void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) = 0;

    virtual void replace(const symbolic::ExpressionMapping& replacements);
};

typedef size_t ElementId;

} // namespace sdfg
