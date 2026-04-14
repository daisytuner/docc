#pragma once

#include <optional>
#include "sdfg/symbolic/symbolic.h"

namespace sdfg::data_flow {

class MemoryAccessPattern {
public:
    virtual ~MemoryAccessPattern() = default;
};

class ConvexAccessPattern : public MemoryAccessPattern {
private:
    symbolic::Expression size_;

public:
    ConvexAccessPattern(symbolic::Expression size) : size_(size) {}

    symbolic::Expression size() const { return size_; }
};

class PointerAccessMeta {
protected:
    PointerAccessMeta() = default;

public:
    virtual ~PointerAccessMeta() = default;
};

/**
 * The pointer is only used for reading. Like const* in Cpp.
 * Data pointed to will not change due to this.
 */
class PointerReadOnly : public PointerAccessMeta {
private:
    symbolic::Expression size_;
    bool no_ptr_escape_;

public:
    PointerReadOnly(symbolic::Expression size, bool no_ptr_escape = false)
        : size_(size), no_ptr_escape_(no_ptr_escape) {}

    /**
     * Despite this being a leak of the pointer,
     * the user will only use it for blocking accesses to the underlying data and not keep a reference to the data in
     * any way Like a Rust temporary borrow for the duration for the LibNode and no more.
     */
    bool no_ptr_escape() const { return no_ptr_escape_; }

    /**
     * Describes which elements are accessed (for example a function may only access the range of [ptr, ptr+8] bytes and
     * touch or care about what comes after) Pointer access metadata only applies to the elements that are part of the
     * pattern.
     */
    std::optional<MemoryAccessPattern> access_pattern() const {
        if (size_.is_null()) {
            return std::nullopt;
        } else {
            return ConvexAccessPattern(size_);
        }
    }
};

/**
 * The pointer is used to overwrite all of the data. No data within the area survives
 * The result could potentially be written to a new memory area
 */
class PointerFullWrite : public PointerAccessMeta {
    /**
     * Describes which elements are accessed (for example a function may only access the range of [ptr, ptr+8] bytes and
     * touch or care about what comes after) Pointer access metadata only applies to the elements that are part of the
     * pattern.
     */
    std::optional<MemoryAccessPattern> access_pattern() const { return std::nullopt; }
};

/**
 * It is unknown what is done with this pointer or it is a mix of reads and writes.
 * This could overwrite some parts of the area pointed to, but leave others as is.
 * Assume the worst: data is made dirty by a black box. You know nothing about the contents after this
 */
class PointerUnknownAccess : public PointerAccessMeta {};

typedef std::variant<PointerUnknownAccess, PointerReadOnly, PointerFullWrite> PointerAccessType;

} // namespace sdfg::data_flow
