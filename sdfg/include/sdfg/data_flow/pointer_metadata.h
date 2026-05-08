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

    /**
     * Despite this being a leak of the pointer,
     * the user will only use it for blocking accesses to the underlying data and not keep a reference to the data in
     * any way. Like a Rust temporary borrow for the duration of the LibNode and no more.
     */
    virtual bool no_ptr_escape() const = 0;

    /**
     * The pointer may be used to read from the backing data
     */
    virtual bool may_contain_reads() const = 0;

    /**
     * The pointe may be used to write to the backing data
     */
    virtual bool may_contain_writes() const = 0;

    /**
     * Describes which elements are accessed (for example a function may only access the range of [ptr, ptr+8] bytes and
     * not touch or care about what comes after) Pointer access metadata only applies to the elements that are part of
     * the pattern.
     */
    virtual std::optional<MemoryAccessPattern> access_pattern() const { return std::nullopt; }
};

/**
 * The pointer is only used for reading. Like const* in Cpp.
 * Data pointed to will not change due to this.
 */
class PointerReadOnly : public PointerAccessMeta {
private:
    symbolic::Expression size_; // simplified until we have more than convex pattern
    bool no_ptr_escape_;

public:
    PointerReadOnly(symbolic::Expression size, bool no_ptr_escape = false)
        : size_(size), no_ptr_escape_(no_ptr_escape) {}

    /**
     * Despite this being a leak of the pointer,
     * the user will only use it for blocking accesses to the underlying data and not keep a reference to the data in
     * any way. Like a Rust temporary borrow for the duration of the LibNode and no more.
     */
    bool no_ptr_escape() const override { return no_ptr_escape_; }

    bool may_contain_reads() const override { return true; }
    bool may_contain_writes() const override { return false; }

    /**
     * Describes which elements behind the pointer are actually read
     */
    std::optional<MemoryAccessPattern> access_pattern() const override {
        if (size_.is_null()) {
            return std::nullopt;
        } else {
            return ConvexAccessPattern(size_);
        }
    }
};

/**
 * The pointer is used to overwrite all the data. No data within the pattern survives
 * The result could potentially be written to a new memory area
 * This must not be used if the write-pattern is not exact. This GUARANTEES that every part of the pattern is written.
 * I.e. if no pattern matches exactly, you must use PointerUnknownAccess
 */
class PointerFullWrite : public PointerAccessMeta {
private:
    symbolic::Expression size_; // simplified until we have more than convex pattern
    bool no_ptr_escape_;
    bool write_only_;

public:
    PointerFullWrite(symbolic::Expression size, bool no_ptr_escape = false, bool write_only = false)
        : size_(size), no_ptr_escape_(no_ptr_escape), write_only_(write_only) {}

    /**
     * Describes which elements are overwritten. If the underlying memory-area is larger,
     * other elements outside of the pattern remain unchanged
     */
    std::optional<MemoryAccessPattern> access_pattern() const override {
        if (size_.is_null()) {
            return std::nullopt;
        } else {
            return ConvexAccessPattern(size_);
        }
    }

    bool no_ptr_escape() const override { return no_ptr_escape_; }

    bool is_write_only() const { return write_only_; }

    bool may_contain_reads() const override { return !write_only_; }
    bool may_contain_writes() const override { return true; }
};

/**
 * It is unknown what is done with this pointer or it is a mix of reads and writes.
 * This could overwrite some parts of the area pointed to, but leave others as is.
 * Assume the worst: data is made dirty by a black box. You know nothing about the contents after this
 */
class PointerUnknownAccess : public PointerAccessMeta {
private:
    symbolic::Expression size_; // simplified until we have more than convex pattern

public:
    PointerUnknownAccess(symbolic::Expression size = SymEngine::null) : size_(size) {}

    /**
     * Allows limiting the undefined behavior (assume required input as well as dirtying) to a specific pattern
     * No pattern means could be all the memory-area pointed to
     */
    std::optional<MemoryAccessPattern> access_pattern() const override {
        if (size_.is_null()) {
            return std::nullopt;
        } else {
            return ConvexAccessPattern(size_);
        }
    }

    bool no_ptr_escape() const override { return false; }

    bool may_contain_reads() const override { return true; }
    bool may_contain_writes() const override { return true; }
};

/**
 * Meaning the underlying memory will be deallocated and use of the pointer after this is no longer valid.
 * Does not represent a leak of the pointer.
 * Read-accesses to the pointer itself after this, but before an overwrite represent accessing most-likely invalid data
 * Memory accesses using this invalid pointer are catastrophic failures.
 */
class PointerInvalidate : public PointerAccessMeta {
    bool no_ptr_escape() const override { return true; }

    bool may_contain_reads() const override { return false; }
    bool may_contain_writes() const override { return false; }
};

typedef std::unique_ptr<PointerAccessMeta> PointerAccessType;


} // namespace sdfg::data_flow
