#pragma once

#include <nlohmann/json.hpp>
#include <optional>

#include "sdfg/symbolic/symbolic.h"

namespace sdfg::data_flow {

class PointerAccessMeta;
class MemoryAccessPattern;

template<typename T>
struct PtrMetaDeleter {
    bool should_delete_;

    PtrMetaDeleter(bool should_delete = true) : should_delete_(should_delete) {}

    void operator()(T* ptr) const {
        if (should_delete_) {
            delete ptr;
        }
    }
};

typedef PtrMetaDeleter<PointerAccessMeta> PtrAccessDeleter;
typedef PtrMetaDeleter<MemoryAccessPattern> AccessPatternsDeleter;
typedef std::unique_ptr<PointerAccessMeta, PtrAccessDeleter> PointerAccessType;
typedef std::unique_ptr<MemoryAccessPattern, AccessPatternsDeleter> MemoryAccessPatternType;

class MemoryAccessPattern {
public:
    virtual ~MemoryAccessPattern() = default;

    /**
     * There are no elements in this pattern = no accesses of this type
     */
    virtual bool empty() const = 0;

    /**
     * Every element that is part of this pattern is actually accessed.
     * If false, this is just an approximation that may contain elements in between the bounds that are not guaranteed
     * to be accessed Ex. indicating write of a[0] and a[9] as ConvexPattern(10) is approximate. Writes a[0], a[1], ..
     * [9] would be every element being accessed. Bounds should never be violated regardless. We may in the future
     */
    virtual bool every_element_accessed() const = 0;

    MemoryAccessPatternType ref() const;

    virtual void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) = 0;

    virtual MemoryAccessPatternType clone() const = 0;

    virtual void serialize_to_json(nlohmann::json& entry) = 0;
};

class ConvexAccessPattern : public MemoryAccessPattern {
private:
    symbolic::Expression size_;
    bool not_sparse_ = false;

public:
    ConvexAccessPattern(symbolic::Expression size, bool not_sparse = false) : size_(size), not_sparse_(not_sparse) {}

    symbolic::Expression size() const { return size_; }

    bool empty() const override { return symbolic::null_safe_eq(size_, symbolic::zero()); }

    bool every_element_accessed() const override { return not_sparse_; }

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override {
        size_ = symbolic::subs(size_, old_expression, new_expression);
    }

    MemoryAccessPatternType clone() const override;

    void serialize_to_json(nlohmann::json& entry) override;

    static MemoryAccessPatternType create(symbolic::Expression size, bool not_sparse = false);
};

class NoAccessPattern : public MemoryAccessPattern {
private:
    NoAccessPattern() {}

public:
    bool empty() const override { return true; }

    bool every_element_accessed() const override { return true; }

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override {}

    static MemoryAccessPatternType instance();

    void serialize_to_json(nlohmann::json& entry) override;

    MemoryAccessPatternType clone() const override;
};

class PointerAccessMeta {
protected:
    PointerAccessMeta() = default;

public:
    virtual ~PointerAccessMeta() = default;

    /**
     * Despite this being a leak of the pointer,
     * the user will only use it for blocking accesses to the underlying data and not capture a reference to the data in
     * any way. Like a Rust temporary borrow for the duration of the LibNode and no more.
     */
    virtual bool no_capture() const = 0;

    /**
     * The pointer may be used to read from the backing data
     */
    virtual bool may_contain_reads() const = 0;

    /**
     * The pointe may be used to write to the backing data
     */
    virtual bool may_contain_writes() const = 0;

    virtual bool invalidated_after() const = 0;

    /**
     * Describes which elements are accessed (for example a function may only access the range of [ptr, ptr+8] bytes and
     * not touch or care about what comes after) Pointer access metadata only applies to the elements that are part of
     * the pattern.
     */
    virtual MemoryAccessPatternType access_read_pattern() const = 0;
    virtual MemoryAccessPatternType access_write_pattern() const = 0;

    PointerAccessType ref() const;

    virtual void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) = 0;

    virtual PointerAccessType clone() const = 0;

    virtual void serialize_to_json(nlohmann::json& entry) = 0;

    static PointerAccessType create_read_only(const symbolic::Expression& size, bool no_capture);
    static PointerAccessType create_invalidate();
    static PointerAccessType create_full_write_only(const symbolic::Expression& size, bool no_capture);
    static PointerAccessType
    create_generic(MemoryAccessPatternType read_pattern, MemoryAccessPatternType write_pattern, bool no_capture);
};


/**
 * The pointer is only used for reading. Like const* in Cpp.
 * Data pointed to will not change due to this.
 */
class PointerReadOnly : public PointerAccessMeta {
private:
    symbolic::Expression size_; // simplified until we have more than convex pattern
    bool no_capture_;

public:
    PointerReadOnly(symbolic::Expression size, bool no_capture = false);

    /**
     * Despite this being a leak of the pointer,
     * the user will only use it for blocking accesses to the underlying data and not keep a reference to the data in
     * any way. Like a Rust temporary borrow for the duration of the LibNode and no more.
     */
    bool no_capture() const override { return no_capture_; }

    bool may_contain_reads() const override { return true; }
    bool may_contain_writes() const override { return false; }

    bool invalidated_after() const override { return false; }

    /**
     * Describes which elements behind the pointer are actually read
     */
    MemoryAccessPatternType access_read_pattern() const override;

    MemoryAccessPatternType access_write_pattern() const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    PointerAccessType clone() const override;

    void serialize_to_json(nlohmann::json& entry) override;
};

/**
 * The pointer is used solely to write, no reading and over all of the data
 */
class PointerFullWriteOnly : public PointerAccessMeta {
private:
    symbolic::Expression size_; // simplified until we have more than convex pattern
    bool no_capture_;

public:
    PointerFullWriteOnly(symbolic::Expression size, bool no_capture = false);

    /**
     * Describes which elements are overwritten. If the underlying memory-area is larger,
     * other elements outside of the pattern remain unchanged
     */
    MemoryAccessPatternType access_write_pattern() const override;

    MemoryAccessPatternType access_read_pattern() const override;

    bool no_capture() const override { return no_capture_; }

    bool may_contain_reads() const override { return false; }
    bool may_contain_writes() const override { return true; }

    bool invalidated_after() const override { return false; }

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    PointerAccessType clone() const override;

    void serialize_to_json(nlohmann::json& entry) override;
};

/**
 * It is unknown what is done with this pointer or it is a mix of reads and writes.
 * This could overwrite some parts of the area pointed to, but leave others as is.
 * Assume the worst: data is made dirty by a black box. You know nothing about the contents after this
 */
class PointerGenericAccess : public PointerAccessMeta {
private:
    MemoryAccessPatternType read_pattern_;
    MemoryAccessPatternType write_pattern_;
    bool no_capture_;

public:
    PointerGenericAccess(MemoryAccessPatternType read_pattern, MemoryAccessPatternType write_pattern, bool no_capture);

    /**
     * Allows limiting the undefined behavior (assume required input as well as dirtying) to a specific pattern
     * No pattern means could be all the memory-area pointed to
     */
    MemoryAccessPatternType access_read_pattern() const override;

    MemoryAccessPatternType access_write_pattern() const override;

    bool no_capture() const override { return no_capture_; }

    bool may_contain_reads() const override;

    bool may_contain_writes() const override;

    bool invalidated_after() const override { return false; }

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    PointerAccessType clone() const override;

    void serialize_to_json(nlohmann::json& entry) override;
};

/**
 * Meaning the underlying memory will be deallocated and use of the pointer after this is no longer valid.
 * Does not represent a leak of the pointer.
 * Read-accesses to the pointer itself after this, but before an overwrite represent accessing most-likely invalid data
 * Memory accesses using this invalid pointer are catastrophic failures.
 */
class PointerInvalidate : public PointerAccessMeta {
public:
    bool no_capture() const override { return true; }

    bool may_contain_reads() const override { return false; }
    bool may_contain_writes() const override { return false; }

    bool invalidated_after() const override { return true; }

    MemoryAccessPatternType access_read_pattern() const override { return NoAccessPattern::instance(); }
    MemoryAccessPatternType access_write_pattern() const override { return NoAccessPattern::instance(); }

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override {}

    PointerAccessType clone() const override;

    void serialize_to_json(nlohmann::json& entry) override;
};

class PointerAccessMetaSerializer {
    friend class PointerAccessMeta;

public:
    static std::vector<PointerAccessType> deserialize_list(nlohmann::json::const_reference list);
    static std::vector<PointerAccessType>
    deserialize_list(nlohmann::json::const_iterator key, const nlohmann::json& parent);

    static PointerAccessType deserialize_read_only(nlohmann::json::const_reference entry);
    static PointerAccessType deserialize_write_only(nlohmann::json::const_reference entry);
    static PointerAccessType deserialize_generic(nlohmann::json::const_reference entry);

    static MemoryAccessPatternType deserialize_convex_pattern(nlohmann::json::const_reference entry);
    static MemoryAccessPatternType deserialize_access_pattern(nlohmann::json::const_reference entry);

    static nlohmann::json serialize(const std::vector<PointerAccessType>& vector);

    static PointerAccessType deserialize(const nlohmann::json& entry);
};


} // namespace sdfg::data_flow
