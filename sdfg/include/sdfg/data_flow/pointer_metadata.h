#pragma once

#include <memory>
#include <nlohmann/json.hpp>
#include <optional>

#include "sdfg/data_flow/library_nodes/math/tensor/tensor_layout.h"
#include "sdfg/symbolic/symbolic.h"
#include "symengine/subs.h"

namespace sdfg::data_flow {

class PointerAccessMeta;

template<typename T>
struct PtrMetaDeleter {
    bool should_delete_;

    PtrMetaDeleter(bool should_delete = true) : should_delete_(should_delete) {
    }

    void operator()(T* ptr) const {
        if (should_delete_) {
            delete ptr;
        }
    }
};

typedef PtrMetaDeleter<PointerAccessMeta> PtrAccessDeleter;
typedef std::unique_ptr<PointerAccessMeta, PtrAccessDeleter> PointerAccessType;

/**
 * One direction (read or write) of a generic pointer access. The region is an
 * affine layout describing the accessed elements; `std::nullopt` means the
 * access is unbounded/unknown. `covers_all` distinguishes an exact region from
 * an over-approximation (some elements within the region may not be touched).
 */
struct AccessRegion {
    bool may_access = false;
    std::optional<math::tensor::TensorLayout> region = std::nullopt;
    bool covers_all = false;
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
     * The affine region (shape/strides/offset, in elements) this pointer reads /
     * writes, or nullptr when the access is unbounded or absent. Replaces the old
     * convex access patterns: a flat convex span of N elements is a 1-D layout
     * `{N}`, a structured operand carries its real shape/strides.
     */
    virtual const math::tensor::TensorLayout* read_layout() const {
        return nullptr;
    }
    virtual const math::tensor::TensorLayout* write_layout() const {
        return nullptr;
    }

    /**
     * Whether every element of the region is actually accessed, as opposed to an
     * over-approximation whose interior may contain untouched elements.
     */
    virtual bool read_covers_all() const {
        return false;
    }
    virtual bool write_covers_all() const {
        return false;
    }

    PointerAccessType ref() const;

    virtual void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) = 0;
    virtual void replace(const symbolic::ExpressionMapping& replacements) = 0;

    virtual PointerAccessType clone() const = 0;

    virtual void serialize_to_json(nlohmann::json& entry) = 0;

    static PointerAccessType create_read_only(
        const symbolic::Expression& size,
        bool no_capture,
        std::optional<math::tensor::TensorLayout> layout = std::nullopt
    );
    static PointerAccessType create_invalidate();
    static PointerAccessType create_full_write_only(
        const symbolic::Expression& size,
        bool no_capture,
        std::optional<math::tensor::TensorLayout> layout = std::nullopt
    );
    static PointerAccessType create_generic(AccessRegion read, AccessRegion write, bool no_capture);
};


/**
 * The pointer is only used for reading. Like const* in Cpp.
 * Data pointed to will not change due to this.
 */
class PointerReadOnly : public PointerAccessMeta {
private:
    std::optional<math::tensor::TensorLayout> region_; // nullopt = unbounded
    bool no_capture_;

public:
    PointerReadOnly(std::optional<math::tensor::TensorLayout> region, bool no_capture = false);

    /**
     * Despite this being a leak of the pointer,
     * the user will only use it for blocking accesses to the underlying data and not keep a reference to the data in
     * any way. Like a Rust temporary borrow for the duration of the LibNode and no more.
     */
    bool no_capture() const override {
        return no_capture_;
    }

    bool may_contain_reads() const override {
        return true;
    }
    bool may_contain_writes() const override {
        return false;
    }

    bool invalidated_after() const override {
        return false;
    }

    const math::tensor::TensorLayout* read_layout() const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;
    void replace(const symbolic::ExpressionMapping& replacements) override;

    PointerAccessType clone() const override;

    void serialize_to_json(nlohmann::json& entry) override;
};

/**
 * The pointer is used solely to write, no reading and over all of the data
 */
class PointerFullWriteOnly : public PointerAccessMeta {
private:
    std::optional<math::tensor::TensorLayout> region_; // nullopt = unbounded
    bool no_capture_;

public:
    PointerFullWriteOnly(std::optional<math::tensor::TensorLayout> region, bool no_capture = false);

    const math::tensor::TensorLayout* write_layout() const override;

    bool write_covers_all() const override {
        return true;
    }

    bool no_capture() const override {
        return no_capture_;
    }

    bool may_contain_reads() const override {
        return false;
    }
    bool may_contain_writes() const override {
        return true;
    }

    bool invalidated_after() const override {
        return false;
    }

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;
    void replace(const symbolic::ExpressionMapping& replacements) override;

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
    AccessRegion read_;
    AccessRegion write_;
    bool no_capture_;

public:
    PointerGenericAccess(AccessRegion read, AccessRegion write, bool no_capture);

    const math::tensor::TensorLayout* read_layout() const override;
    const math::tensor::TensorLayout* write_layout() const override;

    bool read_covers_all() const override {
        return read_.covers_all;
    }
    bool write_covers_all() const override {
        return write_.covers_all;
    }

    bool no_capture() const override {
        return no_capture_;
    }

    bool may_contain_reads() const override {
        return read_.may_access;
    }

    bool may_contain_writes() const override {
        return write_.may_access;
    }

    bool invalidated_after() const override {
        return false;
    }

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;
    void replace(const symbolic::ExpressionMapping& replacements) override;

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
    bool no_capture() const override {
        return true;
    }

    bool may_contain_reads() const override {
        return false;
    }
    bool may_contain_writes() const override {
        return false;
    }

    bool invalidated_after() const override {
        return true;
    }

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override {
    }
    void replace(const symbolic::ExpressionMapping& replacements) override {
    }

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

    static nlohmann::json serialize(const std::vector<PointerAccessType>& vector);

    static PointerAccessType deserialize(const nlohmann::json& entry);
};


} // namespace sdfg::data_flow
