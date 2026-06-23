/**
 * @file polyhedral.h
 * @brief RAII facade over the Integer Set Library (ISL)
 *
 * This file is the single seam between sdfglib's symbolic layer and ISL. It owns
 * the ISL context lifetime via RAII and exposes move-only handle wrappers so that
 * callers never have to hand-manage `isl_*_free` / `isl_ctx_free` on every error
 * branch.
 *
 * The handles deliberately mirror ISL's take/give ownership model:
 * - methods that *consume* an ISL object expect the caller to `release()` the
 *   handle (transferring ownership to ISL), and
 * - methods that *produce* an ISL object wrap the raw pointer back into a handle.
 *
 * Construction order guarantees correct teardown: an `IslCtx` declared before any
 * `IslMap`/`IslSet`/`IslSpace` in the same scope is destroyed last, after every
 * object allocated in it has already been freed.
 *
 * @see utils.h for the symbolic-expression -> ISL string bridge
 * @see maps.h for the higher-level polyhedral predicates built on top of this
 */

#pragma once

#include <isl/ctx.h>
#include <isl/map.h>
#include <isl/set.h>
#include <isl/space.h>

#include "sdfg/symbolic/assumptions.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace symbolic {
namespace polyhedral {

/**
 * @brief Move-only RAII owner of an `isl_ctx`.
 *
 * Allocates a fresh context (with ISL_ON_ERROR_CONTINUE so parse failures yield
 * null pointers instead of aborting) and frees it on destruction. Because ISL
 * objects must outlive nothing beyond their context, declare the `IslCtx` first
 * in any scope so it is destroyed last.
 */
class IslCtx {
public:
    IslCtx();
    ~IslCtx();

    IslCtx(const IslCtx&) = delete;
    IslCtx& operator=(const IslCtx&) = delete;
    IslCtx(IslCtx&& other) noexcept;
    IslCtx& operator=(IslCtx&& other) noexcept;

    isl_ctx* get() const { return ctx_; }
    explicit operator bool() const { return ctx_ != nullptr; }

private:
    isl_ctx* ctx_;
};

/**
 * @brief Move-only RAII owner of a reference-counted ISL object.
 *
 * @tparam T      ISL object type (e.g. `isl_map`, `isl_set`, `isl_space`).
 * @tparam FreeFn The matching ISL free function (e.g. `isl_map_free`).
 */
template<typename T, T* (*FreeFn)(T*)>
class IslHandle {
public:
    IslHandle() : ptr_(nullptr) {}
    explicit IslHandle(T* ptr) : ptr_(ptr) {}
    ~IslHandle() { reset(); }

    IslHandle(const IslHandle&) = delete;
    IslHandle& operator=(const IslHandle&) = delete;

    IslHandle(IslHandle&& other) noexcept : ptr_(other.ptr_) { other.ptr_ = nullptr; }
    IslHandle& operator=(IslHandle&& other) noexcept {
        if (this != &other) {
            reset();
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }

    /// Borrow the raw pointer without transferring ownership.
    T* get() const { return ptr_; }

    /// Relinquish ownership to the caller (e.g. before passing to an ISL
    /// function that *takes* the object). The handle becomes empty.
    T* release() {
        T* p = ptr_;
        ptr_ = nullptr;
        return p;
    }

    /// Free the current object (if any) and adopt @p p.
    void reset(T* p = nullptr) {
        if (ptr_ != nullptr) {
            FreeFn(ptr_);
        }
        ptr_ = p;
    }

    explicit operator bool() const { return ptr_ != nullptr; }

private:
    T* ptr_;
};

using IslMap = IslHandle<isl_map, isl_map_free>;
using IslSet = IslHandle<isl_set, isl_set_free>;
using IslSpace = IslHandle<isl_space, isl_space_free>;

/**
 * @brief Decides whether two access expressions denote the same cell at every
 *        point of the domain implied by @p assums.
 *
 * Semantically tests \f$\forall x \in \mathrm{Dom}:\ f(x) = g(x)\f$ using ISL,
 * i.e. affine/parametric equality rather than structural (SymEngine) equality.
 * This proves equalities such as `i*N + k` ≡ `k + N*i` and rejects shifted
 * accesses such as `A[i]` vs `A[i-1]` (a scan, not a reduction).
 *
 * @param f      First (e.g. write) access subset.
 * @param g      Second (e.g. read-back) access subset.
 * @param indvar The loop induction variable (used to scope the domain).
 * @param assums Assumptions/bounds shared by both accesses (same block/scope).
 * @return true iff @p f and @p g are provably equal across the whole domain.
 *
 * Scalar accesses (empty subsets) are trivially equal. On any ISL parse/analysis
 * failure the result is conservatively `false`.
 */
bool equal_on_domain(const MultiExpression& f, const MultiExpression& g, const Symbol indvar, const Assumptions& assums);

} // namespace polyhedral
} // namespace symbolic
} // namespace sdfg
