#include "sdfg/tiles/layout.h"

#include <algorithm>
#include <sstream>
#include <utility>

namespace sdfg {
namespace tiles {

using symbolic::Expression;
using symbolic::MultiExpression;

namespace {

Expression product_of(const MultiExpression& xs, size_t begin, size_t end) {
    Expression p = symbolic::integer(1);
    for (size_t i = begin; i < end; ++i) {
        p = symbolic::mul(p, xs[i]);
    }
    return p;
}

bool is_int(const Expression& e) { return SymEngine::is_a<SymEngine::Integer>(*e); }
long long as_ll(const Expression& e) { return SymEngine::rcp_static_cast<const SymEngine::Integer>(e)->as_int(); }

/// Modulo that folds on concrete integers (symbolic::mod leaves an opaque imod).
Expression imod(const Expression& a, const Expression& b) {
    if (is_int(a) && is_int(b) && as_ll(b) != 0) {
        return symbolic::integer(as_ll(a) % as_ll(b));
    }
    return symbolic::mod(a, b);
}

/// Structural equality after simplification; conservative (false when unknown).
bool prov_eq(const Expression& a, const Expression& b) {
    if (is_int(a) && is_int(b)) {
        return as_ll(a) == as_ll(b);
    }
    return symbolic::eq(symbolic::simplify(a), symbolic::simplify(b));
}

} // namespace

Layout::Layout(MultiExpression shape, MultiExpression stride, Expression offset)
    : shape_(std::move(shape)), stride_(std::move(stride)), offset_(std::move(offset)) {
    // Default: colex row-major-contiguous strides (first mode fastest).
    if (stride_.empty() && !shape_.empty()) {
        stride_.resize(shape_.size());
        Expression run = symbolic::integer(1);
        for (size_t k = 0; k < shape_.size(); ++k) {
            stride_[k] = run;
            run = symbolic::mul(run, shape_[k]);
        }
    }
}

Layout Layout::identity(const Expression& n) { return Layout({n}, {symbolic::integer(1)}, symbolic::integer(0)); }

Layout Layout::from_tensor(const math::tensor::TensorLayout& t) {
    // TensorLayout is row-major (first mode outermost/slowest); the algebra is
    // colex (first mode fastest), so reverse mode order at the boundary.
    MultiExpression shape(t.shape().rbegin(), t.shape().rend());
    MultiExpression stride(t.strides().rbegin(), t.strides().rend());
    return Layout(std::move(shape), std::move(stride), t.offset());
}

math::tensor::TensorLayout Layout::to_tensor() const {
    MultiExpression shape(shape_.rbegin(), shape_.rend());
    MultiExpression stride(stride_.rbegin(), stride_.rend());
    return math::tensor::TensorLayout(shape, stride, offset_);
}

Expression Layout::size() const {
    if (shape_.empty()) {
        return symbolic::integer(1);
    }
    return product_of(shape_, 0, shape_.size());
}

Expression Layout::cosize() const {
    // Exact for non-negative strides: max image is at all-coords-max.
    Expression c = offset_;
    for (size_t k = 0; k < shape_.size(); ++k) {
        c = symbolic::add(c, symbolic::mul(symbolic::sub(shape_[k], symbolic::integer(1)), stride_[k]));
    }
    return symbolic::add(c, symbolic::integer(1));
}

MultiExpression Layout::coords(const Expression& index) const {
    MultiExpression out(shape_.size());
    Expression divisor = symbolic::integer(1);
    for (size_t k = 0; k < shape_.size(); ++k) {
        Expression q = symbolic::div(index, divisor);
        out[k] = imod(q, shape_[k]);
        divisor = symbolic::mul(divisor, shape_[k]);
    }
    return out;
}

Expression Layout::apply_coords(const MultiExpression& c) const {
    Expression acc = offset_;
    for (size_t k = 0; k < stride_.size() && k < c.size(); ++k) {
        acc = symbolic::add(acc, symbolic::mul(c[k], stride_[k]));
    }
    return acc;
}

Expression Layout::apply(const Expression& index) const { return apply_coords(coords(index)); }

bool Layout::is_bijective() const {
    // A layout is a permutation of [0,size) iff its (non-unit) modes can be
    // ordered so strides are 1, s0, s0*s1, ... (compact). Greedy match.
    size_t n = rank();
    std::vector<bool> used(n, false);
    for (size_t k = 0; k < n; ++k) {
        if (prov_eq(shape_[k], symbolic::integer(1))) {
            used[k] = true; // unit modes contribute nothing
        }
    }
    Expression run = symbolic::integer(1);
    for (;;) {
        bool found = false;
        for (size_t k = 0; k < n; ++k) {
            if (!used[k] && prov_eq(stride_[k], run)) {
                used[k] = true;
                run = symbolic::mul(run, shape_[k]);
                found = true;
                break;
            }
        }
        if (!found) {
            break;
        }
    }
    return std::all_of(used.begin(), used.end(), [](bool b) { return b; });
}

bool Layout::is_injective() const {
    // Non-unit modes sorted by (constant) stride ascending must not overlap:
    // stride_k >= product of the extents already placed below it.
    struct M {
        long long shape;
        long long stride;
    };
    std::vector<M> ms;
    for (size_t k = 0; k < rank(); ++k) {
        if (prov_eq(shape_[k], symbolic::integer(1))) {
            continue; // unit modes contribute nothing
        }
        if (!is_int(shape_[k]) || !is_int(stride_[k])) {
            return false; // conservative: cannot prove non-overlap symbolically
        }
        ms.push_back({as_ll(shape_[k]), as_ll(stride_[k])});
    }
    std::sort(ms.begin(), ms.end(), [](const M& a, const M& b) { return a.stride < b.stride; });
    long long run = 1;
    for (const auto& m : ms) {
        if (m.stride < run) {
            return false; // overlaps an already-placed mode
        }
        run = m.stride * m.shape;
    }
    return true;
}

bool Layout::operator==(const Layout& other) const {
    if (rank() != other.rank()) {
        return false;
    }
    if (!prov_eq(offset_, other.offset_)) {
        return false;
    }
    for (size_t k = 0; k < rank(); ++k) {
        if (!prov_eq(shape_[k], other.shape_[k]) || !prov_eq(stride_[k], other.stride_[k])) {
            return false;
        }
    }
    return true;
}

void Layout::collect_symbols(symbolic::SymbolSet& set) const {
    auto add = [&](const Expression& e) {
        for (const auto& s : symbolic::atoms(e)) {
            set.insert(s);
        }
    };
    for (const auto& e : shape_) {
        add(e);
    }
    for (const auto& e : stride_) {
        add(e);
    }
    add(offset_);
}

void Layout::replace_symbols(const symbolic::ExpressionMapping& replacements) {
    for (auto& e : shape_) {
        e = symbolic::subs(e, replacements);
    }
    for (auto& e : stride_) {
        e = symbolic::subs(e, replacements);
    }
    offset_ = symbolic::subs(offset_, replacements);
}

std::string Layout::to_string() const {
    std::ostringstream os;
    os << "(";
    for (size_t k = 0; k < shape_.size(); ++k) {
        os << (k ? "," : "") << *shape_[k];
    }
    os << "):(";
    for (size_t k = 0; k < stride_.size(); ++k) {
        os << (k ? "," : "") << *stride_[k];
    }
    os << ")+" << *offset_;
    return os.str();
}

std::ostream& operator<<(std::ostream& stream, const Layout& layout) { return stream << layout.to_string(); }

// ================================ Operators =================================

Layout coalesce(const Layout& A) {
    MultiExpression ns, nd;
    for (size_t k = 0; k < A.rank(); ++k) {
        if (prov_eq(A.shape()[k], symbolic::integer(1))) {
            continue; // drop unit modes
        }
        // Merge with the previous mode if contiguous: d_k == d_{prev} * s_{prev}.
        if (!ns.empty() && prov_eq(A.stride()[k], symbolic::mul(nd.back(), ns.back()))) {
            ns.back() = symbolic::mul(ns.back(), A.shape()[k]);
        } else {
            ns.push_back(A.shape()[k]);
            nd.push_back(A.stride()[k]);
        }
    }
    return Layout(std::move(ns), std::move(nd), A.offset());
}

Layout concat(const Layout& A, const Layout& B) {
    MultiExpression ns = A.shape();
    MultiExpression nd = A.stride();
    for (const auto& s : B.shape()) {
        ns.push_back(s);
    }
    for (const auto& d : B.stride()) {
        nd.push_back(d);
    }
    return Layout(std::move(ns), std::move(nd), symbolic::add(A.offset(), B.offset()));
}

} // namespace tiles
} // namespace sdfg
