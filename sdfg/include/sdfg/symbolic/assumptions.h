/**
 * @file assumptions.h
 * @brief Symbol assumptions for reasoning about bounds, constness, and evolution
 *
 * Assumptions are a critical component of the symbolic system that enable reasoning about
 * ranges of symbols, how they evolve, and whether they are constant. Examples:
 * - Ranges of memory accesses
 * - Ranges and evolution of loop iteration variables
 *
 * ## Assumption System
 *
 * Each symbol can have associated assumptions that describe its properties:
 * - **Bounds**: Lower and upper bounds that constrain the symbol's possible values
 * - **Tight bounds**: Exact minimum and maximum values the symbol can take
 * - **Constness**: Whether the symbol's value remains constant
 * - **Map**: How the symbol evolves (e.g., in loop iterations)
 *
 * ## Example Usage
 *
 * @code
 * // Create a symbol with type-based assumptions
 * auto i = symbolic::symbol("i");
 * auto int32_type = types::Scalar::create(types::PrimitiveType::Int32);
 * auto assumption = Assumption::create(i, int32_type);
 *
 * // Add loop bounds (e.g., 0 <= i < 10)
 * assumption.add_lower_bound(symbolic::zero());
 * assumption.add_upper_bound(symbolic::integer(10));
 *
 * // Mark as non-constant (evolves in loop)
 * assumption.constant(false);
 *
 * // Set evolution map (e.g., i' = i + 1)
 * assumption.map(symbolic::add(i, symbolic::one()));
 * @endcode
 *
 * @see extreme_values.h for computing bounds of expressions using assumptions
 * @see sets.h for integer set operations using assumptions
 */

#pragma once

#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace symbolic {

/**
 * @class Assumption
 * @brief Represents assumptions about a symbol's properties and behavior
 */
class Assumption {
private:
    // Copy-on-write payload: scope propagation copies whole Assumptions maps, so copies share it.
    struct Data {
        ExpressionSet lower_bounds;
        ExpressionSet upper_bounds;
        Expression tight_lower_bound;
        Expression tight_upper_bound;
        ExpressionSet constraints;
        bool constant = false;
        Expression map;
    };

    Symbol symbol_;
    std::shared_ptr<Data> data_;

    Data& mut();

public:
    /**
     * @brief Default constructor creating an empty assumption
     */
    Assumption();

    /**
     * @brief Constructs an assumption for a specific symbol
     * @param symbol The symbol this assumption is for
     */
    Assumption(const Symbol symbol);

    /**
     * @brief Copy constructor
     * @param a Assumption to copy
     */
    Assumption(const Assumption& a);

    /**
     * @brief Assignment operator
     * @param a Assumption to assign from
     * @return Reference to this assumption
     */
    Assumption& operator=(const Assumption& a);

    /**
     * @brief Gets the symbol this assumption is for
     * @return The symbol
     */
    const Symbol symbol() const;

    /**
     * @brief Gets all lower bound constraints
     * @return Set of all lower bound expressions
     *
     * A symbol may have multiple lower bound constraints (e.g., from different
     * analysis passes). The effective lower bound is the maximum of these.
     */
    const ExpressionSet& lower_bounds() const;

    /**
     * @brief Adds a lower bound constraint
     * @param lb The lower bound expression to add
     *
     * The symbol is constrained to be greater than or equal to this expression.
     * Multiple lower bounds can be added; the effective bound is their maximum.
     */
    void add_lower_bound(const Expression lb);

    /**
     * @brief Checks if a specific lower bound exists
     * @param lb The lower bound expression to check
     * @return true if this lower bound is present
     */
    bool contains_lower_bound(const Expression lb);

    /**
     * @brief Removes a lower bound constraint
     * @param lb The lower bound expression to remove
     * @return true if the bound was found and removed
     */
    bool remove_lower_bound(const Expression lb);

    /**
     * @brief Gets all upper bound constraints
     * @return Set of all upper bound expressions
     *
     * A symbol may have multiple upper bound constraints (e.g., from different
     * analysis passes). The effective upper bound is the minimum of these.
     */
    const ExpressionSet& upper_bounds() const;

    /**
     * @brief Adds an upper bound constraint
     * @param ub The upper bound expression to add
     *
     * The symbol is constrained to be less than or equal to this expression.
     * Multiple upper bounds can be added; the effective bound is their minimum.
     */
    void add_upper_bound(const Expression ub);

    /**
     * @brief Checks if a specific upper bound exists
     * @param ub The upper bound expression to check
     * @return true if this upper bound is present
     */
    bool contains_upper_bound(const Expression ub);

    /**
     * @brief Removes an upper bound constraint
     * @param ub The upper bound expression to remove
     * @return true if the bound was found and removed
     */
    bool remove_upper_bound(const Expression ub);

    /**
     * @brief Gets the tight lower bound
     * @return The exact minimum value the symbol can take, or null if unknown
     *
     * Unlike lower_bounds which are constraints, the tight lower bound represents
     * the exact minimum value the symbol will take during execution.
     */
    const Expression tight_lower_bound() const;

    /**
     * @brief Sets the tight lower bound
     * @param tight_lb The exact minimum value
     *
     * Sets the exact minimum value the symbol will take. This is more precise
     * than lower bound constraints.
     */
    void tight_lower_bound(const Expression tight_lb);

    /**
     * @brief Gets the tight upper bound
     * @return The exact maximum value the symbol can take, or null if unknown
     *
     * Unlike upper_bounds which are constraints, the tight upper bound represents
     * the exact maximum value the symbol will take during execution.
     */
    const Expression tight_upper_bound() const;

    /**
     * @brief Sets the tight upper bound
     * @param tight_ub The exact maximum value
     *
     * Sets the exact maximum value the symbol will take. This is more precise
     * than upper bound constraints.
     */
    void tight_upper_bound(const Expression tight_ub);

    /**
     * @brief Gets the constraints registered on this symbol
     * @return Set of constraint expressions, each with implicit `<= 0` semantics
     *
     * A constraint is an affine expression involving this symbol together with
     * other symbols (e.g. `i + j - 15` to express `i + j <= 15`). The same
     * constraint expression is registered under each non-constant symbol it
     * mentions, so consumers holding any one of those symbols can find the
     * coupling fact via `constraints()`.
     *
     * Equality `a = b` is represented as two constraints: `a - b` and `b - a`.
     */
    const ExpressionSet& constraints() const;

    /**
     * @brief Adds a constraint involving this symbol
     * @param c Expression with implicit `<= 0` semantics
     *
     * The expression must mention this symbol (otherwise it does not constrain
     * it) and should be affine in every non-constant symbol it mentions, so
     * that downstream consumers can project it onto a single variable when
     * needed.
     */
    void add_constraint(const Expression c);

    /**
     * @brief Checks if a specific constraint is present
     */
    bool contains_constraint(const Expression c);

    /**
     * @brief Removes a constraint
     * @return true if the constraint was found and removed
     */
    bool remove_constraint(const Expression c);

    /**
     * @brief Checks if the symbol is constant
     * @return true if the symbol's value does not change
     *
     * A constant symbol has the same value throughout its scope. Non-constant
     * symbols may evolve (e.g., loop iteration variables).
     */
    bool constant() const;

    /**
     * @brief Sets whether the symbol is constant
     * @param constant true if the symbol is constant, false if it evolves
     */
    void constant(bool constant);

    /**
     * @brief Gets the evolution map for this symbol
     * @return Expression describing how the symbol evolves, or null if none
     *
     * The map describes how the symbol's value changes, typically in loop iterations.
     * For example, a loop counter might have map(i) = i + 1, indicating it increments
     * by 1 each iteration.
     */
    const Expression map() const;

    /**
     * @brief Sets the evolution map for this symbol
     * @param map Expression describing how the symbol evolves
     *
     * Sets how the symbol evolves over time or iterations. This is used to track
     * induction variables in loops and analyze their behavior.
     */
    void map(const Expression map);

    /**
     * @brief Creates an assumption with bounds derived from a type
     * @param symbol The symbol to create assumption for
     * @param type The type to derive bounds from
     * @return Assumption with bounds matching the type's range
     *
     * Creates an assumption initialized with bounds appropriate for the given type.
     * For example, an Int32 type would set bounds to [-2^31, 2^31-1].
     *
     * @code
     * auto i = symbolic::symbol("i");
     * auto int32_type = types::Scalar::create(types::PrimitiveType::Int32);
     * auto assumption = Assumption::create(i, int32_type);
     * @endcode
     */
    static Assumption create(const Symbol symbol, const types::IType& type);

    enum class ReplaceResult { IdSame = 0, IdChanged };

    ReplaceResult replace(const symbolic::ExpressionMapping& replacements);
};

/**
 * @brief Map from symbols to their assumptions
 *
 * This type represents a collection of assumptions for multiple symbols.
 * Each symbol maps to its Assumption containing bounds, constness, and evolution.
 *
 * It behaves as an `std::unordered_map` from symbols to assumptions, but may share an immutable base map with other
 * instances and hold only the entries that differ from it. Looking an entry up finds it in either; writing to one
 * of the base copies it into the instance first; iterating visits the instance's entries, then the base's others.
 * Iterating, or erasing from, a mutable instance gives it its own copy of the base first.
 */
class Assumptions {
public:
    using Map = std::unordered_map<Symbol, Assumption, SymEngine::RCPBasicHash, SymEngine::RCPBasicKeyEq>;
    using key_type = Symbol;
    using mapped_type = Assumption;
    using value_type = Map::value_type;
    using size_type = std::size_t;
    using iterator = Map::iterator;

    /// Visits the own entries, then the base's entries the own ones do not shadow.
    class const_iterator {
        friend class Assumptions;
        const Assumptions* owner_ = nullptr;
        Map::const_iterator own_;
        Map::const_iterator base_;
        bool in_base_ = false;

        const_iterator(const Assumptions* owner, Map::const_iterator own, Map::const_iterator base, bool in_base)
            : owner_(owner), own_(own), base_(base), in_base_(in_base) {
        }

        void skip_shadowed() {
            while (base_ != owner_->base_->end() && owner_->own_.find(base_->first) != owner_->own_.end()) {
                ++base_;
            }
        }

    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = Assumptions::value_type;
        using difference_type = std::ptrdiff_t;
        using pointer = const value_type*;
        using reference = const value_type&;

        const_iterator() = default;

        reference operator*() const {
            return in_base_ ? *base_ : *own_;
        }
        pointer operator->() const {
            return in_base_ ? &*base_ : &*own_;
        }

        const_iterator& operator++() {
            if (in_base_) {
                ++base_;
                skip_shadowed();
                return *this;
            }
            ++own_;
            if (own_ == owner_->own_.end() && owner_->base_) {
                in_base_ = true;
                base_ = owner_->base_->begin();
                skip_shadowed();
            }
            return *this;
        }

        const_iterator operator++(int) {
            auto copy = *this;
            ++*this;
            return copy;
        }

        bool operator==(const const_iterator& other) const {
            if (in_base_ != other.in_base_) {
                return false;
            }
            return in_base_ ? base_ == other.base_ : own_ == other.own_;
        }
        bool operator!=(const const_iterator& other) const {
            return !(*this == other);
        }
    };

    Assumptions() = default;
    Assumptions(std::initializer_list<value_type> entries) : own_(entries) {
    }
    Assumptions(const Map& entries) : own_(entries) {
    }
    Assumptions(Map&& entries) : own_(std::move(entries)) {
    }

    /// An instance holding no entries of its own over @p base, which it shares.
    static Assumptions over(std::shared_ptr<const Map> base) {
        Assumptions result;
        result.base_ = std::move(base);
        if (result.base_ && result.base_->empty()) {
            result.base_.reset();
        }
        return result;
    }

    size_type size() const {
        return own_.size() + (base_ ? base_->size() : 0) - shadowed_;
    }
    bool empty() const {
        return size() == 0;
    }

    void clear() {
        own_.clear();
        base_.reset();
        shadowed_ = 0;
    }

    const_iterator begin() const {
        if (own_.empty() && base_) {
            const_iterator it(this, own_.end(), base_->begin(), true);
            it.skip_shadowed();
            return it;
        }
        return const_iterator(this, own_.begin(), {}, false);
    }
    const_iterator end() const {
        return base_ ? const_iterator(this, own_.end(), base_->end(), true)
                     : const_iterator(this, own_.end(), {}, false);
    }
    const_iterator cbegin() const {
        return begin();
    }
    const_iterator cend() const {
        return end();
    }

    /// Mutable iteration: the base is copied in first. The end is the own entries', which mutable finds return too.
    iterator begin() {
        densify();
        return own_.begin();
    }
    iterator end() {
        return own_.end();
    }

    const_iterator find(const Symbol& symbol) const {
        auto own = own_.find(symbol);
        if (own != own_.end()) {
            return const_iterator(this, own, {}, false);
        }
        if (base_) {
            auto base = base_->find(symbol);
            if (base != base_->end()) {
                return const_iterator(this, own_.end(), base, true);
            }
        }
        return end();
    }

    /// A mutable entry, copied from the base first if only the base holds it.
    iterator find(const Symbol& symbol) {
        auto own = own_.find(symbol);
        if (own != own_.end() || !base_) {
            return own;
        }
        auto base = base_->find(symbol);
        if (base == base_->end()) {
            return own_.end();
        }
        return promote(*base);
    }

    bool contains(const Symbol& symbol) const {
        return own_.contains(symbol) || (base_ && base_->contains(symbol));
    }
    size_type count(const Symbol& symbol) const {
        return contains(symbol) ? 1 : 0;
    }

    const Assumption& at(const Symbol& symbol) const {
        auto own = own_.find(symbol);
        if (own != own_.end()) {
            return own->second;
        }
        if (base_) {
            auto base = base_->find(symbol);
            if (base != base_->end()) {
                return base->second;
            }
        }
        throw std::out_of_range("Assumptions::at: no assumption for " + symbol->get_name());
    }

    Assumption& at(const Symbol& symbol) {
        auto it = find(symbol);
        if (it == own_.end()) {
            throw std::out_of_range("Assumptions::at: no assumption for " + symbol->get_name());
        }
        return it->second;
    }

    Assumption& operator[](const Symbol& symbol) {
        auto it = find(symbol);
        if (it != own_.end()) {
            return it->second;
        }
        return own_[symbol];
    }

    /// Inserts @p entry unless its symbol has an entry, here or in the base; true if it did. The entry the symbol has
    /// either way is a mutable one.
    std::pair<iterator, bool> insert(const value_type& entry) {
        auto present = find(entry.first);
        if (present != own_.end()) {
            return {present, false};
        }
        return own_.insert(entry);
    }
    std::pair<iterator, bool> insert(value_type&& entry) {
        auto present = find(entry.first);
        if (present != own_.end()) {
            return {present, false};
        }
        return own_.insert(std::move(entry));
    }
    template<typename InputIt>
    void insert(InputIt first, InputIt last) {
        for (; first != last; ++first) {
            insert(value_type(*first));
        }
    }
    template<typename... Args>
    std::pair<iterator, bool> emplace(Args&&... args) {
        return insert(value_type(std::forward<Args>(args)...));
    }
    template<typename... Args>
    std::pair<iterator, bool> try_emplace(const Symbol& symbol, Args&&... args) {
        auto present = find(symbol);
        if (present != own_.end()) {
            return {present, false};
        }
        return own_.try_emplace(symbol, std::forward<Args>(args)...);
    }
    template<typename M>
    std::pair<iterator, bool> insert_or_assign(const Symbol& symbol, M&& value) {
        auto present = find(symbol);
        if (present != own_.end()) {
            present->second = std::forward<M>(value);
            return {present, false};
        }
        return own_.insert_or_assign(symbol, std::forward<M>(value));
    }

    /// Node handles, as for renaming an entry's symbol, work on own entries: the base is copied in first.
    Map::node_type extract(const Symbol& symbol) {
        densify();
        return own_.extract(symbol);
    }
    Map::insert_return_type insert(Map::node_type&& node) {
        densify();
        return own_.insert(std::move(node));
    }

    size_type erase(const Symbol& symbol) {
        densify();
        return own_.erase(symbol);
    }
    iterator erase(iterator position) {
        densify();
        return own_.erase(position);
    }

private:
    Map own_;
    std::shared_ptr<const Map> base_;
    // The own entries that shadow one of the base's.
    size_type shadowed_ = 0;

    iterator promote(const value_type& base_entry) {
        ++shadowed_;
        return own_.insert(base_entry).first;
    }

    void densify() {
        if (!base_) {
            return;
        }
        for (auto& entry : *base_) {
            own_.insert(entry);
        }
        base_.reset();
        shadowed_ = 0;
    }
};

bool substitute(Assumptions& assumptions, const symbolic::ExpressionMapping& replacements);
;

} // namespace symbolic
} // namespace sdfg
