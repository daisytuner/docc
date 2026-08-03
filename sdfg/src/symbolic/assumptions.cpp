#include "sdfg/symbolic/assumptions.h"

#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"

namespace sdfg {
namespace symbolic {

Assumption::Assumption()
    : symbol_(symbolic::symbol("")), lower_bounds_(), upper_bounds_(), tight_lower_bound_(SymEngine::null),
      tight_upper_bound_(SymEngine::null), constraints_(), constant_(false), map_(SymEngine::null) {

      };

Assumption::Assumption(const Symbol symbol)
    : symbol_(symbol), lower_bounds_(), upper_bounds_(), tight_lower_bound_(SymEngine::null),
      tight_upper_bound_(SymEngine::null), constraints_(), constant_(false), map_(SymEngine::null) {

      };

Assumption::Assumption(const Assumption& a)
    : symbol_(a.symbol_), lower_bounds_(a.lower_bounds_), upper_bounds_(a.upper_bounds_),
      tight_lower_bound_(a.tight_lower_bound_), tight_upper_bound_(a.tight_upper_bound_), constraints_(a.constraints_),
      constant_(a.constant_), map_(a.map_) {

      };

Assumption& Assumption::operator=(const Assumption& a) {
    this->symbol_ = a.symbol_;
    this->lower_bounds_ = a.lower_bounds_;
    this->upper_bounds_ = a.upper_bounds_;
    this->tight_lower_bound_ = a.tight_lower_bound_;
    this->tight_upper_bound_ = a.tight_upper_bound_;
    this->constraints_ = a.constraints_;
    this->constant_ = a.constant_;
    this->map_ = a.map_;
    return *this;
};

const Symbol Assumption::symbol() const { return this->symbol_; };

const ExpressionSet& Assumption::lower_bounds() const { return this->lower_bounds_; }

void Assumption::add_lower_bound(const Expression lb) { this->lower_bounds_.insert(lb); }

bool Assumption::contains_lower_bound(const Expression lb) { return this->lower_bounds_.contains(lb); }

bool Assumption::remove_lower_bound(const Expression lb) { return this->lower_bounds_.erase(lb) > 0; }


const ExpressionSet& Assumption::upper_bounds() const { return this->upper_bounds_; }

void Assumption::add_upper_bound(const Expression ub) { this->upper_bounds_.insert(ub); }

bool Assumption::contains_upper_bound(const Expression ub) { return this->upper_bounds_.contains(ub); }

bool Assumption::remove_upper_bound(const Expression ub) { return this->upper_bounds_.erase(ub) > 0; }

const Expression Assumption::tight_lower_bound() const { return this->tight_lower_bound_; }

void Assumption::tight_lower_bound(const Expression tight_lb) { this->tight_lower_bound_ = tight_lb; }

const Expression Assumption::tight_upper_bound() const { return this->tight_upper_bound_; }

void Assumption::tight_upper_bound(const Expression tight_ub) { this->tight_upper_bound_ = tight_ub; }

const ExpressionSet& Assumption::constraints() const { return this->constraints_; }

void Assumption::add_constraint(const Expression c) { this->constraints_.insert(c); }

bool Assumption::contains_constraint(const Expression c) { return this->constraints_.contains(c); }

bool Assumption::remove_constraint(const Expression c) { return this->constraints_.erase(c) > 0; }

bool Assumption::constant() const { return constant_; };

void Assumption::constant(bool constant) { constant_ = constant; };

const Expression Assumption::map() const { return map_; };

void Assumption::map(const Expression map) { map_ = map; };

Assumption Assumption::create(const Symbol symbol, const types::IType& type) {
    if (auto scalar_type = dynamic_cast<const types::Scalar*>(&type)) {
        auto assum = Assumption(symbol);

        types::PrimitiveType primitive_type = scalar_type->primitive_type();
        switch (primitive_type) {
            case types::PrimitiveType::Bool: {
                assum.add_lower_bound(zero());
                assum.add_upper_bound(one());
                break;
            }
            case types::PrimitiveType::UInt8: {
                assum.add_lower_bound(integer(std::numeric_limits<uint8_t>::min()));
                assum.add_upper_bound(integer(std::numeric_limits<uint8_t>::max()));
                break;
            }
            case types::PrimitiveType::UInt16: {
                assum.add_lower_bound(integer(std::numeric_limits<uint16_t>::min()));
                assum.add_upper_bound(integer(std::numeric_limits<uint16_t>::max()));
                break;
            }
            case types::PrimitiveType::UInt32: {
                assum.add_lower_bound(integer(std::numeric_limits<uint32_t>::min()));
                assum.add_upper_bound(integer(std::numeric_limits<uint32_t>::max()));
                break;
            }
            case types::PrimitiveType::UInt64: {
                assum.add_lower_bound(integer(std::numeric_limits<uint64_t>::min()));
                assum.add_upper_bound(SymEngine::Inf);
                break;
            }
            case types::PrimitiveType::UInt128: {
                assum.add_lower_bound(integer(0));
                assum.add_upper_bound(SymEngine::Inf);
                break;
            }
            case types::PrimitiveType::Int8: {
                assum.add_lower_bound(integer(std::numeric_limits<int8_t>::min()));
                assum.add_upper_bound(integer(std::numeric_limits<int8_t>::max()));
                break;
            }
            case types::PrimitiveType::Int16: {
                assum.add_lower_bound(integer(std::numeric_limits<int16_t>::min()));
                assum.add_upper_bound(integer(std::numeric_limits<int16_t>::max()));
                break;
            }
            case types::PrimitiveType::Int32: {
                assum.add_lower_bound(integer(std::numeric_limits<int32_t>::min()));
                assum.add_upper_bound(integer(std::numeric_limits<int32_t>::max()));
                break;
            }
            case types::PrimitiveType::Int64: {
                assum.add_lower_bound(integer(std::numeric_limits<int64_t>::min()));
                assum.add_upper_bound(integer(std::numeric_limits<int64_t>::max()));
                break;
            }
            case types::PrimitiveType::Int128: {
                assum.add_lower_bound(SymEngine::NegInf);
                assum.add_upper_bound(SymEngine::Inf);
                break;
            }
            default: {
                throw std::runtime_error("Unsupported type");
            }
        };
        return assum;
    } else if (auto ptr_type = dynamic_cast<const types::Pointer*>(&type)) {
        auto assum = Assumption(symbol);
        assum.add_lower_bound(integer(std::numeric_limits<uint64_t>::min()));
        assum.add_upper_bound(SymEngine::Inf);
        return assum;
    } else {
        throw std::runtime_error("Unsupported type");
    }
}

Assumption::ReplaceResult Assumption::replace(const symbolic::ExpressionMapping& replacements) {
    bool replaced_some = true;

    replaced_some |= substitute(lower_bounds_, replacements);
    replaced_some |= substitute(upper_bounds_, replacements);
    if (!tight_lower_bound_.is_null()) {
        tight_lower_bound_ = tight_lower_bound_->subs(replacements);
    }
    if (!tight_lower_bound_.is_null()) {
        tight_upper_bound_ = tight_upper_bound_->subs(replacements);
    }
    replaced_some |= substitute(constraints_, replacements);
    if (!map_.is_null()) {
        map_ = map_->subs(replacements);
    }
    // update constant?

    auto replacement_it = replacements.find(symbol_);
    if (replacement_it != replacements.end()) {
        auto& replacement = replacement_it->second;
        if (SymEngine::is_a<SymEngine::Symbol>(*replacement)) {
            auto new_symbol = SymEngine::rcp_static_cast<const SymEngine::Symbol>(replacement);
            symbol_ = new_symbol;
            return ReplaceResult::IdChanged;
        } else {
            throw std::runtime_error(
                "Trying to replace Assumption symbol '" + this->symbol_->get_name() +
                "' with not a symbol: " + replacement->__str__()
            );
        }
    }
    return ReplaceResult::IdSame;
}

bool substitute(Assumptions& assumptions, const symbolic::ExpressionMapping& replacements) {
    bool remapped_some = false;

    std::vector<std::tuple<symbolic::Symbol, symbolic::Symbol>> replacements_vec;

    for (auto it = assumptions.begin(); it != assumptions.end(); ++it) {
        auto sym_change = it->second.replace(replacements);
        if (sym_change == Assumption::ReplaceResult::IdChanged) {
            replacements_vec.emplace_back(it->first, it->second.symbol());
        }
    }

    for (auto& [old_sym, new_sym] : replacements_vec) {
        auto new_it = assumptions.find(new_sym);
        if (new_it != assumptions.end()) { // new already exists, overwrite
            new_it->second = assumptions[old_sym];
            assumptions.erase(old_sym);
        } else {
            auto extracted = assumptions.extract(old_sym);
            extracted.key() = new_sym;
            assumptions.insert(std::move(extracted));
        }

        remapped_some = true;
    }

    return remapped_some;
}


} // namespace symbolic
} // namespace sdfg
