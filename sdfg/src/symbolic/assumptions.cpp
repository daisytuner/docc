#include "sdfg/symbolic/assumptions.h"

#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"

namespace sdfg {
namespace symbolic {

Assumption::Assumption() : symbol_(symbolic::symbol("")), data_(std::make_shared<Data>()) {};

Assumption::Assumption(const Symbol symbol) : symbol_(symbol), data_(std::make_shared<Data>()) {};

Assumption::Assumption(const Assumption& a) : symbol_(a.symbol_), data_(a.data_) {};

Assumption& Assumption::operator=(const Assumption& a) {
    this->symbol_ = a.symbol_;
    this->data_ = a.data_;
    return *this;
};

Assumption::Data& Assumption::mut() {
    if (data_.use_count() > 1) {
        data_ = std::make_shared<Data>(*data_);
    }
    return *data_;
}

const Symbol Assumption::symbol() const {
    return this->symbol_;
};

const ExpressionSet& Assumption::lower_bounds() const {
    return data_->lower_bounds;
}

void Assumption::add_lower_bound(const Expression lb) {
    if (!data_->lower_bounds.contains(lb)) {
        mut().lower_bounds.insert(lb);
    }
}

bool Assumption::contains_lower_bound(const Expression lb) {
    return data_->lower_bounds.contains(lb);
}

bool Assumption::remove_lower_bound(const Expression lb) {
    return data_->lower_bounds.contains(lb) && mut().lower_bounds.erase(lb) > 0;
}

const ExpressionSet& Assumption::upper_bounds() const {
    return data_->upper_bounds;
}

void Assumption::add_upper_bound(const Expression ub) {
    if (!data_->upper_bounds.contains(ub)) {
        mut().upper_bounds.insert(ub);
    }
}

bool Assumption::contains_upper_bound(const Expression ub) {
    return data_->upper_bounds.contains(ub);
}

bool Assumption::remove_upper_bound(const Expression ub) {
    return data_->upper_bounds.contains(ub) && mut().upper_bounds.erase(ub) > 0;
}

const Expression Assumption::tight_lower_bound() const {
    return data_->tight_lower_bound;
}

void Assumption::tight_lower_bound(const Expression tight_lb) {
    mut().tight_lower_bound = tight_lb;
}

const Expression Assumption::tight_upper_bound() const {
    return data_->tight_upper_bound;
}

void Assumption::tight_upper_bound(const Expression tight_ub) {
    mut().tight_upper_bound = tight_ub;
}

const ExpressionSet& Assumption::constraints() const {
    return data_->constraints;
}

void Assumption::add_constraint(const Expression c) {
    if (!data_->constraints.contains(c)) {
        mut().constraints.insert(c);
    }
}

bool Assumption::contains_constraint(const Expression c) {
    return data_->constraints.contains(c);
}

bool Assumption::remove_constraint(const Expression c) {
    return data_->constraints.contains(c) && mut().constraints.erase(c) > 0;
}

bool Assumption::constant() const {
    return data_->constant;
};

void Assumption::constant(bool constant) {
    if (data_->constant != constant) {
        mut().constant = constant;
    }
};

const Expression Assumption::map() const {
    return data_->map;
};

void Assumption::map(const Expression map) {
    mut().map = map;
};

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
    auto& d = mut();
    substitute(d.lower_bounds, replacements);
    substitute(d.upper_bounds, replacements);
    if (!d.tight_lower_bound.is_null()) {
        d.tight_lower_bound = d.tight_lower_bound->subs(replacements);
    }
    if (!d.tight_upper_bound.is_null()) {
        d.tight_upper_bound = d.tight_upper_bound->subs(replacements);
    }
    substitute(d.constraints, replacements);
    if (!d.map.is_null()) {
        d.map = d.map->subs(replacements);
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
