#include "sdfg/transformations/index_set_split.h"

#include <functional>
#include <optional>
#include <set>
#include <string>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/while.h"
#include "sdfg/symbolic/polynomials.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/transformations/loop_split.h"
#include "sdfg/visitor/for_each.h"

namespace sdfg {
namespace transformations {

namespace {

enum class Truth { False, True, Unknown };

std::optional<int64_t> constant(const symbolic::Expression& expr) {
    auto expanded = symbolic::expand(expr);
    if (!SymEngine::is_a<SymEngine::Integer>(*expanded)) {
        return std::nullopt;
    }
    try {
        return SymEngine::down_cast<const SymEngine::Integer&>(*expanded).as_int();
    } catch (const SymEngine::SymEngineException&) {
        return std::nullopt;
    }
}

/// The iterations [first, end) of @p loop, if it steps by 1 between constants.
std::optional<std::pair<int64_t, int64_t>> range_of(structured_control_flow::StructuredLoop& loop) {
    if (!loop.is_contiguous()) {
        return std::nullopt;
    }
    auto first = constant(loop.init());
    auto bound = loop.canonical_bound();
    auto end = bound.is_null() ? std::nullopt : constant(bound);
    if (!first || !end || *first >= *end) {
        return std::nullopt;
    }
    return std::make_pair(*first, *end);
}

std::vector<symbolic::Condition> conjuncts(const symbolic::Condition& condition) {
    if (SymEngine::is_a<SymEngine::And>(*condition)) {
        auto& args = SymEngine::down_cast<const SymEngine::And&>(*condition).get_container();
        return {args.begin(), args.end()};
    }
    return {condition};
}

/// Whether @p atom compares expressions affine in @p indvar alone, and so is monotone in it.
bool is_monotone(const symbolic::Condition& atom, const symbolic::Symbol& indvar) {
    if (!SymEngine::is_a<SymEngine::StrictLessThan>(*atom) && !SymEngine::is_a<SymEngine::LessThan>(*atom)) {
        return false;
    }
    auto& relational = SymEngine::down_cast<const SymEngine::Relational&>(*atom);
    auto difference = symbolic::expand(symbolic::sub(relational.get_arg1(), relational.get_arg2()));
    auto atoms = symbolic::atoms(difference);
    return atoms.size() == 1 && symbolic::eq(*atoms.begin(), indvar) && symbolic::is_integer_affine(difference);
}

bool holds(const symbolic::Condition& atom, const symbolic::Symbol& indvar, int64_t value) {
    return symbolic::is_true(symbolic::subs(atom, indvar, symbolic::integer(value)));
}

/// The first iteration in [first, end) at which monotone @p atom changes, if it does.
std::optional<int64_t>
change_point(const symbolic::Condition& atom, const symbolic::Symbol& indvar, int64_t first, int64_t end) {
    bool initially = holds(atom, indvar, first);
    if (holds(atom, indvar, end - 1) == initially) {
        return std::nullopt;
    }
    auto low = first;
    auto high = end - 1;
    while (high - low > 1) {
        auto middle = low + (high - low) / 2;
        (holds(atom, indvar, middle) == initially ? low : high) = middle;
    }
    return high;
}

/// Whether @p condition holds on all, none or some of the iterations [first, end).
Truth truth(const symbolic::Condition& condition, const symbolic::Symbol& indvar, int64_t first, int64_t end) {
    if (symbolic::is_true(condition)) {
        return Truth::True;
    }
    if (symbolic::is_false(condition)) {
        return Truth::False;
    }
    bool always = true;
    for (auto& atom : conjuncts(condition)) {
        if (!is_monotone(atom, indvar)) {
            always = false;
            continue;
        }
        bool at_first = holds(atom, indvar, first);
        if (at_first != holds(atom, indvar, end - 1)) {
            always = false;
        } else if (!at_first) {
            return Truth::False;
        }
    }
    return always ? Truth::True : Truth::Unknown;
}

void for_each_if_else(
    structured_control_flow::ControlFlowNode& node, const std::function<void(structured_control_flow::IfElse&)>& visit
) {
    if (auto sequence = dynamic_cast<structured_control_flow::Sequence*>(&node)) {
        for (size_t i = 0; i < sequence->size(); ++i) {
            for_each_if_else(sequence->at(i), visit);
        }
    } else if (auto if_else = dynamic_cast<structured_control_flow::IfElse*>(&node)) {
        visit(*if_else);
        for (size_t i = 0; i < if_else->size(); ++i) {
            for_each_if_else(if_else->at(i).first, visit);
        }
    } else if (auto loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&node)) {
        for_each_if_else(loop->root(), visit);
    } else if (auto loop = dynamic_cast<structured_control_flow::While*>(&node)) {
        for_each_if_else(loop->root(), visit);
    }
}

std::set<int64_t> split_points(structured_control_flow::StructuredLoop& loop) {
    std::set<int64_t> points;
    auto range = range_of(loop);
    if (!range) {
        return points;
    }
    auto indvar = loop.indvar();
    for_each_if_else(loop.root(), [&](structured_control_flow::IfElse& if_else) {
        for (size_t i = 0; i < if_else.size(); ++i) {
            for (auto& atom : conjuncts(if_else.at(i).second)) {
                if (!is_monotone(atom, indvar)) {
                    continue;
                }
                if (auto point = change_point(atom, indvar, range->first, range->second)) {
                    points.insert(*point);
                }
            }
        }
    });
    return points;
}

/// Drops the branches of @p if_else that never run on [first, end) and inlines one that always does; false if none.
bool fold(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::IfElse& if_else,
    const symbolic::Symbol& indvar,
    int64_t first,
    int64_t end
) {
    std::vector<size_t> dropped;
    std::optional<size_t> taken;
    bool undecided = false;
    for (size_t i = 0; i < if_else.size(); ++i) {
        if (taken) {
            dropped.push_back(i);
            continue;
        }
        switch (truth(if_else.at(i).second, indvar, first, end)) {
            case Truth::False:
                dropped.push_back(i);
                break;
            case Truth::True:
                taken = i;
                break;
            case Truth::Unknown:
                undecided = true;
                break;
        }
    }
    bool inline_taken = taken && !undecided;
    if (dropped.empty() && !inline_taken) {
        return false;
    }
    auto& parent = static_cast<structured_control_flow::Sequence&>(*if_else.get_parent());
    if (inline_taken) {
        builder.move_children(if_else.at(*taken).first, parent, parent.index(if_else));
        builder.remove_child(parent, parent.index(if_else));
        return true;
    }
    for (auto i = dropped.rbegin(); i != dropped.rend(); ++i) {
        builder.remove_case(if_else, *i);
    }
    if (if_else.size() == 0) {
        builder.remove_child(parent, parent.index(if_else));
    }
    return true;
}

void fold_all(
    builder::StructuredSDFGBuilder& builder, structured_control_flow::StructuredLoop& loop, int64_t first, int64_t end
) {
    for (bool changed = true; changed;) {
        changed = false;
        std::vector<structured_control_flow::IfElse*> branches;
        for_each_if_else(loop.root(), [&](structured_control_flow::IfElse& if_else) {
            branches.push_back(&if_else);
        });
        // Folding removes or moves nodes, so look again after each change.
        for (auto* if_else : branches) {
            if (fold(builder, *if_else, loop.indvar(), first, end)) {
                changed = true;
                break;
            }
        }
    }
}

} // namespace

IndexSetSplit::IndexSetSplit(structured_control_flow::StructuredLoop& loop) : loop_(loop) {
}

std::string IndexSetSplit::name() const {
    return "IndexSetSplit";
}

bool IndexSetSplit::can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    return !split_points(loop_).empty() && dynamic_cast<structured_control_flow::Sequence*>(loop_.get_parent());
}

void IndexSetSplit::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto points = split_points(loop_);
    auto [first, end] = *range_of(loop_);
    auto& parent = static_cast<structured_control_flow::Sequence&>(*loop_.get_parent());
    for (auto point : points) {
        LoopSplit split(loop_, symbolic::integer(point));
        split.apply(builder, analysis_manager);
    }
    std::vector<int64_t> bounds{first};
    bounds.insert(bounds.end(), points.begin(), points.end());
    bounds.push_back(end);
    size_t start = parent.index(loop_) - points.size();
    for (size_t k = 0; k + 1 < bounds.size(); ++k) {
        auto& part = static_cast<structured_control_flow::StructuredLoop&>(parent.at(start + k));
        auto indvar = part.indvar();
        builder
            .update_loop(part, indvar, symbolic::Lt(indvar, symbolic::integer(bounds[k + 1])), part.init(), part.update());
        fold_all(builder, part, bounds[k], bounds[k + 1]);
        // Maps, whose induction variables end with them, count from zero as in loop normal form.
        bool read_as_data = false;
        visitor::for_each_block(part.root(), [&](structured_control_flow::Block& block) {
            for (auto* access : block.dataflow().data_nodes()) {
                read_as_data |= access->data() == indvar->get_name();
            }
        });
        if (bounds[k] != 0 && dynamic_cast<structured_control_flow::Map*>(&part) && !read_as_data) {
            part.root().replace(indvar, symbolic::add(indvar, symbolic::integer(bounds[k])));
            builder.update_loop(
                part,
                indvar,
                symbolic::Lt(indvar, symbolic::integer(bounds[k + 1] - bounds[k])),
                symbolic::zero(),
                part.update()
            );
        }
        loops_.push_back(&part);
    }
    analysis_manager.invalidate_all();
}

void IndexSetSplit::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["parameters"] = nlohmann::json::object();
    serializer::JSONSerializer serializer(false);
    j["subgraph"] = nlohmann::json::object();
    j["subgraph"]["0"] = nlohmann::json::object();
    serializer.serialize_node(j["subgraph"]["0"], loop_);
}

IndexSetSplit IndexSetSplit::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc) {
    auto loop_id = desc["subgraph"]["0"]["element_id"].get<size_t>();
    auto loop = dynamic_cast<structured_control_flow::StructuredLoop*>(builder.find_element_by_id(loop_id));
    if (!loop) {
        throw InvalidTransformationDescriptionException("Loop with ID " + std::to_string(loop_id) + " not found.");
    }
    return IndexSetSplit(*loop);
}

const std::vector<structured_control_flow::StructuredLoop*>& IndexSetSplit::loops() const {
    return loops_;
}

} // namespace transformations
} // namespace sdfg
