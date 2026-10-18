#include "sdfg/transformations/map_delinearize.h"

#include <functional>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <utility>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/structured_control_flow/while.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/visitor/for_each.h"

namespace sdfg {
namespace transformations {

namespace {

/// The digits `(i mod high) div low` of an induction variable i in [0, extent), both bounds at most the extent.
struct Digits {
    int64_t low;
    int64_t high;
};

/// What delinearizing a map needs: the chains of divisions and remainders of its induction variable, and the bounds
/// of their digits, 1 = b_0 < b_1 < ... < b_m = extent, each dividing the next.
struct Plan {
    int64_t extent;
    std::vector<std::pair<symbolic::Expression, Digits>> chains;
    std::vector<int64_t> bounds;
};

/// @p a times @p b, or @p extent if that reaches it: digits from there on are 0.
int64_t times(int64_t a, int64_t b, int64_t extent) {
    return a >= (extent + b - 1) / b ? extent : a * b;
}

std::optional<int64_t> to_int(const SymEngine::Integer& integer) {
    try {
        return integer.as_int();
    } catch (const SymEngine::SymEngineException&) {
        return std::nullopt;
    }
}

/// The digits @p expr selects of @p indvar, if it is the variable or a chain of divisions and remainders of it by
/// positive constants.
std::optional<Digits> digits_of(const symbolic::Expression& expr, const symbolic::Symbol& indvar, int64_t extent) {
    if (symbolic::eq(expr, indvar)) {
        return Digits{1, extent};
    }
    if (!SymEngine::is_a<SymEngine::FunctionSymbol>(*expr)) {
        return std::nullopt;
    }
    auto& function = SymEngine::down_cast<const SymEngine::FunctionSymbol&>(*expr);
    auto name = function.get_name();
    auto args = function.get_args();
    if ((name != "idiv" && name != "imod") || args.size() != 2 || !SymEngine::is_a<SymEngine::Integer>(*args[1])) {
        return std::nullopt;
    }
    auto divisor = to_int(SymEngine::down_cast<const SymEngine::Integer&>(*args[1]));
    if (!divisor || *divisor <= 0) {
        return std::nullopt;
    }
    auto inner = digits_of(args[0], indvar, extent);
    if (!inner) {
        return std::nullopt;
    }
    auto scaled = times(inner->low, *divisor, extent);
    // ((i mod h) div l) div c = (i mod h) div (l*c); ((i mod h) div l) mod c = (i mod min(h, l*c)) div l.
    if (name == "idiv") {
        return Digits{scaled, inner->high};
    }
    return Digits{inner->low, std::min(inner->high, scaled)};
}

void collect(
    const symbolic::Expression& expr,
    const symbolic::Symbol& indvar,
    int64_t extent,
    std::vector<std::pair<symbolic::Expression, Digits>>& chains
) {
    if (!symbolic::uses(expr, indvar)) {
        return;
    }
    if (auto digits = digits_of(expr, indvar, extent)) {
        chains.emplace_back(expr, *digits);
        return;
    }
    for (auto& arg : expr->get_args()) {
        collect(arg, indvar, extent, chains);
    }
}

/// Visits the expressions of memlet subsets, assignments, conditions and loop headers in @p node.
void for_each_expression(
    structured_control_flow::ControlFlowNode& node, const std::function<void(const symbolic::Expression&)>& visit
) {
    if (auto block = dynamic_cast<structured_control_flow::Block*>(&node)) {
        for (auto& memlet : block->dataflow().edges()) {
            for (auto& dim : memlet.subset()) {
                visit(dim);
            }
        }
    } else if (auto assignments = dynamic_cast<structured_control_flow::AssignmentBlock*>(&node)) {
        for (auto& [symbol, value] : assignments->assignments()) {
            visit(value);
        }
    } else if (auto sequence = dynamic_cast<structured_control_flow::Sequence*>(&node)) {
        for (size_t i = 0; i < sequence->size(); ++i) {
            for_each_expression(sequence->at(i), visit);
        }
    } else if (auto if_else = dynamic_cast<structured_control_flow::IfElse*>(&node)) {
        for (size_t i = 0; i < if_else->size(); ++i) {
            auto [branch, condition] = if_else->at(i);
            visit(condition);
            for_each_expression(branch, visit);
        }
    } else if (auto loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&node)) {
        visit(loop->init());
        visit(loop->condition());
        visit(loop->update());
        for_each_expression(loop->root(), visit);
    } else if (auto loop = dynamic_cast<structured_control_flow::While*>(&node)) {
        for_each_expression(loop->root(), visit);
    }
}

std::optional<Plan> plan_of(structured_control_flow::Map& map) {
    if (map.schedule_type().value() != structured_control_flow::ScheduleType_Sequential::value() ||
        !map.is_contiguous() || !symbolic::eq(map.init(), symbolic::zero())) {
        return std::nullopt;
    }
    auto bound = map.canonical_bound();
    if (bound.is_null() || !SymEngine::is_a<SymEngine::Integer>(*bound)) {
        return std::nullopt;
    }
    auto extent = to_int(SymEngine::down_cast<const SymEngine::Integer&>(*bound));
    if (!extent || *extent < 2 || *extent > std::numeric_limits<int64_t>::max() / 2) {
        return std::nullopt;
    }
    Plan plan{*extent, {}, {}};
    auto indvar = map.indvar();
    for_each_expression(map.root(), [&](const symbolic::Expression& expr) {
        collect(expr, indvar, plan.extent, plan.chains);
    });

    std::set<int64_t> bounds{1, plan.extent};
    for (auto& [expr, digits] : plan.chains) {
        // Digits past the extent are 0, whatever the bounds.
        if (digits.low < digits.high) {
            bounds.insert(digits.low);
            bounds.insert(digits.high);
        }
    }
    plan.bounds.assign(bounds.begin(), bounds.end());
    if (plan.bounds.size() < 3) {
        return std::nullopt;
    }
    for (size_t j = 0; j + 1 < plan.bounds.size(); ++j) {
        if (plan.bounds[j + 1] % plan.bounds[j] != 0) {
            return std::nullopt;
        }
    }
    return plan;
}

} // namespace

MapDelinearize::MapDelinearize(structured_control_flow::Map& map) : map_(map) {
}

std::string MapDelinearize::name() const {
    return "MapDelinearize";
}

bool MapDelinearize::can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    return plan_of(map_).has_value();
}

void MapDelinearize::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto plan = *plan_of(map_);
    auto& bounds = plan.bounds;
    auto indvar = map_.indvar();
    auto& parent = static_cast<structured_control_flow::Sequence&>(*map_.get_parent());

    // One map per digit, x_j of weight b_j, outermost first: i = sum_j b_j * x_j.
    size_t count = bounds.size() - 1;
    std::vector<symbolic::Symbol> digits(count);
    for (size_t j = 0; j < count; ++j) {
        auto name = builder.find_new_name(indvar->get_name() + "_d");
        builder.add_container(name, builder.subject().type(indvar->get_name()));
        digits[j] = symbolic::symbol(name);
    }
    structured_control_flow::Sequence* body = nullptr;
    for (size_t j = count; j-- > 0;) {
        auto digit = digits[j];
        auto condition = symbolic::Lt(digit, symbolic::integer(bounds[j + 1] / bounds[j]));
        auto update = symbolic::add(digit, symbolic::one());
        auto& map =
            body == nullptr
                ? builder.add_map_before(
                      parent, map_, digit, condition, symbolic::zero(), update, map_.schedule_type(), map_.debug_info()
                  )
                : builder
                      .add_map(*body, digit, condition, symbolic::zero(), update, map_.schedule_type(), map_.debug_info());
        maps_.push_back(&map);
        body = &map.root();
    }

    auto value = [&](const Digits& range) {
        symbolic::Expression sum = symbolic::zero();
        for (size_t j = 0; j < count; ++j) {
            if (range.low <= bounds[j] && bounds[j] < range.high) {
                sum = symbolic::add(sum, symbolic::mul(symbolic::integer(bounds[j] / range.low), digits[j]));
            }
        }
        return sum;
    };
    // Chains are replaced whole before their parts.
    symbolic::ExpressionMapping replacements;
    for (auto& [expr, range] : plan.chains) {
        replacements[expr] = value(range);
    }
    auto linear = value(Digits{1, plan.extent});
    replacements[indvar] = linear;

    // Tasklets read the induction variable as data, which a replacement cannot express.
    bool read_as_data = false;
    visitor::for_each_block(map_.root(), [&](structured_control_flow::Block& block) {
        for (auto* access : block.dataflow().data_nodes()) {
            read_as_data |= access->data() == indvar->get_name();
        }
    });
    builder.move_children(map_.root(), *body);
    body->replace(replacements);
    if (read_as_data) {
        builder.add_assignments_at(*body, 0, {{indvar, linear}});
    }
    builder.remove_child(parent, parent.index(map_));
    analysis_manager.invalidate_all();
}

void MapDelinearize::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["parameters"] = nlohmann::json::object();
    serializer::JSONSerializer serializer(false);
    j["subgraph"] = nlohmann::json::object();
    j["subgraph"]["0"] = nlohmann::json::object();
    serializer.serialize_node(j["subgraph"]["0"], map_);
}

MapDelinearize MapDelinearize::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc) {
    auto map_id = desc["subgraph"]["0"]["element_id"].get<size_t>();
    auto map = dynamic_cast<structured_control_flow::Map*>(builder.find_element_by_id(map_id));
    if (!map) {
        throw InvalidTransformationDescriptionException("Map with ID " + std::to_string(map_id) + " not found.");
    }
    return MapDelinearize(*map);
}

const std::vector<structured_control_flow::Map*>& MapDelinearize::maps() const {
    return maps_;
}

} // namespace transformations
} // namespace sdfg
