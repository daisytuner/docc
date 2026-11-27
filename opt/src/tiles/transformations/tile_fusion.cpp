#include "sdfg/tiles/transformations/tile_fusion.h"

#include <climits>
#include <cmath>
#include <isl/aff.h>
#include <isl/ctx.h>
#include <isl/map.h>
#include <isl/options.h>
#include <isl/set.h>
#include <isl/val.h>
#include <optional>
#include <set>
#include <stdexcept>

#include <symengine/solve.h>

#include "sdfg/analysis/arguments_analysis.h"
#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/analysis/memory_layout_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/pointer_metadata.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/symbolic/conjunctive_normal_form.h"
#include "sdfg/symbolic/delinearization.h"
#include "sdfg/symbolic/maps.h"
#include "sdfg/symbolic/polynomials.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/symbolic/utils.h"
#include "sdfg/tiles/analysis/tile_analysis.h"
#include "sdfg/types/array.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"

namespace sdfg {
namespace transformations {

namespace {

/// Collect all Block nodes reachable from a Sequence, including through nested For/Map loops.
void collect_blocks(structured_control_flow::Sequence& seq, std::vector<structured_control_flow::Block*>& blocks) {
    for (size_t i = 0; i < seq.size(); ++i) {
        auto& child = seq.at(i);
        if (auto* block = dyn_cast<structured_control_flow::Block*>(&child)) {
            blocks.push_back(block);
        } else if (auto* loop = dyn_cast<structured_control_flow::StructuredLoop*>(&child)) {
            collect_blocks(loop->root(), blocks);
        }
    }
}

/// Get the first block from a sequence, recursively descending into loops.
structured_control_flow::Block* get_first_block(structured_control_flow::Sequence& seq) {
    for (size_t i = 0; i < seq.size(); ++i) {
        auto& child = seq.at(i);
        if (auto* block = dyn_cast<structured_control_flow::Block*>(&child)) {
            return block;
        } else if (auto* loop = dyn_cast<structured_control_flow::StructuredLoop*>(&child)) {
            auto* result = get_first_block(loop->root());
            if (result) {
                return result;
            }
        }
    }
    return nullptr;
}

/// One access of a container in a loop's body: the elements it may touch, as an expression in the counters of the
/// loops around it and of the access's footprint, which @ref assumptions bound; none if it may touch any element.
struct Access {
    std::string container;
    bool write = false;
    std::optional<symbolic::MultiExpression> elements;
    symbolic::Assumptions assumptions;
};

/// @p assumptions with @p from replaced by @p to in their bounds, and renamed if @p to is a symbol, else dropped.
symbolic::Assumptions
renamed(const symbolic::Assumptions& assumptions, const symbolic::Symbol& from, const symbolic::Expression& to) {
    auto to_symbol = SymEngine::is_a<SymEngine::Symbol>(*to) ? SymEngine::rcp_static_cast<const SymEngine::Symbol>(to)
                                                             : symbolic::Symbol();
    symbolic::Assumptions result;
    for (auto& [symbol, assumption] : assumptions) {
        // `to` names the renamed symbol now; its own assumptions, if any, belong to another scope.
        if (!to_symbol.is_null() && !symbolic::eq(from, to_symbol) && symbolic::eq(symbol, to_symbol)) {
            continue;
        }
        if (to_symbol.is_null() && symbolic::eq(symbol, from)) {
            continue;
        }
        auto key = symbolic::eq(symbol, from) ? to_symbol : symbol;
        symbolic::Assumption copy(key);
        auto subs = [&](const symbolic::Expression& expression) {
            return expression.is_null() ? expression : symbolic::subs(expression, from, to);
        };
        for (auto& bound : assumption.lower_bounds()) {
            copy.add_lower_bound(subs(bound));
        }
        for (auto& bound : assumption.upper_bounds()) {
            copy.add_upper_bound(subs(bound));
        }
        for (auto& constraint : assumption.constraints()) {
            copy.add_constraint(subs(constraint));
        }
        if (!assumption.tight_lower_bound().is_null()) {
            copy.tight_lower_bound(subs(assumption.tight_lower_bound()));
        }
        if (!assumption.tight_upper_bound().is_null()) {
            copy.tight_upper_bound(subs(assumption.tight_upper_bound()));
        }
        if (!assumption.map().is_null()) {
            copy.map(subs(assumption.map()));
        }
        copy.constant(assumption.constant());
        result.insert_or_assign(key, std::move(copy));
    }
    return result;
}

/// The elements @p layout lays out, as an expression over fresh counters bounded in @p assumptions.
symbolic::MultiExpression
footprint(const math::tensor::TensorLayout& layout, symbolic::Assumptions& assumptions, size_t& counters) {
    auto element = layout.offset();
    for (size_t d = 0; d < layout.shape().size(); ++d) {
        auto counter = symbolic::symbol("__tile_fusion_element" + std::to_string(counters++));
        auto assumption = symbolic::Assumption::create(counter, types::Scalar(types::PrimitiveType::Int64));
        auto last = symbolic::sub(layout.shape().at(d), symbolic::one());
        assumption.add_lower_bound(symbolic::zero());
        assumption.add_upper_bound(last);
        assumption.tight_lower_bound(symbolic::zero());
        assumption.tight_upper_bound(last);
        assumption.constant(false);
        assumptions.insert_or_assign(counter, std::move(assumption));
        element = symbolic::add(element, symbolic::mul(counter, layout.strides().at(d)));
    }
    return symbolic::MultiExpression{element};
}

/**
 * Collects every access of a container in @p node into @p accesses, with @p from renamed to @p to and the counters
 * @p varying varying. Memlets access the elements their subsets give; a library node accesses through a pointer the
 * elements analysis::library_node_operand_layout lays out, and any element if it gives none; references and other
 * memlets may access any element of both containers.
 * @return false if @p node holds control flow the accesses of which cannot be bounded
 */
bool collect_accesses(
    structured_control_flow::ControlFlowNode& node,
    analysis::AssumptionsAnalysis& assumptions_analysis,
    const symbolic::SymbolSet& varying,
    const symbolic::Symbol& from,
    const symbolic::Symbol& to,
    std::vector<Access>& accesses,
    size_t& counters
) {
    auto rename = [&](const symbolic::Expression& expression) {
        return symbolic::subs(expression, from, to);
    };
    if (auto* block = dyn_cast<structured_control_flow::Block*>(&node)) {
        // The counters of the loop and of the loops within vary between the accesses compared; the block's
        // assumptions hold them constant.
        auto scope = assumptions_analysis.get(*block, true);
        for (auto& counter : varying) {
            if (scope.contains(counter)) {
                scope.at(counter).constant(false);
            }
        }
        auto assumptions = renamed(scope, from, to);
        auto add = [&](const std::string& container, bool write, std::optional<symbolic::MultiExpression> elements) {
            Access access{container, write, std::nullopt, assumptions};
            if (elements) {
                symbolic::MultiExpression renamed_elements;
                for (auto& element : *elements) {
                    renamed_elements.push_back(rename(element));
                }
                access.elements = std::move(renamed_elements);
            }
            accesses.push_back(std::move(access));
        };
        auto& dataflow = block->dataflow();
        for (auto& memlet : dataflow.edges()) {
            auto* src = dynamic_cast<const data_flow::AccessNode*>(&memlet.src());
            auto* dst = dynamic_cast<const data_flow::AccessNode*>(&memlet.dst());
            if (dynamic_cast<const data_flow::ConstantNode*>(src)) {
                src = nullptr;
            }
            if (dynamic_cast<const data_flow::ConstantNode*>(dst)) {
                dst = nullptr;
            }
            bool computational = memlet.type() == data_flow::MemletType::Computational;
            if (src && dst) {
                add(src->data(), !computational, std::nullopt);
                add(dst->data(), true, std::nullopt);
                continue;
            }
            if (src) {
                auto* library = dynamic_cast<const data_flow::LibraryNode*>(&memlet.dst());
                auto meta = library && memlet.subset().empty() ? library->pointer_access_type(memlet) : nullptr;
                if (meta) {
                    auto operand = [&](bool write) {
                        auto pattern = write ? meta->access_write_pattern() : meta->access_read_pattern();
                        if (!(write ? meta->may_contain_writes() : meta->may_contain_reads()) ||
                            (pattern && pattern->empty())) {
                            return;
                        }
                        Access access{src->data(), write, std::nullopt, assumptions};
                        auto layout = analysis::library_node_operand_layout(
                            memlet, write ? analysis::OperandAccess::Write : analysis::OperandAccess::Read
                        );
                        if (layout) {
                            access.elements = symbolic::MultiExpression{
                                rename(footprint(*layout, access.assumptions, counters).front())
                            };
                        }
                        accesses.push_back(std::move(access));
                    };
                    operand(false);
                    operand(true);
                    continue;
                }
                if (computational) {
                    add(src->data(), false, memlet.subset());
                } else {
                    add(src->data(), true, std::nullopt);
                }
            }
            if (dst) {
                add(dst->data(),
                    true,
                    computational ? std::optional<symbolic::MultiExpression>(memlet.subset()) : std::nullopt);
            }
        }
        return true;
    }
    if (auto* sequence = dyn_cast<structured_control_flow::Sequence*>(&node)) {
        for (size_t i = 0; i < sequence->size(); ++i) {
            if (!collect_accesses(sequence->at(i), assumptions_analysis, varying, from, to, accesses, counters)) {
                return false;
            }
        }
        return true;
    }
    if (auto* loop = dyn_cast<structured_control_flow::StructuredLoop*>(&node)) {
        return collect_accesses(loop->root(), assumptions_analysis, varying, from, to, accesses, counters);
    }
    if (auto* branches = dyn_cast<structured_control_flow::IfElse*>(&node)) {
        for (size_t i = 0; i < branches->size(); ++i) {
            if (!collect_accesses(branches->at(i).first, assumptions_analysis, varying, from, to, accesses, counters)) {
                return false;
            }
        }
        return true;
    }
    return false;
}

/// @p access with its loop's counter @p from replaced by @p to, an expression in the loop's iteration number, which
/// @p iteration bounds.
Access at_iteration(
    const Access& access,
    const symbolic::Symbol& from,
    const symbolic::Expression& to,
    const symbolic::Assumption& iteration
) {
    Access result{access.container, access.write, std::nullopt, renamed(access.assumptions, from, to)};
    if (access.elements) {
        symbolic::MultiExpression elements;
        for (auto& element : *access.elements) {
            elements.push_back(symbolic::subs(element, from, to));
        }
        result.elements = std::move(elements);
    }
    result.assumptions.insert_or_assign(iteration.symbol(), iteration);
    return result;
}

/// @p assumptions with an assumption for each symbol only @p other bounds, so that both bound every symbol of a pair.
symbolic::Assumptions completed(symbolic::Assumptions assumptions, const symbolic::Assumptions& other) {
    for (auto& [symbol, assumption] : other) {
        if (!assumptions.contains(symbol)) {
            assumptions.insert_or_assign(symbol, assumption);
        }
    }
    return assumptions;
}

/// Whether @p first, in an iteration of the loop counting @p counter, and @p second, in a later one, may access the
/// same element.
bool later_overlap(const Access& first, const Access& second, const symbolic::Symbol& counter) {
    if (!first.elements || !second.elements) {
        return true;
    }
    auto deltas = symbolic::maps::dependence_deltas(
        *first.elements,
        *second.elements,
        counter,
        completed(first.assumptions, second.assumptions),
        completed(second.assumptions, first.assumptions)
    );
    return !deltas.empty;
}

/**
 * The most iterations of the loop counting @p point by which an element @p consumer reads lies past the iteration of
 * @p producer writing it, at least 0: the halo by which the producer must run ahead. None if unbounded.
 */
std::optional<int64_t>
halo(const Access& producer, const Access& consumer, const symbolic::Symbol& point, const symbolic::Symbol& counter) {
    if (!producer.elements || !consumer.elements || producer.elements->size() != consumer.elements->size()) {
        return std::nullopt;
    }
    auto producer_assumptions = completed(producer.assumptions, consumer.assumptions);
    auto consumer_assumptions = completed(consumer.assumptions, producer.assumptions);
    auto [map_1, map_2, map_3] = symbolic::expressions_to_intersection_map_str(
        *producer.elements, *consumer.elements, counter, producer_assumptions, consumer_assumptions
    );
    auto* ctx = isl_ctx_alloc();
    isl_options_set_on_error(ctx, ISL_ON_ERROR_CONTINUE);
    std::optional<int64_t> result;
    isl_map* writes = isl_map_read_from_str(ctx, map_1.c_str());
    isl_map* reads = isl_map_read_from_str(ctx, map_2.c_str());
    if (writes && reads) {
        auto dimension = isl_map_find_dim_by_name(writes, isl_dim_in, (point->get_name() + "_1").c_str());
        // { producer iteration -> consumer iteration : the consumer reads what the producer writes }
        isl_map* pairs = isl_map_apply_range(writes, isl_map_reverse(reads));
        writes = nullptr;
        reads = nullptr;
        if (dimension < 0) {
            isl_map_free(pairs);
        } else if (isl_map_is_empty(pairs) == isl_bool_true) {
            isl_map_free(pairs);
            result = 0;
        } else {
            // Consumer minus producer iteration per dimension; the halo is the producer's lead, -min.
            // The least difference must be one constant, independent of the parameters and finite.
            isl_pw_aff* least = isl_set_dim_min(isl_map_deltas(pairs), dimension);
            struct Piece {
                int count = 0;
                std::optional<int64_t> value;
            } piece;
            isl_pw_aff_foreach_piece(
                least,
                [](isl_set* domain, isl_aff* aff, void* user) -> isl_stat {
                    auto* piece = static_cast<Piece*>(user);
                    ++piece->count;
                    if (isl_aff_is_cst(aff) == isl_bool_true && isl_aff_is_nan(aff) == isl_bool_false) {
                        isl_val* constant = isl_aff_get_constant_val(aff);
                        if (isl_val_is_int(constant) == isl_bool_true) {
                            piece->value = isl_val_get_num_si(constant);
                        }
                        isl_val_free(constant);
                    }
                    isl_set_free(domain);
                    isl_aff_free(aff);
                    return isl_stat_ok;
                },
                &piece
            );
            isl_pw_aff_free(least);
            if (piece.count == 1 && piece.value) {
                result = std::max<int64_t>(0, -*piece.value);
            }
        }
    }
    isl_map_free(writes);
    isl_map_free(reads);
    isl_ctx_free(ctx);
    return result;
}

} // namespace

TileFusion::TileFusion(structured_control_flow::Map& first_map, structured_control_flow::Map& second_map)
    : first_map_(first_map), second_map_(second_map) {
}

std::string TileFusion::name() const {
    return "TileFusion";
}

bool TileFusion::can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    candidates_.clear();
    cyclic_containers_.clear();
    radius_ = 0;
    sequential_ = false;

    // Get analyses upfront
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();

    // ==========================================================================
    // Criterion 1: Both maps are in the same parent sequence, consecutive
    // ==========================================================================
    auto* first_parent = first_map_.get_parent();
    auto* second_parent = second_map_.get_parent();
    if (first_parent == nullptr || second_parent == nullptr) {
        return false;
    }
    if (first_parent != second_parent) {
        return false;
    }

    auto* parent_sequence = dyn_cast<structured_control_flow::Sequence*>(first_parent);
    if (parent_sequence == nullptr) {
        return false;
    }

    int first_index = parent_sequence->index(first_map_);
    int second_index = parent_sequence->index(second_map_);
    if (first_index == -1 || second_index == -1) {
        return false;
    }
    if (second_index != first_index + 1) {
        return false;
    }

    // ==========================================================================
    // Criterion 2: Both run as many iterations under the same schedule
    // The fused loop runs the n-th iteration of the first map, then the n-th of the second.
    // ==========================================================================
    auto first_counter = first_map_.indvar();
    auto second_counter = second_map_.indvar();
    auto positive_stride = [](structured_control_flow::StructuredLoop& loop) -> int64_t {
        auto stride = loop.stride();
        if (stride.is_null() || !SymEngine::is_a<SymEngine::Integer>(*stride) || loop.canonical_bound().is_null()) {
            return 0;
        }
        return std::max<int64_t>(SymEngine::down_cast<const SymEngine::Integer&>(*stride).as_int(), 0);
    };
    // A loop within that counts with the same name hides the counter: the body is no function of the iteration.
    for (auto* loop : {&first_map_, &second_map_}) {
        for (auto* inner : loop_analysis.descendants(loop)) {
            auto* structured = dynamic_cast<const structured_control_flow::StructuredLoop*>(inner);
            if (structured && symbolic::eq(structured->indvar(), loop->indvar())) {
                return false;
            }
        }
    }
    auto first_step = positive_stride(first_map_);
    auto second_step = positive_stride(second_map_);
    if (first_step == 0 || second_step == 0) {
        return false;
    }
    auto iterations = first_map_.num_iterations();
    if (iterations.is_null() || second_map_.num_iterations().is_null() ||
        !symbolic::eq(iterations, second_map_.num_iterations())) {
        return false;
    }
    int tile_size = static_cast<int>(first_step);
    // Maps over the same iterations, under the same counter name.
    bool identical = symbolic::eq(first_map_.init(), second_map_.init()) && first_step == second_step &&
                     symbolic::
                         eq(first_map_.canonical_bound(),
                            symbolic::subs(second_map_.canonical_bound(), second_counter, first_counter));
    if (first_map_.schedule_type().value() != second_map_.schedule_type().value()) {
        return false;
    }

    // ==========================================================================
    // Criterion 3: No dependence runs from an iteration of the second map to a later one of the first
    // Fused, each iteration runs the first map's body, then the second's: every pair of accesses to the same
    // container, one a write, must not touch the same element with the second map's iteration earlier. If they
    // touch it with the first map's iteration earlier, the fused loop is sequential; otherwise it stays a map.
    // ==========================================================================
    auto counters_of = [&](structured_control_flow::StructuredLoop& loop) {
        symbolic::SymbolSet varying{loop.indvar()};
        for (auto* inner : loop_analysis.descendants(&loop)) {
            if (auto* structured = dynamic_cast<const structured_control_flow::StructuredLoop*>(inner)) {
                varying.insert(structured->indvar());
            }
        }
        return varying;
    };
    auto first_varying = counters_of(first_map_);
    auto second_varying = counters_of(second_map_);
    std::vector<Access> first_accesses;
    std::vector<Access> second_accesses;
    size_t counters = 0;
    if (!collect_accesses(
            first_map_.root(), assumptions_analysis, first_varying, first_counter, first_counter, first_accesses, counters
        ) ||
        !collect_accesses(
            second_map_.root(),
            assumptions_analysis,
            second_varying,
            second_counter,
            second_counter,
            second_accesses,
            counters
        )) {
        return false;
    }
    // Accesses through another name of a container's memory escape the comparison: its pointer must not escape, be
    // overwritten or be captured in either map.
    std::set<std::string> containers;
    for (auto* accesses : {&first_accesses, &second_accesses}) {
        for (auto& access : *accesses) {
            containers.insert(access.container);
        }
    }
    for (auto& container : containers) {
        for (auto* loop : {&first_map_, &second_map_}) {
            if (tiles::TileAnalysis::summarize(builder.subject(), *loop, container).aliased) {
                return false;
            }
        }
    }

    bool forward = false;
    bool reversed_flow = false;
    std::set<std::string> reversed_containers;
    std::set<std::string> first_writes, first_reads, second_writes;
    for (auto& access : first_accesses) {
        (access.write ? first_writes : first_reads).insert(access.container);
    }
    for (auto& access : second_accesses) {
        if (access.write) {
            second_writes.insert(access.container);
        }
    }
    // Both maps' accesses at the iteration number k in [0, iterations - 1].
    auto k = symbolic::symbol("__tile_fusion_iteration");
    auto iteration = symbolic::Assumption::create(k, types::Scalar(types::PrimitiveType::Int64));
    auto last = symbolic::sub(iterations, symbolic::one());
    iteration.add_lower_bound(symbolic::zero());
    iteration.add_upper_bound(last);
    iteration.tight_lower_bound(symbolic::zero());
    iteration.tight_upper_bound(last);
    iteration.map(symbolic::add(k, symbolic::one()));
    iteration.constant(false);
    auto at = [&](const Access& access, structured_control_flow::StructuredLoop& loop, int64_t step) {
        return at_iteration(
            access, loop.indvar(), symbolic::add(loop.init(), symbolic::mul(symbolic::integer(step), k)), iteration
        );
    };
    for (auto& first_access : first_accesses) {
        auto first = at(first_access, first_map_, first_step);
        for (auto& second_access : second_accesses) {
            if (first_access.container != second_access.container || (!first_access.write && !second_access.write)) {
                continue;
            }
            auto second = at(second_access, second_map_, second_step);
            forward |= later_overlap(first, second, k);
            if (later_overlap(second, first, k)) {
                reversed_containers.insert(first.container);
                reversed_flow |= first.write && !second.write;
            }
        }
    }
    if (reversed_containers.empty()) {
        sequential_ = forward;
        return true;
    }

    // ==========================================================================
    // Criterion 4: A halo resolves the reversed dependences
    // Each tile's producer iterations run ahead by a radius, recomputing what the consumer reads of the next tiles.
    // This needs the stencil structure (a tile map over one point map each), a producer whose recomputed iterations
    // write what the consumer does not, and inputs the consumer writes double-buffered (Criterion 8).
    // ==========================================================================
    if (!reversed_flow || !identical) {
        return false;
    }
    for (auto& container : second_writes) {
        if (first_writes.contains(container)) {
            return false;
        }
    }
    if (first_map_.root().size() != 1 || second_map_.root().size() != 1) {
        return false;
    }
    auto* first_inner_map = dyn_cast<structured_control_flow::Map*>(&first_map_.root().at(0));
    auto* second_inner_map = dyn_cast<structured_control_flow::Map*>(&second_map_.root().at(0));
    if (first_inner_map == nullptr || second_inner_map == nullptr) {
        return false;
    }
    auto first_loop_info = loop_analysis.loop_info(&first_map_);
    auto second_loop_info = loop_analysis.loop_info(&second_map_);
    if (!first_loop_info.is_perfectly_nested || !second_loop_info.is_perfectly_nested ||
        !first_loop_info.is_perfectly_parallel || !second_loop_info.is_perfectly_parallel) {
        return false;
    }
    std::vector<structured_control_flow::Block*> producer_blocks;
    collect_blocks(first_inner_map->root(), producer_blocks);

    // The consumer's point counter takes the producer's name, so that their difference is one dimension.
    auto point = first_inner_map->indvar();
    std::vector<Access> consumer_reads;
    {
        std::vector<Access> renamed_second;
        size_t renamed_counters = counters;
        std::vector<Access> point_renamed;
        if (!collect_accesses(
                second_map_.root(),
                assumptions_analysis,
                second_varying,
                second_counter,
                first_counter,
                renamed_second,
                renamed_counters
            )) {
            return false;
        }
        for (auto& access : renamed_second) {
            if (access.write) {
                continue;
            }
            Access copy{
                access.container, false, std::nullopt, renamed(access.assumptions, second_inner_map->indvar(), point)
            };
            if (access.elements) {
                symbolic::MultiExpression elements;
                for (auto& element : *access.elements) {
                    elements.push_back(symbolic::subs(element, second_inner_map->indvar(), point));
                }
                copy.elements = std::move(elements);
            }
            consumer_reads.push_back(std::move(copy));
        }
    }
    int overall_radius = 0;
    for (auto& producer : first_accesses) {
        if (!producer.write || !reversed_containers.contains(producer.container)) {
            continue;
        }
        for (auto& consumer : consumer_reads) {
            if (consumer.container != producer.container) {
                continue;
            }
            auto radius = halo(producer, consumer, point, first_counter);
            if (!radius || *radius >= tile_size) {
                return false;
            }
            overall_radius = std::max(overall_radius, static_cast<int>(*radius));
            candidates_.push_back({producer.container, static_cast<int>(*radius)});
        }
    }
    if (overall_radius == 0) {
        return false;
    }
    radius_ = overall_radius;
    sequential_ = true;

    // The inputs of the producer the consumer writes, which a halo would otherwise read updated.
    std::unordered_set<std::string> cyclic_container_names;
    for (auto& container : second_writes) {
        if (first_reads.contains(container)) {
            cyclic_container_names.insert(container);
        }
    }

    // ==========================================================================
    // Criterion 8: Analyze cyclic containers for double buffering
    // Use MemoryLayoutAnalysis to delinearize accesses and reason about
    // multi-dimensional stencil patterns (e.g., A[i*N+j] → A[i][j]).
    // For each cyclic container, identify the tiled dimension, extract
    // stencil offsets in that dimension, and compute buffer geometry.
    // Only needed when radius > 0 (tiles overlap).
    // ==========================================================================
    cyclic_containers_.clear();

    if (overall_radius > 0) {
        auto& mla = analysis_manager.get<analysis::MemoryLayoutAnalysis>();

        for (const auto& container : cyclic_container_names) {
            // Collect all K1 read memlets for this container
            struct ReadMemletInfo {
                const data_flow::Memlet* memlet;
            };
            std::vector<ReadMemletInfo> read_memlets;
            for (auto* block : producer_blocks) {
                auto& dataflow = block->dataflow();
                for (auto& node : dataflow.nodes()) {
                    auto* access = dynamic_cast<data_flow::AccessNode*>(&node);
                    if (access == nullptr || access->data() != container) {
                        continue;
                    }
                    if (dataflow.out_degree(*access) > 0) {
                        for (auto& memlet : dataflow.out_edges(*access)) {
                            if (memlet.type() != data_flow::MemletType::Computational) {
                                continue;
                            }
                            read_memlets.push_back({&memlet});
                        }
                    }
                }
            }
            if (read_memlets.empty()) {
                return false;
            }

            auto inner_indvar = first_inner_map->indvar();

            int min_offset = std::numeric_limits<int>::max();
            int max_offset = std::numeric_limits<int>::min();
            int tiled_dim = -1;
            std::vector<symbolic::Expression> layout_dimensions;
            std::vector<symbolic::Expression> layout_strides;
            symbolic::Expression layout_offset = symbolic::integer(0);

            for (auto& info : read_memlets) {
                // Use MemoryLayoutAnalysis to get delinearized access
                auto* mem_access = mla.access(*info.memlet);
                if (mem_access == nullptr) {
                    return false; // Delinearization failed
                }

                auto& delin_subset = mem_access->min_subset;

                // Find which dimension contains the inner (tiled) indvar
                for (size_t d = 0; d < delin_subset.size(); d++) {
                    if (symbolic::uses(delin_subset[d], inner_indvar->get_name())) {
                        if (tiled_dim == -1) {
                            tiled_dim = static_cast<int>(d);
                        } else if (tiled_dim != static_cast<int>(d)) {
                            return false; // Indvar appears in multiple dimensions
                        }

                        // Decompose this dimension: should be coeff*indvar + offset
                        auto decomp = symbolic::affine_decomposition(delin_subset[d], inner_indvar);
                        if (!decomp.success) {
                            return false;
                        }
                        auto coeff_int = SymEngine::rcp_static_cast<const SymEngine::Integer>(decomp.coeff)->as_int();
                        if (coeff_int != 1) {
                            return false; // Non-unit stride in tiled dimension
                        }
                        auto offset_expr = decomp.offset;
                        if (!SymEngine::is_a<SymEngine::Integer>(*offset_expr)) {
                            return false; // Non-constant offset
                        }
                        int offset =
                            static_cast<int>(SymEngine::down_cast<const SymEngine::Integer&>(*offset_expr).as_int());
                        min_offset = std::min(min_offset, offset);
                        max_offset = std::max(max_offset, offset);
                    }
                }

                // Capture layout info from first access (all accesses to same container share layout)
                if (layout_dimensions.empty()) {
                    auto& layout = mem_access->layout;
                    for (size_t d = 0; d < layout.dims(); d++) {
                        layout_dimensions.push_back(layout.shape().at(d));
                        layout_strides.push_back(layout.strides().at(d));
                    }
                    layout_offset = layout.offset();
                }
            }

            if (tiled_dim == -1) {
                return false; // No dimension depends on inner indvar
            }

            int tiled_dim_buf_size = tile_size + 2 * overall_radius + max_offset - min_offset;

            // Compute total buffer size: tiled_dim_buf_size * product(other dimensions)
            symbolic::Expression total_buf_size = symbolic::integer(tiled_dim_buf_size);
            for (size_t d = 0; d < layout_dimensions.size(); d++) {
                if (static_cast<int>(d) != tiled_dim) {
                    total_buf_size = symbolic::mul(total_buf_size, layout_dimensions[d]);
                }
            }

            // Total buffer size must be a constant integer for now
            // (non-tiled dimensions may be symbolic like N — that's fine, buffer is symbolic-sized)
            CyclicContainerInfo cyc_info;
            cyc_info.container = container;
            cyc_info.min_offset = min_offset;
            cyc_info.max_offset = max_offset;
            cyc_info.tiled_dim = tiled_dim;
            cyc_info.tiled_dim_buf_size = tiled_dim_buf_size;
            cyc_info.layout = math::tensor::TensorLayout(layout_dimensions, layout_strides, layout_offset);
            // buffer_size is now used as a symbolic expression in apply, but store int for 1D compat
            if (SymEngine::is_a<SymEngine::Integer>(*total_buf_size)) {
                cyc_info.buffer_size =
                    static_cast<int>(SymEngine::down_cast<const SymEngine::Integer&>(*total_buf_size).as_int());
            } else {
                cyc_info.buffer_size = -1; // Symbolic — will use expression in apply
            }
            cyclic_containers_.push_back(cyc_info);
        }
    } // if (overall_radius > 0)

    return true;
}

void TileFusion::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    if (radius_ == 0) {
        apply_aligned(builder, analysis_manager);
        return;
    }
    auto& sdfg = builder.subject();
    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();

    // Get parent sequence
    auto* parent = static_cast<structured_control_flow::Sequence*>(first_map_.get_parent());
    size_t first_index = parent->index(first_map_);

    // Extract tile loop properties from first map (representative)
    auto tile_indvar = first_map_.indvar();
    auto tile_init = first_map_.init();
    auto tile_condition = first_map_.condition();
    auto tile_update = first_map_.update();

    // Extract tile size
    auto stride = first_map_.stride();
    auto tile_size_expr = stride;

    // Get references to inner maps before moving
    auto* first_inner_map = dyn_cast<structured_control_flow::Map*>(&first_map_.root().at(0));
    auto* second_inner_map = dyn_cast<structured_control_flow::Map*>(&second_map_.root().at(0));

    // Extract inner map properties before they get moved
    auto first_inner_indvar = first_inner_map->indvar();
    auto first_inner_init = first_inner_map->init();
    auto first_inner_condition = first_inner_map->condition();
    auto first_inner_update = first_inner_map->update();
    auto first_inner_schedule = first_inner_map->schedule_type();

    auto second_inner_indvar = second_inner_map->indvar();
    auto second_inner_init = second_inner_map->init();
    auto second_inner_condition = second_inner_map->condition();
    auto second_inner_update = second_inner_map->update();
    auto second_inner_schedule = second_inner_map->schedule_type();

    // Get the old tile indvars to substitute later
    auto first_tile_indvar = first_map_.indvar();
    auto second_tile_indvar = second_map_.indvar();

    // Step 1: Create a new For tile loop with the same bounds as the first tile Map
    // Use the condition from the first map (which uses first_tile_indvar)
    // We need a fresh indvar for the new tile loop
    auto new_tile_indvar_name = builder.find_new_name(tile_indvar->get_name());
    builder.add_container(new_tile_indvar_name, sdfg.type(tile_indvar->get_name()));
    auto new_tile_indvar = symbolic::symbol(new_tile_indvar_name);

    // Substitute the old tile indvar in the condition with the new one
    auto new_tile_condition = symbolic::subs(tile_condition, tile_indvar, new_tile_indvar);

    auto& fused_for = builder.add_for_before(
        *parent,
        first_map_,
        new_tile_indvar,
        new_tile_condition,
        tile_init,
        symbolic::subs(tile_update, tile_indvar, new_tile_indvar),
        first_map_.debug_info()
    );

    // Step 2: Create the producer inner Map inside the fused For
    // Its init and condition need to reference the new tile indvar
    auto new_first_inner_init = symbolic::subs(first_inner_init, first_tile_indvar, new_tile_indvar);
    auto new_first_inner_condition = symbolic::subs(first_inner_condition, first_tile_indvar, new_tile_indvar);

    // Extend the producer's range by the radius
    if (radius_ > 0) {
        // Extend init: max(0, tile - radius) -> take the original init and subtract radius
        // Original init is typically: tile_indvar (i.e., the iteration starts at the tile boundary)
        // New init: max(original_lower_bound, new_init - radius)
        auto radius_expr = symbolic::integer(radius_);
        auto extended_init = symbolic::max(symbolic::integer(0), symbolic::sub(new_first_inner_init, radius_expr));
        new_first_inner_init = extended_init;

        // Extend condition: the condition has form like And(i < tile + S, i < N)
        // We need to adjust the tile-relative bound: i < tile + S becomes i < tile + S + radius
        // We do this by substituting new_tile_indvar with (new_tile_indvar + radius) in the
        // tile-relative part. Since the condition is a conjunction, we can reconstruct it:
        // Original: And(inner_indvar < new_tile_indvar + S, inner_indvar < N)
        // Extended: And(inner_indvar < new_tile_indvar + S + radius, inner_indvar < N)
        auto extended_tile_bound =
            symbolic::Lt(first_inner_indvar, symbolic::add(new_tile_indvar, symbolic::add(tile_size_expr, radius_expr)));

        // Extract non-tile upper bounds from the original inner map condition.
        // canonical_bound() returns min of ALL bounds including the tile bound,
        // but we only want bounds independent of the tile indvar (e.g., array size).
        auto original_condition = first_inner_map->condition();
        symbolic::CNF cnf;
        try {
            cnf = symbolic::conjunctive_normal_form(original_condition);
        } catch (...) {
            cnf.clear();
        }

        // Collect non-tile upper bounds on the inner indvar
        std::vector<symbolic::Expression> non_tile_bounds;
        for (auto& clause : cnf) {
            for (auto& literal : clause) {
                // Check for i < bound forms
                if (SymEngine::is_a<SymEngine::StrictLessThan>(*literal)) {
                    auto lt = SymEngine::rcp_static_cast<const SymEngine::StrictLessThan>(literal);
                    auto lhs = lt->get_arg1();
                    auto rhs = lt->get_arg2();
                    if (symbolic::eq(lhs, first_inner_indvar) && !symbolic::uses(rhs, first_tile_indvar->get_name())) {
                        non_tile_bounds.push_back(rhs);
                    }
                }
            }
        }

        if (!non_tile_bounds.empty()) {
            // Build condition: extended_tile_bound AND all non-tile bounds
            auto result_condition = extended_tile_bound;
            for (auto& bound : non_tile_bounds) {
                result_condition = symbolic::And(result_condition, symbolic::Lt(first_inner_indvar, bound));
            }
            new_first_inner_condition = result_condition;
        } else {
            new_first_inner_condition = extended_tile_bound;
        }
    }

    auto& new_first_inner = builder.add_map(
        fused_for.root(),
        first_inner_indvar,
        new_first_inner_condition,
        new_first_inner_init,
        first_inner_update,
        first_inner_schedule
    );

    // Move the producer body into the new inner map
    builder.move_children(first_inner_map->root(), new_first_inner.root());

    // Step 3: Create the consumer inner Map inside the fused For, after the producer
    auto new_second_inner_init = symbolic::subs(second_inner_init, second_tile_indvar, new_tile_indvar);
    auto new_second_inner_condition = symbolic::subs(second_inner_condition, second_tile_indvar, new_tile_indvar);

    auto& new_second_inner = builder.add_map(
        fused_for.root(),
        second_inner_indvar,
        new_second_inner_condition,
        new_second_inner_init,
        second_inner_update,
        second_inner_schedule
    );

    // Move the consumer body into the new inner map
    builder.move_children(second_inner_map->root(), new_second_inner.root());

    // ==========================================================================
    // Step 3.5: Double-buffer pre-fetch for cyclic containers
    //
    // Uses layout info from MemoryLayoutAnalysis (computed in can_be_applied)
    // to handle multi-dimensional stencils. For each cyclic container,
    // generates nested copy loops and rewrites K1's reads to use the buffer.
    // ==========================================================================

    for (const auto& cyc : cyclic_containers_) {
        auto& container_type = sdfg.type(cyc.container);
        // Get scalar element type from container (Pointer or Array)
        const types::IType* scalar_type_ptr = nullptr;
        if (auto* ptr_type = dynamic_cast<const types::Pointer*>(&container_type)) {
            scalar_type_ptr = &ptr_type->pointee_type();
        } else if (auto* arr_type = dynamic_cast<const types::Array*>(&container_type)) {
            scalar_type_ptr = &arr_type->element_type();
        } else {
            throw std::runtime_error("TileFusion: cyclic container '" + cyc.container + "' has unsupported type");
        }
        auto& scalar_type = *scalar_type_ptr;
        bool is_pointer = dynamic_cast<const types::Pointer*>(&container_type) != nullptr;

        auto radius_expr = symbolic::integer(radius_);
        auto min_offset_expr = symbolic::integer(cyc.min_offset);
        types::Scalar idx_type(types::PrimitiveType::UInt64);

        // Build buffer dimensions: same as layout but with tiled dim replaced by tiled_dim_buf_size
        std::vector<symbolic::Expression> buf_dimensions;
        for (size_t d = 0; d < cyc.layout.shape().size(); d++) {
            if (static_cast<int>(d) == cyc.tiled_dim) {
                buf_dimensions.push_back(symbolic::integer(cyc.tiled_dim_buf_size));
            } else {
                buf_dimensions.push_back(cyc.layout.shape()[d]);
            }
        }

        // Compute total buffer size
        symbolic::Expression total_buf_size = symbolic::integer(1);
        for (auto& dim : buf_dimensions) {
            total_buf_size = symbolic::mul(total_buf_size, dim);
        }

        // Create two buffer containers (flat 1D arrays of total_buf_size)
        auto buf_cur_name = "__tf_buf_cur_" + cyc.container;
        auto buf_pf_name = "__tf_buf_pf_" + cyc.container;
        types::Array buf_type(scalar_type, total_buf_size);
        builder.add_container(buf_cur_name, buf_type);
        builder.add_container(buf_pf_name, buf_type);

        // Helper: linearize multi-dim indices into flat buffer index (row-major)
        auto linearize_buf = [&](const std::vector<symbolic::Expression>& indices) -> symbolic::Expression {
            symbolic::Expression linear_idx = symbolic::integer(0);
            symbolic::Expression stride = symbolic::integer(1);
            for (int i = static_cast<int>(indices.size()) - 1; i >= 0; i--) {
                linear_idx = symbolic::add(linear_idx, symbolic::mul(indices[i], stride));
                stride = symbolic::mul(stride, buf_dimensions[i]);
            }
            return linear_idx;
        };

        // Helper: build original source subset from multi-dim copy indices
        // For the tiled dim, the source index is: base + copy_indvar
        // For non-tiled dims, the source index is just: copy_indvar
        // If the container is a Pointer, re-linearize using layout strides
        auto build_src_subset = [&](const std::vector<symbolic::Expression>& copy_indices,
                                    const symbolic::Expression& tiled_base) -> data_flow::Subset {
            std::vector<symbolic::Expression> full_indices;
            for (size_t d = 0; d < cyc.layout.shape().size(); d++) {
                if (static_cast<int>(d) == cyc.tiled_dim) {
                    full_indices.push_back(symbolic::add(tiled_base, copy_indices[d]));
                } else {
                    full_indices.push_back(copy_indices[d]);
                }
            }
            if (is_pointer) {
                // Re-linearize using layout strides: offset + sum(stride[d] * index[d])
                symbolic::Expression linear = cyc.layout.offset();
                for (size_t d = 0; d < full_indices.size(); d++) {
                    linear = symbolic::add(linear, symbolic::mul(cyc.layout.strides()[d], full_indices[d]));
                }
                return {linear};
            } else {
                return data_flow::Subset(full_indices.begin(), full_indices.end());
            }
        };

        // Helper: generate nested copy loops and a copy block
        // copy_kind: 0 = src→buf_cur, 1 = src→buf_pf, 2 = buf_pf→buf_cur
        auto generate_copy_loops = [&](structured_control_flow::Sequence& parent_scope,
                                       structured_control_flow::ControlFlowNode* insert_before,
                                       const symbolic::Expression& tiled_base,
                                       int copy_kind) {
            // Create indvars for each dimension
            std::vector<symbolic::Symbol> copy_indvars;
            structured_control_flow::Sequence* scope = &parent_scope;
            bool first_loop = true;

            for (size_t d = 0; d < buf_dimensions.size(); d++) {
                std::string suffix = (copy_kind == 0) ? "init" : (copy_kind == 1) ? "pf" : "sw";
                auto indvar_name = builder.find_new_name("__tf_" + suffix + "_d" + std::to_string(d));
                builder.add_container(indvar_name, idx_type);
                auto indvar = symbolic::symbol(indvar_name);
                copy_indvars.push_back(indvar);

                auto init = symbolic::integer(0);
                auto condition = symbolic::Lt(indvar, buf_dimensions[d]);
                auto update = symbolic::add(indvar, symbolic::integer(1));

                if (first_loop && insert_before != nullptr) {
                    auto& loop = builder.add_map_before(
                        *scope,
                        *insert_before,
                        indvar,
                        condition,
                        init,
                        update,
                        structured_control_flow::ScheduleType_Sequential::create()
                    );
                    scope = &loop.root();
                    first_loop = false;
                } else if (first_loop) {
                    auto& loop = builder.add_map(
                        *scope,
                        indvar,
                        condition,
                        init,
                        update,
                        structured_control_flow::ScheduleType_Sequential::create()
                    );
                    scope = &loop.root();
                    first_loop = false;
                } else {
                    auto& loop = builder.add_map(
                        *scope,
                        indvar,
                        condition,
                        init,
                        update,
                        structured_control_flow::ScheduleType_Sequential::create()
                    );
                    scope = &loop.root();
                }
            }

            // Create copy block
            auto& blk = builder.add_block(*scope);
            std::vector<symbolic::Expression> copy_exprs(copy_indvars.begin(), copy_indvars.end());
            auto buf_subset = data_flow::Subset{linearize_buf(copy_exprs)};

            if (copy_kind == 0 || copy_kind == 1) {
                // src container → buffer
                auto src_subset = build_src_subset(copy_exprs, tiled_base);
                auto& src = builder.add_access(blk, cyc.container);
                auto& dst = builder.add_access(blk, (copy_kind == 0) ? buf_cur_name : buf_pf_name);
                auto& tasklet = builder.add_tasklet(blk, data_flow::TaskletCode::assign, "_out", {"_in"});
                builder.add_computational_memlet(blk, src, tasklet, "_in", src_subset, container_type);
                builder.add_computational_memlet(blk, tasklet, "_out", dst, buf_subset, buf_type);
            } else {
                // buf_pf → buf_cur
                auto& src = builder.add_access(blk, buf_pf_name);
                auto& dst = builder.add_access(blk, buf_cur_name);
                auto& tasklet = builder.add_tasklet(blk, data_flow::TaskletCode::assign, "_out", {"_in"});
                builder.add_computational_memlet(blk, src, tasklet, "_in", buf_subset, buf_type);
                builder.add_computational_memlet(blk, tasklet, "_out", dst, buf_subset, buf_type);
            }
        };

        // ==================================================================
        // 3.5a: Initial copy INSIDE the fused For, guarded by if (tile == init)
        // On the first iteration, we load from the original array.
        // On subsequent iterations, buf_cur already has the right data from swap.
        // Base for tiled dim: tile_init - radius + min_offset
        // ==================================================================
        auto init_tiled_base = symbolic::sub(symbolic::add(tile_init, min_offset_expr), radius_expr);

        // Get a reference to the pre-fetch map (first child of fused_for) to insert before it
        auto& first_child_in_fused = fused_for.root().at(0);

        auto& init_if_else = builder.add_if_else_before(fused_for.root(), first_child_in_fused);
        auto init_cond = symbolic::Eq(new_tile_indvar, tile_init);
        auto& init_then = builder.add_case(init_if_else, init_cond);
        // Empty else branch (on subsequent iters, buf_cur has data from swap)
        builder.add_case(init_if_else, symbolic::Not(init_cond));

        generate_copy_loops(init_then, nullptr, init_tiled_base, 0);

        // ==================================================================
        // 3.5b: Pre-fetch INSIDE the fused For, BEFORE K1
        // Base for tiled dim: tile + S - radius + min_offset
        // ==================================================================
        auto pf_tiled_base =
            symbolic::sub(symbolic::add(new_tile_indvar, symbolic::add(tile_size_expr, min_offset_expr)), radius_expr);
        generate_copy_loops(fused_for.root(), &new_first_inner, pf_tiled_base, 1);

        // ==================================================================
        // 3.5c: Redirect K1's reads of container to buf_cur
        // Use MemoryLayoutAnalysis to get delinearized indices, rebase
        // the tiled dimension, then re-linearize into the flat buffer.
        // ==================================================================
        auto buf_cur_base_tiled = symbolic::sub(symbolic::add(new_tile_indvar, min_offset_expr), radius_expr);

        // Helper: rebase a raw subset for the buffer.
        // For Pointer (linearized) access A[expr]:
        //   buf_index = expr - stride[tiled_dim] * buf_cur_base_tiled
        // For Array (multi-dim) access A[idx0, idx1, ...]:
        //   rebase tiled_dim, linearize into flat buffer index
        auto rebase_to_buffer = [&](const data_flow::Subset& subset) -> data_flow::Subset {
            if (subset.size() == 1 && cyc.layout.shape().size() > 1 && is_pointer) {
                // Pointer type: subtract the tiled dimension's contribution to the linear base
                auto linear_base = symbolic::mul(cyc.layout.strides()[cyc.tiled_dim], buf_cur_base_tiled);
                return {symbolic::sub(subset.at(0), linear_base)};
            }
            if (subset.size() == cyc.layout.shape().size()) {
                // Multi-dim: rebase tiled dimension, linearize into buffer
                std::vector<symbolic::Expression> buf_indices;
                for (size_t d = 0; d < cyc.layout.shape().size(); d++) {
                    if (static_cast<int>(d) == cyc.tiled_dim) {
                        buf_indices.push_back(symbolic::sub(subset[d], buf_cur_base_tiled));
                    } else {
                        buf_indices.push_back(subset[d]);
                    }
                }
                return {linearize_buf(buf_indices)};
            }
            // 1D Array: rebase directly
            if (subset.size() == 1) {
                return {symbolic::sub(subset.at(0), buf_cur_base_tiled)};
            }
            return subset; // Fallback: no rebase
        };

        std::function<void(structured_control_flow::ControlFlowNode&)> rewrite_accesses;
        rewrite_accesses = [&](structured_control_flow::ControlFlowNode& node) {
            if (auto* block = dyn_cast<structured_control_flow::Block*>(&node)) {
                auto& dfg = block->dataflow();
                for (auto* access : dfg.data_nodes()) {
                    if (access->data() != cyc.container) {
                        continue;
                    }
                    bool has_writes = dfg.in_degree(*access) > 0;
                    for (auto& memlet : dfg.out_edges(*access)) {
                        if (memlet.type() != data_flow::MemletType::Computational) {
                            continue;
                        }
                        memlet.set_subset(rebase_to_buffer(memlet.subset()));
                        memlet.set_base_type(buf_type);
                    }
                    if (!has_writes) {
                        access->data(buf_cur_name);
                    }
                }
            } else if (auto* seq = dyn_cast<structured_control_flow::Sequence*>(&node)) {
                for (size_t i = 0; i < seq->size(); i++) {
                    rewrite_accesses(seq->at(i));
                }
            } else if (auto* loop = dyn_cast<structured_control_flow::StructuredLoop*>(&node)) {
                rewrite_accesses(loop->root());
            } else if (auto* if_else = dyn_cast<structured_control_flow::IfElse*>(&node)) {
                for (size_t i = 0; i < if_else->size(); i++) {
                    rewrite_accesses(if_else->at(i).first);
                }
            }
        };
        rewrite_accesses(new_first_inner.root());

        // ==================================================================
        // 3.5d: Swap INSIDE the fused For, AFTER K2
        // buf_cur = buf_pf
        // ==================================================================
        generate_copy_loops(fused_for.root(), nullptr, symbolic::integer(0), 2);
    }

    // Step 4: Remove the original two tile Map nests
    // After we've moved the children out, remove the old maps from the parent
    // The fused_for was inserted before first_map_, so first_map_ is now at first_index + 1
    // and second_map_ is at first_index + 2
    // Remove second first (higher index) to avoid invalidating first's index
    size_t current_first_index = parent->index(first_map_);
    size_t current_second_index = parent->index(second_map_);
    builder.remove_child(*parent, current_second_index);
    builder.remove_child(*parent, current_first_index);

    analysis_manager.invalidate_all();
    applied_ = true;
    fused_loop_ = &fused_for;
}

void TileFusion::apply_aligned(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& sdfg = builder.subject();
    auto& parent = static_cast<structured_control_flow::Sequence&>(*first_map_.get_parent());
    auto first_counter = first_map_.indvar();
    auto second_counter = second_map_.indvar();
    auto counter = symbolic::symbol(builder.find_new_name(first_counter->get_name()));
    builder.add_container(counter->get_name(), sdfg.type(first_counter->get_name()));
    // The fused loop runs the first map's iterations, the second's of the same number aligned to them: offset if the
    // steps agree, else both at the iteration number counting from zero in steps of one.
    symbolic::Condition condition;
    symbolic::Expression init, update, first_at, second_at;
    if (symbolic::eq(first_map_.stride(), second_map_.stride())) {
        condition = symbolic::subs(first_map_.condition(), first_counter, counter);
        init = first_map_.init();
        update = symbolic::subs(first_map_.update(), first_counter, counter);
        first_at = counter;
        second_at = symbolic::add(counter, symbolic::sub(second_map_.init(), first_map_.init()));
    } else {
        condition = symbolic::Lt(counter, first_map_.num_iterations());
        init = symbolic::zero();
        update = symbolic::add(counter, symbolic::one());
        first_at = symbolic::add(first_map_.init(), symbolic::mul(first_map_.stride(), counter));
        second_at = symbolic::add(second_map_.init(), symbolic::mul(second_map_.stride(), counter));
    }
    structured_control_flow::StructuredLoop* fused;
    if (sequential_) {
        fused = &builder.add_for_before(parent, first_map_, counter, condition, init, update, first_map_.debug_info());
    } else {
        fused = &builder.add_map_before(
            parent, first_map_, counter, condition, init, update, first_map_.schedule_type(), first_map_.debug_info()
        );
    }
    first_map_.root().replace(first_counter, first_at);
    second_map_.root().replace(second_counter, second_at);
    builder.move_children(first_map_.root(), fused->root());
    builder.move_children(second_map_.root(), fused->root());
    builder.remove_child(parent, parent.index(first_map_));
    builder.remove_child(parent, parent.index(second_map_));
    analysis_manager.invalidate_all();
    applied_ = true;
    fused_loop_ = fused;
}

void TileFusion::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["parameters"] = nlohmann::json::object();

    serializer::JSONSerializer ser_flat(false);
    j["subgraph"] = nlohmann::json::object();
    j["subgraph"]["0"] = nlohmann::json::object();
    ser_flat.serialize_node(j["subgraph"]["0"], first_map_);

    j["subgraph"]["1"] = nlohmann::json::object();
    ser_flat.serialize_node(j["subgraph"]["1"], second_map_);
}

TileFusion TileFusion::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc) {
    auto first_map_id = desc["subgraph"]["0"]["element_id"].get<size_t>();
    auto second_map_id = desc["subgraph"]["1"]["element_id"].get<size_t>();

    auto first_element = builder.find_element_by_id(first_map_id);
    auto second_element = builder.find_element_by_id(second_map_id);

    if (first_element == nullptr) {
        throw InvalidTransformationDescriptionException("Element with ID " + std::to_string(first_map_id) + " not found.");
    }
    if (second_element == nullptr) {
        throw InvalidTransformationDescriptionException(
            "Element with ID " + std::to_string(second_map_id) + " not found."
        );
    }

    auto* first_map = dyn_cast<structured_control_flow::Map*>(first_element);
    auto* second_map = dyn_cast<structured_control_flow::Map*>(second_element);

    if (first_map == nullptr) {
        throw InvalidTransformationDescriptionException(
            "Element with ID " + std::to_string(first_map_id) + " is not a Map."
        );
    }
    if (second_map == nullptr) {
        throw InvalidTransformationDescriptionException(
            "Element with ID " + std::to_string(second_map_id) + " is not a Map."
        );
    }

    return TileFusion(*first_map, *second_map);
}

structured_control_flow::StructuredLoop* TileFusion::fused_loop() const {
    if (!applied_) {
        throw InvalidTransformationException("Accessing fused loop before apply.");
    }
    return fused_loop_;
}

bool TileFusion::sequential() const {
    return sequential_;
}

int TileFusion::radius() const {
    return radius_;
}

} // namespace transformations
} // namespace sdfg
