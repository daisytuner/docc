#pragma once

#include <memory>
#include <string>
#include <vector>

#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {

namespace builder {
class StructuredSDFGBuilder;
}
namespace structured_control_flow {

/**
 * @brief Associative and commutative reduction operators
 *
 * A Reduce loop combines the values produced across its iterations into an
 * accumulator using one of these operators. Only operators that are both
 * associative and commutative are representable here, because that is exactly
 * the property that makes the loop-carried dependency safe to parallelize
 * (the partial results of disjoint iteration subsets may be combined in any
 * order).
 */
enum class ReductionOperation { Add, Mul, Min, Max };

/**
 * @brief Serialize a reduction operator to its canonical string identifier
 */
std::string reduction_operation_to_string(ReductionOperation op);

/**
 * @brief Parse a reduction operator from its canonical string identifier
 * @throws InvalidSDFGException if the string does not name a known operator
 */
ReductionOperation reduction_operation_from_string(const std::string& value);

/**
 * @brief A single reduction carried by a @ref Reduce loop
 *
 * Pairs an associative/commutative @ref ReductionOperation with the name of the
 * accumulator container it combines into. The container name identifies which
 * access node / memlet in the loop body holds the accumulator; the concrete
 * subset (for partial reductions over arrays or struct members) is left to the
 * body's memlets so that there is a single source of truth for *where* the
 * accumulator lives.
 *
 * A @ref Reduce may carry several of these at once (e.g. a fused sum and max),
 * which is why the node stores a collection rather than a single operator.
 */
struct ReductionInfo {
    ReductionOperation operation;
    std::string container;
};

/**
 * @brief Represents a loop with an associative/commutative reduction carried
 *        dependency
 *
 * A Reduce loop has the same shape as a @ref For loop (induction variable,
 * init, update, condition and a body sequence), but additionally carries the
 * semantic information that its body combines per-iteration values into an
 * accumulator using an associative and commutative @ref ReductionOperation.
 *
 * Unlike a @ref Map, the iterations of a Reduce loop are **not** independent:
 * they all contribute to a shared accumulator. Unlike a plain @ref For, the
 * combine is known to be reorderable, which is what later transformations
 * exploit to lower it to a parallel (tree / warp-shuffle) reduction.
 *
 * The accumulator container(s) and the concrete combine tasklet remain in the
 * loop body; this node records, per reduction, the operator together with the
 * name of the accumulator container (see @ref ReductionInfo) so that codegen
 * and transformations know how to initialize partials and combine them. A
 * single Reduce may carry multiple reductions at once.
 *
 * @see StructuredLoop
 * @see For
 * @see Map
 */
class Reduce : public StructuredLoop {
    friend class sdfg::builder::StructuredSDFGBuilder;

private:
    std::vector<ReductionInfo> reductions_;

    Reduce(
        size_t element_id,
        const DebugInfo& debug_info,
        ControlFlowNode* parent,
        symbolic::Symbol indvar,
        symbolic::Expression init,
        symbolic::Expression update,
        symbolic::Condition condition,
        std::vector<ReductionInfo> reductions,
        const ScheduleType& schedule_type
    );

    static constexpr size_t REQUIRED_ELEMENT_IDS = StructuredLoop::REQUIRED_ELEMENT_IDS;

public:
    Reduce(const Reduce& node) = delete;
    Reduce& operator=(const Reduce&) = delete;

    ElementType type_id() const override { return ElementType::Reduce; }

    static bool classof(const Element& element) { return element.type_id() == ElementType::Reduce; }

    bool accept(visitor::ActualStructuredSDFGVisitor& visitor) override;

    void validate(const Function& function) const override;

    /**
     * @brief Get the reductions carried by this loop
     *
     * Each entry pairs an associative/commutative operator with the name of the
     * accumulator container it combines into. There is at least one entry.
     */
    const std::vector<ReductionInfo>& reductions() const;
};

} // namespace structured_control_flow
} // namespace sdfg
