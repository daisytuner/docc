#pragma once

#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {

namespace builder {
class StructuredSDFGBuilder;
}
namespace structured_control_flow {

/**
 * @brief Categories of schedule types for Map nodes
 * Defines the high-level classification of scheduling strategies.
 */
enum class ScheduleTypeCategory { Offloader, Parallelizer, Vectorizer, None };

/**
 * @brief Represents a schedule type for Map nodes
 *
 * ScheduleType encapsulates the execution schedule for Map loops, including
 * the scheduling strategy and associated properties. Different schedule types
 * control how loop iterations are distributed and executed.
 */
class ScheduleType {
private:
    std::unordered_map<std::string, std::string> properties_;
    std::string value_;
    ScheduleTypeCategory category_;

public:
    ScheduleType(std::string value, ScheduleTypeCategory category) : value_(value), category_(category) {}

    /**
     * @brief Get the schedule type identifier
     * @return Schedule type string (e.g., "SEQUENTIAL", "CPU_PARALLEL")
     */
    const std::string& value() const { return value_; }

    /**
     * @brief Get all schedule properties
     * @return Map of property names to values
     */
    const std::unordered_map<std::string, std::string>& properties() const { return properties_; }

    /**
     * @brief Get the schedule type category
     * @return Schedule type category enum
     */
    ScheduleTypeCategory category() const { return category_; }

    /**
     * @brief Set a schedule property
     * @param key Property name
     * @param value Property value
     */
    void set_property(const std::string& key, const std::string& value) {
        if (properties_.find(key) == properties_.end()) {
            properties_.insert({key, value});
            return;
        }
        properties_.at(key) = value;
    }

    void operator=(const ScheduleType& rhs) {
        value_ = rhs.value_;
        properties_.clear();
        for (const auto& entry : rhs.properties_) {
            properties_.insert(entry);
        }
        category_ = rhs.category_;
    }
};

/**
 * @brief Sequential schedule type for Map nodes
 *
 * Indicates that loop iterations execute sequentially in order.
 */
class ScheduleType_Sequential {
public:
    static const std::string value() { return "SEQUENTIAL"; }
    static ScheduleType create() { return ScheduleType(value(), ScheduleTypeCategory::None); }
};

/**
 * @brief Base class for structured loop constructs
 *
 * StructuredLoop is the abstract base class for all structured loop types in
 * a StructuredSDFG. It provides common functionality for loops with:
 * - An induction variable (loop counter)
 * - An initialization expression
 * - An update expression (how the induction variable changes each iteration)
 * - A loop condition (when to continue iterating)
 * - A body sequence containing the loop's control flow
 *
 * Derived loop types include:
 * - For: Traditional for-loops with explicit initialization, condition, and update
 * - Map: Parallel loops that can be mapped to parallel execution
 *
 * **Loop Structure:**
 * ```
 * indvar = init
 * while (condition):
 *   <body sequence>
 *   indvar = update
 * ```
 *
 * The loop body is a Sequence that can contain any control flow nodes (blocks,
 * nested loops, conditionals, etc.).
 *
 * @see For
 * @see Map
 * @see Sequence
 */
class StructuredLoop : public ControlFlowNode {
    friend class sdfg::builder::StructuredSDFGBuilder;

protected:
    symbolic::Symbol indvar_;
    symbolic::Expression init_;
    symbolic::Expression update_;
    symbolic::Condition condition_;

    std::unique_ptr<Sequence> root_;

    ScheduleType schedule_type_;

    StructuredLoop(
        size_t element_id,
        const DebugInfo& debug_info,
        ControlFlowNode* parent,
        symbolic::Symbol indvar,
        symbolic::Expression init,
        symbolic::Expression update,
        symbolic::Condition condition,
        const ScheduleType& schedule_type = ScheduleType_Sequential::create()
    );

    static constexpr size_t REQUIRED_ELEMENT_IDS = 2;

public:
    virtual ~StructuredLoop() = default;

    StructuredLoop(const StructuredLoop& node) = delete;
    StructuredLoop& operator=(const StructuredLoop&) = delete;

    void validate(const Function& function) const override;

    /**
     * @brief Get the induction variable (loop counter) symbol
     * @return The loop's induction variable
     */
    const symbolic::Symbol indvar() const;

    /**
     * @brief Get the initialization expression for the induction variable
     * @return Expression evaluated to initialize the induction variable
     */
    const symbolic::Expression init() const;

    /**
     * @brief Get the update expression for the induction variable
     * @return Expression evaluated to update the induction variable each iteration
     */
    const symbolic::Expression update() const;

    /**
     * @brief Get the loop continuation condition
     * @return Boolean expression evaluated before each iteration; loop continues while true
     */
    const symbolic::Condition condition() const;

    /**
     * @brief Access the loop body sequence
     * @return Reference to the sequence containing the loop body
     */
    Sequence& root() const;

    /**
     * @brief Get the scheduling strategy for this Map
     * @return The schedule type (sequential, parallel, etc.)
     */
    const ScheduleType& schedule_type() const;

    /**
     * @brief Replace occurrences of an expression in loop parameters and body
     * @param old_expression Expression to replace
     * @param new_expression Expression to replace with
     */
    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    void replace(const symbolic::ExpressionMapping& replacements) override;

    /**
     * @brief Describes the stride of a loop's update as a constant.
     *
     * @return The stride of the loop's update as a constant, otherwise null.
     */
    symbolic::Integer stride();

    /**
     * @brief Checks if the loop has a positive unit stride (i.e., update is indvar + 1).
     *
     * @return True if the loop has a positive unit stride, false otherwise.
     */
    bool is_contiguous();

    /**
     * @brief Checks if the loop is monotonic (i.e., stride is a positive integer).
     *
     * @return True if the loop is monotonic, false otherwise.
     */
    bool is_monotonic();

    /**
     * @brief Describes the bound of a loop as a closed-form expression.
     *
     * Example: i <= N && i < M -> i < min(N + 1, M)
     *
     * @return The bound of the loop as a closed-form expression, otherwise null.
     */
    symbolic::Expression canonical_bound();

    /**
     * @brief Describes the upper bound of a loop (for positive stride).
     *
     * Extracts the exclusive upper bound from conditions like:
     *   - i < N -> N
     *   - i <= N -> N + 1
     *   - i + offset < N -> N - offset
     *
     * @return The upper bound expression, or null if not extractable.
     */
    symbolic::Expression canonical_bound_upper();

    /**
     * @brief Describes the lower bound of a loop (for negative stride).
     *
     * Extracts the exclusive lower bound from conditions like:
     *   - bound < i -> bound
     *   - bound <= i -> bound - 1
     *
     * @return The lower bound expression, or null if not extractable.
     */
    symbolic::Expression canonical_bound_lower();

    /**
     * @brief Describes the number of iterations of a loop as a closed-form expression.
     *
     * @return The number of iterations of the loop as a closed-form expression, otherwise null.
     */
    symbolic::Expression num_iterations();

    /**
     * @brief Overapproximated (upper-bound) number of iterations.
     *
     * Same formula as @ref num_iterations, but applies @ref symbolic::overapproximate
     * to the numerator before dividing by the stride. This collapses tile-style
     * patterns where the canonical bound is a `min(...)` containing the loop's
     * init expression. For example, with `init = k_tile0` and condition
     * `k < k_tile0 + 8 && k < 500`, @ref num_iterations returns
     * `max(0, min(500 - k_tile0, 8))` while this helper returns the constant `8`.
     *
     * The returned expression is always >= @ref num_iterations for all symbol
     * valuations. Use this when a conservative integer upper bound is needed
     * (e.g. tile sizing, working-set estimates) and the exact symbolic form is
     * not required.
     *
     * @return An expression that upper-bounds the iteration count, otherwise null.
     */
    symbolic::Expression num_iterations_approx();

    /**
     * @brief Checks if the loop is in a normal form.
     *
     * Criteria:
     *      - Loop starts from zero
     *      - Loop has positive unit stride (i + 1)
     *      - Loop has canonical bound
     */
    bool is_loop_normal_form();
};

} // namespace structured_control_flow
} // namespace sdfg
