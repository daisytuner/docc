#pragma once

#include <cstdint>
#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <sdfg/data_flow/access_node.h>
#include <sdfg/data_flow/data_flow_node.h>
#include <sdfg/data_flow/memlet.h>
#include <sdfg/data_flow/tasklet.h>
#include <sdfg/einsum/einsum_node.h>
#include <sdfg/structured_control_flow/block.h>
#include <sdfg/structured_control_flow/structured_loop.h>
#include <sdfg/symbolic/conjunctive_normal_form.h>
#include <sdfg/symbolic/symbolic.h>
#include <sdfg/visitor/structured_sdfg_visitor.h>

namespace sdfg::einsum {

/**
 * @brief A single einsum dimension's contribution to one operand's access subset.
 *
 * The dimension's induction variable appears in exactly one index position of the operand's
 * subset, scaled by an integer factor to support linearized accesses (e.g. `i` in `[k + i*8]`
 * contributes to index position 0 with factor 8).
 */
struct EinsumIndexContribution {
    /// Induction variable of the contributing dimension.
    symbolic::Symbol indvar;
    /// Index position within the original subset that this dimension contributes to.
    size_t index = 0;
    /// Integer coefficient of the indvar within that index expression.
    int64_t factor = 1;

    symbolic::Expression factor_expr() const;
};

/**
 * @brief How the consumed einsum dimensions map onto one operand's access subset.
 *
 * Records, per operand, which dimensions contribute to which original index position (and with
 * which factor), plus how many of the original subset's index positions - counted from the
 * innermost outward - are fully reconstructed by those contributions. The outermost
 * `subset.size() - covered` positions are outer/irrelevant to the einsum operation.
 */
struct EinsumIndexing {
    /// Contributions of individual dimensions to this operand's index positions.
    std::vector<EinsumIndexContribution> contributions;
    /// Number of innermost original index positions fully covered by the dimensions.
    size_t covered = 0;

    /**
     * When a index is applied to a linearized subset, it can make additional contributions to the stride in the amount
     * of factor.
     * @param idx the index of the dimension in the contributions vector
     * @param shape a list of the shapes of the underlying type onto which subset we apply
     * @return will calculate the stride based on how many inner dimensions of shape are below the current contribution
     */
    symbolic::Expression get_stride_including_subsets(int idx, const std::vector<symbolic::Expression>& shape) const;
};

/**
 * @brief A single consumed dimension's contribution to one operand, found during promotion.
 *
 * `contributes` is false when the dimension's indvar does not appear in the operand's subset.
 */
struct OperandContribution {
    bool contributes = false;
    /// Index position within the operand's subset it contributes to.
    size_t index = 0;
    /// Integer coefficient of the indvar in that index expression.
    int64_t factor = 1;
};

/**
 * @brief Result of checking whether a loop can be folded into a cluster.
 *
 * Lists the new dimension's contribution to the output and to every input operand. `ok` is false
 * when the indvar violates the integer-affine index pattern in some operand, or contributes to no
 * operand at all.
 */
struct PromotionCheck {
    bool ok = false;
    OperandContribution output;
    std::vector<OperandContribution> inputs;
};

/**
 * @brief Non-destructive counterpart of an EinsumNode.
 *
 * Carries the same properties as an EinsumNode (dims, out/in indices, inputs)
 * but instead of being materialized in the graph it only references the graph
 * elements it represents. Additionally it records which dataflow nodes and which
 * surrounding loops have been consumed (folded) into the cluster.
 */
class EinsumCluster {
public:
    /// Block that holds the reduction core of this cluster.
    structured_control_flow::Block* block = nullptr;

    // Core einsum properties (mirroring EinsumNode, but excluding the implicit
    // "__einsum_out" reduction input, which is represented by output_node/out_indices).
    std::vector<EinsumDimension> dims;
    std::vector<data_flow::Memlet*> in_edges;
    std::vector<std::string> inputs;

    /// Inner-index mapping per input operand (parallel to in_indices/inputs).
    std::vector<EinsumIndexing> einsum_in_indices;
    /// Inner-index mapping of the output operand.
    EinsumIndexing einsum_out_indices;

    /// Reduction/output access node of the cluster.
    data_flow::Memlet* output_edge = nullptr;
    data_flow::AccessNode* output_node = nullptr;
    /// Source node for each input connector (may be null for synthetic constants).
    std::unordered_map<std::string, data_flow::DataFlowNode*> input_nodes;
    /// True if the core reduction was a subtraction (carries an implicit -1 factor).
    bool subtraction = false;

    /// Dataflow nodes folded into this cluster (reduction tasklet, mul tasklets, ...).
    std::list<data_flow::DataFlowNode*> consumed_nodes;
    /// Surrounding loops folded into this cluster as reduction/free dimensions. From outermost to innermost
    std::list<structured_control_flow::StructuredLoop*> consumed_loops;

    std::pair<data_flow::AccessNode*, data_flow::Memlet*> get_input_for(size_t input_idx);

    const std::vector<EinsumIndexing>& get_input_indexings() const;

    std::string toStr() const;

    const data_flow::Subset& get_input_subset(size_t input_idx) const;

    const EinsumIndexing& get_output_indexing() const;

    symbolic::Expression get_linearized_outer_offset(int input_idx) const;

    const data_flow::Subset& out_indices() const;
};

/**
 * @brief Non-destructive detection of einsum clusters in a StructuredSDFG.
 *
 * Traverses the graph from a user-chosen starting point, builds the loop stack
 * and tries to represent each Block of dataflow with a single EinsumCluster. If
 * a loop scope only contains a single EinsumCluster and no other operation, the
 * loop is folded into that cluster on the way out of the scope.
 *
 * The SDFG is never modified.
 */
class EinsumDetection : public visitor::ActualStructuredSDFGVisitor {
protected:
    struct BlockState {
        std::unordered_set<data_flow::DataFlowNode*> consumed_nodes;
        std::list<EinsumCluster*> clusters;
    };
    struct LoopScopeState {
        /// found more than a single block in this scope, or sth. else that makes it impossible to consume the
        /// surrounding loop into the singular einsum cluster that represents its contents.
        bool non_reducible = false;
        EinsumCluster* representing_last_block_ = nullptr;
    };


    /// all that have been found
    std::vector<std::unique_ptr<EinsumCluster>> all_einsums_;
    /// those that are relevant to the current scope.
    std::vector<LoopScopeState> loop_stack_;

    static bool subsets_eq(const data_flow::Subset& subset1, const data_flow::Subset& subset2);

    EinsumCluster& new_cluster(structured_control_flow::Block& block);

    /// Try to lift a reduction tasklet into a fresh einsum cluster core.
    EinsumCluster* try_lift(structured_control_flow::Block& block, data_flow::Tasklet& tasklet);

    /// Mirror an already existing EinsumNode into a cluster.
    EinsumCluster* from_einsum_node(structured_control_flow::Block& block, EinsumNode& einsum_node);

    /// Try to fold a multiplication feeding an input into the cluster. Returns true if applied.
    bool try_extend(BlockState& state, EinsumCluster& cluster);

    static symbolic::Expression cnf_to_upper_bound(const symbolic::CNF& cnf, const symbolic::Symbol indvar);
    static bool subset_contains_symbol(const data_flow::Subset& subset, const symbolic::Symbol& symbol);

    /// Verify the loop can be folded and return the new dimension's per-operand contributions.
    PromotionCheck can_promote(EinsumCluster& cluster, structured_control_flow::StructuredLoop& loop);
    void apply_promote(EinsumCluster& cluster, structured_control_flow::StructuredLoop& loop, const PromotionCheck& check);

    /// Recompute the per-operand inner-index mappings from the current dims and subsets.
    void recompute_einsum_indices(EinsumCluster& cluster);

    bool block_fully_consumed(BlockState& state, structured_control_flow::Block& block);
    void register_cluster_in_scope(EinsumCluster* cluster);

public:
    EinsumDetection() = default;

    /// Loop for operations (reductions) that can form the core of einsum clusters.
    void find_einsum_core_ops(BlockState& state, structured_control_flow::Block& block);

    /// for all the clusters of the block last scanned into BlockState, try to consume all the input operations that
    /// feed into them and are still unconsumed, and add them to the clusters.
    void consume_input_operations(BlockState& state);

    /**
     * If there is a cluster representing the full contents of the loop, try to have it consume this loop
     * @param state
     * @param loop
     * @return true if the loop was successfully consumed by the cluster, false otherwise.
     */
    bool try_to_consume_loop(LoopScopeState& state, structured_control_flow::StructuredLoop& loop);

    void filter_for_coverage();

    /// Run detection starting from the given control flow node.
    void run(structured_control_flow::ControlFlowNode& start);

    /// All einsum clusters found during the last run.
    const std::vector<std::unique_ptr<EinsumCluster>>& einsums() const { return all_einsums_; }

    using visitor::ActualStructuredSDFGVisitor::visit;

    bool visit(structured_control_flow::Block& node) override;
    bool visit(structured_control_flow::AssignmentBlock& node) override;
    bool visit(structured_control_flow::IfElse& node) override;
    bool visit(structured_control_flow::While& node) override;
    bool visit(structured_control_flow::Return& node) override;
    bool handleStructuredLoop(structured_control_flow::StructuredLoop& loop) override;
};

} // namespace sdfg::einsum
