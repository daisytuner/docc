#pragma once

#include <memory>
#include <utility>

#include "sdfg/builder/function_builder.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/function.h"
#include "sdfg/sdfg.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/reduce.h"
#include "sdfg/structured_control_flow/return.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/while.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/types/scalar.h"

using namespace sdfg::control_flow;
using namespace sdfg::structured_control_flow;

namespace sdfg {
namespace builder {

template<typename T>
struct ConditionalDeleter {
    bool should_delete_;

    ConditionalDeleter(bool should_delete = true) : should_delete_(should_delete) {}

    void operator()(T* ptr) const {
        if (should_delete_) {
            delete ptr;
        }
    }
};

/**
 * Note: Even though the class references unique_ptr, it will never delete an SDFG it has a reference to
 */
class StructuredSDFGBuilder : public FunctionBuilder {
private:
    std::unique_ptr<StructuredSDFG, ConditionalDeleter<StructuredSDFG>> structured_sdfg_;

    using owned = ConditionalDeleter<StructuredSDFG>;

    std::unordered_set<const control_flow::State*>
    determine_loop_nodes(SDFG& sdfg, const control_flow::State& start, const control_flow::State& end) const;

    structured_control_flow::While* current_traverse_loop_;

    void traverse(SDFG& sdfg);

    void structure_region(
        SDFG& sdfg,
        Sequence& scope,
        const State* entry,
        const State* exit,
        const std::unordered_set<const InterstateEdge*>& continues,
        const std::unordered_set<const InterstateEdge*>& breaks,
        const std::unordered_map<const control_flow::State*, const control_flow::State*>& pdom_tree,
        std::unordered_set<const control_flow::State*>& visited,
        bool is_loop_body = false
    );

    void add_dataflow(const data_flow::DataFlowGraph& from, Block& to);

    static constexpr int32_t INSERT_AT_END = -10;

    template<typename T, typename... Args>
    std::pair<T&, Transition&> insert_node_internal(
        Sequence& parent,
        int32_t insert_idx,
        const sdfg::control_flow::Assignments* assignments,
        const DebugInfo& debug_info,
        Args&&... args
    ) {
        auto child = std::unique_ptr<
            T>(new T(this->new_element_id_batch(T::REQUIRED_ELEMENT_IDS), debug_info, &parent, std::forward<Args>(args)...)
        );
        auto& new_child = *child;

        std::unique_ptr<Transition> transition;
        if (assignments) {
            transition =
                std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent, *assignments));
        } else {
            Assignments empty;
            transition = std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent, empty));
        }
        auto& new_transition = *transition;

        if (insert_idx == INSERT_AT_END) {
            parent.children_.push_back(std::move(child));
            parent.transitions_.push_back(std::move(transition));
        } else {
            parent.children_.insert(parent.children_.begin() + insert_idx, std::move(child));

            parent.transitions_.insert(parent.transitions_.begin() + insert_idx, std::move(transition));
        }

        return {new_child, new_transition};
    }

    std::pair<Block&, Transition&> insert_block_internal(
        Sequence& parent,
        int32_t insert_idx,
        const data_flow::DataFlowGraph* import_from,
        const sdfg::control_flow::Assignments* assignments,
        const DebugInfo& debug_info
    );

protected:
    Function& function() const override;

public:
    using InsertionPoint = int32_t;

    /**
     * To modify an existing SDFG
     */
    StructuredSDFGBuilder(StructuredSDFG& sdfg);

    /**
     * Will take ownership of the SDFG
     * Increases compatibility with legacy code. Also more idiomatic for SDFGs that are being deserialized and are not
     * yet owned_ by the registry
     */
    StructuredSDFGBuilder(std::unique_ptr<StructuredSDFG>& sdfg);

    StructuredSDFGBuilder(const std::string& name, FunctionType type);

    StructuredSDFGBuilder(const std::string& name, FunctionType type, const types::IType& return_type);

    StructuredSDFGBuilder(SDFG& sdfg);

    StructuredSDFG& subject() const;

    /**
     * @deprecated the unique ptr required SDFGs were removed from the registry during modification to make sense.
     * This builder does not change the pointer. But this will release any references the builder has to the SDFG to end
     * any modification
     */
    std::unique_ptr<StructuredSDFG> move();

    void rename_container(const std::string& old_name, const std::string& new_name) const override;

    Element* find_element_by_id(const size_t& element_id) const;

    Sequence& add_sequence(
        Sequence& parent,
        const sdfg::control_flow::Assignments& assignments = {},
        const DebugInfo& debug_info = DebugInfo()
    );

    Sequence& add_sequence_before(
        Sequence& parent,
        ControlFlowNode& block,
        const sdfg::control_flow::Assignments& assignments = {},
        const DebugInfo& debug_info = DebugInfo()
    );

    Sequence& add_sequence_after(
        Sequence& parent,
        ControlFlowNode& block,
        const sdfg::control_flow::Assignments& assignments = {},
        const DebugInfo& debug_info = DebugInfo()
    );

    Sequence& add_sequence_at(
        Sequence& parent,
        InsertionPoint insertion_point,
        const DebugInfo& debug_info = DebugInfo(),
        const sdfg::control_flow::Assignments* assignments = nullptr
    );

    [[deprecated("use method with explicit assignments instead")]]
    std::pair<Sequence&, Transition&>
    add_sequence_before(Sequence& parent, ControlFlowNode& block, const DebugInfo& debug_info = DebugInfo());

    void remove_from_parent(ControlFlowNode& child);

    void remove_child(Sequence& parent, size_t index);

    void remove_children(Sequence& parent);

    void move_child(Sequence& source, size_t source_index, Sequence& target);

    void move_child(Sequence& source, size_t source_index, Sequence& target, size_t target_index);

    void move_children(Sequence& source, Sequence& target);

    void move_children(Sequence& source, Sequence& target, size_t target_index);

    Sequence& hoist_root();

    Block& add_block(
        Sequence& parent,
        const sdfg::control_flow::Assignments& assignments = {},
        const DebugInfo& debug_info = DebugInfo()
    );

    /**
     * @param data_flow_graph copies the contents of this into the new block
     **/
    Block& add_block(
        Sequence& parent,
        const data_flow::DataFlowGraph& data_flow_graph,
        const sdfg::control_flow::Assignments& assignments = {},
        const DebugInfo& debug_info = DebugInfo()
    );

    Block& add_block_before(
        Sequence& parent,
        ControlFlowNode& child,
        const sdfg::control_flow::Assignments& assignments = {},
        const DebugInfo& debug_info = DebugInfo()
    );

    Block& add_block_before(
        Sequence& parent,
        ControlFlowNode& child,
        data_flow::DataFlowGraph& data_flow_graph,
        const sdfg::control_flow::Assignments& assignments = {},
        const DebugInfo& debug_info = DebugInfo()
    );

    Block& add_block_after(
        Sequence& parent,
        ControlFlowNode& child,
        const sdfg::control_flow::Assignments& assignments = {},
        const DebugInfo& debug_info = DebugInfo()
    );

    Block& add_block_after(
        Sequence& parent,
        ControlFlowNode& child,
        data_flow::DataFlowGraph& data_flow_graph,
        const sdfg::control_flow::Assignments& assignments = {},
        const DebugInfo& debug_info = DebugInfo()
    );

    Block& add_block_at(
        Sequence& parent,
        InsertionPoint insertion_point,
        const DebugInfo& debug_info = DebugInfo(),
        const sdfg::control_flow::Assignments* assignments = nullptr
    );

    [[deprecated("use method with explicit assignments instead")]]
    std::pair<Block&, Transition&>
    add_block_before(Sequence& parent, ControlFlowNode& child, const DebugInfo& debug_info = DebugInfo());

    [[deprecated("use method with explicit assignments instead")]]
    std::pair<Block&, Transition&> add_block_before(
        Sequence& parent,
        ControlFlowNode& child,
        data_flow::DataFlowGraph& data_flow_graph,
        const DebugInfo& debug_info = DebugInfo()
    );

    [[deprecated("use method with explicit assignments instead")]]
    std::pair<
        Block&,
        Transition&> add_block_after(Sequence& parent, ControlFlowNode& child, const DebugInfo& debug_info = DebugInfo());

    [[deprecated("use method with explicit assignments instead")]]
    std::pair<Block&, Transition&> add_block_after(
        Sequence& parent,
        ControlFlowNode& child,
        data_flow::DataFlowGraph& data_flow_graph,
        const DebugInfo& debug_info = DebugInfo()
    );

    IfElse& add_if_else(
        Sequence& parent,
        const sdfg::control_flow::Assignments& assignments = {},
        const DebugInfo& debug_info = DebugInfo()
    );

    IfElse& add_if_else_before(
        Sequence& parent,
        ControlFlowNode& child,
        const sdfg::control_flow::Assignments& assignments = {},
        const DebugInfo& debug_info = DebugInfo()
    );

    IfElse& add_if_else_after(
        Sequence& parent,
        ControlFlowNode& child,
        const sdfg::control_flow::Assignments& assignments = {},
        const DebugInfo& debug_info = DebugInfo()
    );

    [[deprecated("use method with explicit assignments instead")]]
    std::pair<IfElse&, Transition&>
    add_if_else_before(Sequence& parent, ControlFlowNode& child, const DebugInfo& debug_info = DebugInfo());

    Sequence& add_case(IfElse& scope, const sdfg::symbolic::Condition cond, const DebugInfo& debug_info = DebugInfo());

    void remove_case(IfElse& scope, size_t i, const DebugInfo& debug_info = DebugInfo());

    While& add_while(
        Sequence& parent,
        const sdfg::control_flow::Assignments& assignments = {},
        const DebugInfo& debug_info = DebugInfo()
    );

    For& add_for(
        Sequence& parent,
        const symbolic::Symbol indvar,
        const symbolic::Condition condition,
        const symbolic::Expression init,
        const symbolic::Expression update,
        const sdfg::control_flow::Assignments& assignments = {},
        const DebugInfo& debug_info = DebugInfo()
    );

    For& add_for_before(
        Sequence& parent,
        ControlFlowNode& child,
        const symbolic::Symbol indvar,
        const symbolic::Condition condition,
        const symbolic::Expression init,
        const symbolic::Expression update,
        const sdfg::control_flow::Assignments& assignments = {},
        const DebugInfo& debug_info = DebugInfo()
    );

    For& add_for_after(
        Sequence& parent,
        ControlFlowNode& child,
        const symbolic::Symbol indvar,
        const symbolic::Condition condition,
        const symbolic::Expression init,
        const symbolic::Expression update,
        const sdfg::control_flow::Assignments& assignments = {},
        const DebugInfo& debug_info = DebugInfo()
    );

    For& add_for_at(
        Sequence& parent,
        InsertionPoint insertion_point,
        const symbolic::Symbol indvar,
        const symbolic::Condition condition,
        const symbolic::Expression init,
        const symbolic::Expression update,
        const ScheduleType& schedule_type,
        const DebugInfo& debug_info = DebugInfo(),
        const sdfg::control_flow::Assignments* assignments = nullptr
    );

    Map& add_map(
        Sequence& parent,
        const symbolic::Symbol indvar,
        const symbolic::Condition condition,
        const symbolic::Expression init,
        const symbolic::Expression update,
        const ScheduleType& schedule_type,
        const sdfg::control_flow::Assignments& assignments = {},
        const DebugInfo& debug_info = DebugInfo()
    );

    Map& add_map_after(
        Sequence& parent,
        ControlFlowNode& child,
        const symbolic::Symbol indvar,
        const symbolic::Condition condition,
        const symbolic::Expression init,
        const symbolic::Expression update,
        const ScheduleType& schedule_type,
        const sdfg::control_flow::Assignments& assignments = {},
        const DebugInfo& debug_info = DebugInfo()
    );

    Map& add_map_before(
        Sequence& parent,
        ControlFlowNode& child,
        const symbolic::Symbol indvar,
        const symbolic::Condition condition,
        const symbolic::Expression init,
        const symbolic::Expression update,
        const ScheduleType& schedule_type,
        const sdfg::control_flow::Assignments& assignments = {},
        const DebugInfo& debug_info = DebugInfo()
    );

    Map& add_map_at(
        Sequence& parent,
        InsertionPoint insertion_point,
        const symbolic::Symbol indvar,
        const symbolic::Condition condition,
        const symbolic::Expression init,
        const symbolic::Expression update,
        const ScheduleType& schedule_type,
        const DebugInfo& debug_info = DebugInfo(),
        const sdfg::control_flow::Assignments* assignments = nullptr
    );

    Reduce& add_reduce(
        Sequence& parent,
        const symbolic::Symbol indvar,
        const symbolic::Condition condition,
        const symbolic::Expression init,
        const symbolic::Expression update,
        const std::vector<structured_control_flow::ReductionInfo>& reductions,
        const ScheduleType& schedule_type,
        const sdfg::control_flow::Assignments& assignments = {},
        const DebugInfo& debug_info = DebugInfo()
    );

    Reduce& add_reduce_before(
        Sequence& parent,
        ControlFlowNode& child,
        const symbolic::Symbol indvar,
        const symbolic::Condition condition,
        const symbolic::Expression init,
        const symbolic::Expression update,
        const std::vector<structured_control_flow::ReductionInfo>& reductions,
        const ScheduleType& schedule_type,
        const sdfg::control_flow::Assignments& assignments = {},
        const DebugInfo& debug_info = DebugInfo()
    );

    Reduce& add_reduce_after(
        Sequence& parent,
        ControlFlowNode& child,
        const symbolic::Symbol indvar,
        const symbolic::Condition condition,
        const symbolic::Expression init,
        const symbolic::Expression update,
        const std::vector<structured_control_flow::ReductionInfo>& reductions,
        const ScheduleType& schedule_type,
        const sdfg::control_flow::Assignments& assignments = {},
        const DebugInfo& debug_info = DebugInfo()
    );

    Continue& add_continue(
        Sequence& parent,
        const sdfg::control_flow::Assignments& assignments = {},
        const DebugInfo& debug_info = DebugInfo()
    );

    Break& add_break(
        Sequence& parent,
        const sdfg::control_flow::Assignments& assignments = {},
        const DebugInfo& debug_info = DebugInfo()
    );

    Return& add_return(
        Sequence& parent,
        const std::string& data,
        const sdfg::control_flow::Assignments& assignments = {},
        const DebugInfo& debug_info = DebugInfo()
    );

    Return& add_constant_return(
        Sequence& parent,
        const std::string& data,
        const types::IType& type,
        const sdfg::control_flow::Assignments& assignments = {},
        const DebugInfo& debug_info = DebugInfo()
    );

    For& convert_while(
        Sequence& parent,
        While& loop,
        const symbolic::Symbol indvar,
        const symbolic::Condition condition,
        const symbolic::Expression init,
        const symbolic::Expression update
    );

    Map& convert_for(Sequence& parent, For& loop);

    Reduce& convert_for_to_reduce(
        Sequence& parent, For& loop, const std::vector<structured_control_flow::ReductionInfo>& reductions
    );

    void update_if_else_condition(IfElse& if_else, size_t branch, const symbolic::Condition cond);

    void update_loop(
        StructuredLoop& loop,
        const symbolic::Symbol indvar,
        const symbolic::Condition condition,
        const symbolic::Expression init,
        const symbolic::Expression update
    );

    void update_schedule_type(StructuredLoop& loop, const ScheduleType& schedule_type);

    /***** Section: Dataflow Graph *****/

    data_flow::AccessNode& add_access(
        structured_control_flow::Block& block, const std::string& data, const DebugInfo& debug_info = DebugInfo()
    );

    data_flow::ConstantNode& add_constant(
        structured_control_flow::Block& block,
        const std::string& data,
        const types::IType& type,
        const DebugInfo& debug_info = DebugInfo()
    );

    data_flow::Tasklet& add_tasklet(
        structured_control_flow::Block& block,
        const data_flow::TaskletCode code,
        const std::string& output,
        const std::vector<std::string>& inputs,
        const DebugInfo& debug_info = DebugInfo()
    );

    data_flow::Memlet& add_memlet(
        structured_control_flow::Block& block,
        data_flow::DataFlowNode& src,
        const std::string& src_conn,
        data_flow::DataFlowNode& dst,
        const std::string& dst_conn,
        const data_flow::Subset& subset,
        const types::IType& base_type,
        const DebugInfo& debug_info
    );

    data_flow::Memlet& add_computational_memlet(
        structured_control_flow::Block& block,
        data_flow::AccessNode& src,
        data_flow::CodeNode& dst,
        const std::string& dst_conn,
        const data_flow::Subset& subset,
        const types::IType& base_type,
        const DebugInfo& debug_info = DebugInfo()
    );

    data_flow::Memlet& add_computational_memlet(
        structured_control_flow::Block& block,
        data_flow::CodeNode& src,
        const std::string& src_conn,
        data_flow::AccessNode& dst,
        const data_flow::Subset& subset,
        const types::IType& base_type,
        const DebugInfo& debug_info = DebugInfo()
    );

    data_flow::Memlet& add_computational_memlet(
        structured_control_flow::Block& block,
        data_flow::AccessNode& src,
        data_flow::Tasklet& dst,
        const std::string& dst_conn,
        const data_flow::Subset& subset,
        const DebugInfo& debug_info = DebugInfo()
    );

    data_flow::Memlet& add_computational_memlet(
        structured_control_flow::Block& block,
        data_flow::Tasklet& src,
        const std::string& src_conn,
        data_flow::AccessNode& dst,
        const data_flow::Subset& subset,
        const DebugInfo& debug_info = DebugInfo()
    );

    data_flow::Memlet& add_reference_memlet(
        structured_control_flow::Block& block,
        data_flow::AccessNode& src,
        data_flow::AccessNode& dst,
        const data_flow::Subset& subset,
        const types::IType& base_type,
        const DebugInfo& debug_info = DebugInfo()
    );

    data_flow::Memlet& add_dereference_memlet(
        structured_control_flow::Block& block,
        data_flow::AccessNode& src,
        data_flow::AccessNode& dst,
        bool derefs_src,
        const types::IType& base_type,
        const DebugInfo& debug_info = DebugInfo()
    );

    template<typename T, typename... Args>
    data_flow::LibraryNode&
    add_library_node(structured_control_flow::Block& block, const DebugInfo& debug_info, Args... arguments) {
        static_assert(std::is_base_of<data_flow::LibraryNode, T>::value, "T must be a subclass of data_flow::LibraryNode");

        auto& dataflow = block.dataflow();
        auto vertex = boost::add_vertex(dataflow.graph_);
        auto node =
            std::unique_ptr<T>(new T(this->new_element_id(), debug_info, vertex, dataflow, std::move(arguments)...));
        auto res = dataflow.nodes_.insert({vertex, std::move(node)});

        return static_cast<data_flow::LibraryNode&>(*(res.first->second));
    }

    data_flow::DataFlowNode& copy_node(structured_control_flow::Block& block, const data_flow::DataFlowNode& node) {
        auto& dataflow = block.dataflow();
        auto vertex = boost::add_vertex(dataflow.graph_);
        auto node_clone = node.clone(this->new_element_id(), vertex, dataflow);
        auto res = dataflow.nodes_.insert({vertex, std::move(node_clone)});
        return *res.first->second;
    };

    void remove_memlet(structured_control_flow::Block& block, const data_flow::Memlet& edge);

    void remove_node(structured_control_flow::Block& block, const data_flow::DataFlowNode& node);

    /**
     * Removes a code node and all its input and output edges. Will remove unused access nodes at the ends of those
     * edges. Caller must ensure that targets of output edges remain valid with their input edges removed.
     */
    void clear_code_node_legacy(structured_control_flow::Block& block, const data_flow::CodeNode& node);
    /**
     * Removes a code node and maybe some of its inputs as well.
     */
    void clear_access_node_legacy(structured_control_flow::Block& block, const data_flow::AccessNode& node);

    /**
     * Shorthand for @ref clear_node with the node itself being the only one to ignore side-effects on
     * @return number of removed nodes
     */
    int clear_node(structured_control_flow::Block& block, const data_flow::DataFlowNode& node);

    /**
     * New function to remove all unneeded edges and nodes transitively.
     * Checks for side effects and other output uses. WARNING: it will remove output edges that are unused,
     * but potentially leave the produce alive, if it has other uses. Do not use, until we support nodes with
     * unpopulated outputs or are sure that it cannot happen as a result.
     * @param block the dataflow graph in which to remove
     * @param node the original node, where the start the removal process
     * @param ignore_side_effects list of nodes, for which we can ignore side effects
     *   (for access nodes, this would be writes, for code_nodes it would depend on their side_effect flag)
     *   The original node itself must also be in that list, to ignore any of its side effects!
     * @return number of removed nodes
     */
    int clear_node(
        structured_control_flow::Block& block,
        const data_flow::DataFlowNode& node,
        const std::unordered_set<const data_flow::DataFlowNode*>& ignore_side_effects
    );

    int clear_ptr_borrow_edge(Block& block, const data_flow::Memlet& edge);

    void merge_siblings(data_flow::AccessNode& in_node);

    /**
     * Walks all sink nodes of a block's dataflow graph and merges access nodes that refer to the same
     * container into a single sink access node, redirecting their incoming memlets.
     */
    void merge_sinks(structured_control_flow::Block& block);
};

} // namespace builder
} // namespace sdfg
