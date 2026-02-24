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

protected:
    Function& function() const override;

public:
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

    [[deprecated("use method with explicit assignments instead")]]
    std::pair<Sequence&, Transition&>
    add_sequence_before(Sequence& parent, ControlFlowNode& block, const DebugInfo& debug_info = DebugInfo());

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

    void update_if_else_condition(IfElse& if_else, size_t branch, const symbolic::Condition cond);

    void update_loop(
        StructuredLoop& loop,
        const symbolic::Symbol indvar,
        const symbolic::Condition condition,
        const symbolic::Expression init,
        const symbolic::Expression update
    );

    void update_schedule_type(Map& map, const ScheduleType& schedule_type);

    [[deprecated("use ScopeAnalysis instead")]]
    Sequence& parent(const ControlFlowNode& node);

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
        data_flow::Tasklet& dst,
        const std::string& dst_conn,
        const data_flow::Subset& subset,
        const types::IType& base_type,
        const DebugInfo& debug_info = DebugInfo()
    );

    data_flow::Memlet& add_computational_memlet(
        structured_control_flow::Block& block,
        data_flow::Tasklet& src,
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

    data_flow::Memlet& add_computational_memlet(
        structured_control_flow::Block& block,
        data_flow::LibraryNode& src,
        const std::string& src_conn,
        data_flow::AccessNode& dst,
        const data_flow::Subset& subset,
        const types::IType& base_type,
        const DebugInfo& debug_info = DebugInfo()
    );

    data_flow::Memlet& add_computational_memlet(
        structured_control_flow::Block& block,
        data_flow::AccessNode& src,
        data_flow::LibraryNode& dst,
        const std::string& dst_conn,
        const data_flow::Subset& subset,
        const types::IType& base_type,
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
    };

    data_flow::DataFlowNode& copy_node(structured_control_flow::Block& block, const data_flow::DataFlowNode& node) {
        auto& dataflow = block.dataflow();
        auto vertex = boost::add_vertex(dataflow.graph_);
        auto node_clone = node.clone(this->new_element_id(), vertex, dataflow);
        auto res = dataflow.nodes_.insert({vertex, std::move(node_clone)});
        return *res.first->second;
    };

    void remove_memlet(structured_control_flow::Block& block, const data_flow::Memlet& edge);

    void remove_node(structured_control_flow::Block& block, const data_flow::DataFlowNode& node);

    void clear_node(structured_control_flow::Block& block, const data_flow::CodeNode& node);

    void clear_node(structured_control_flow::Block& block, const data_flow::AccessNode& node);

    void merge_siblings(data_flow::AccessNode& in_node);
};

} // namespace builder
} // namespace sdfg
