#pragma once

#include <utility>

#include "sdfg/builder/function_builder.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/sdfg.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace builder {

class SDFGBuilder : public FunctionBuilder {
private:
    std::unique_ptr<SDFG> sdfg_;

protected:
    Function& function() const override;

public:
    SDFGBuilder(std::unique_ptr<SDFG>& sdfg);

    SDFGBuilder(const std::string& name, FunctionType type);

    SDFGBuilder(const std::string& name, FunctionType type, const types::IType& return_type);

    SDFG& subject() const;

    sdfg::control_flow::State* get_non_const_state(const sdfg::control_flow::State* state) const {
        for (auto& entry : sdfg_->states_) {
            if (entry.second.get() == state) {
                return entry.second.get();
            }
        }
        return nullptr;
    }

    std::unique_ptr<SDFG> move();

    void rename_container(const std::string& old_name, const std::string& new_name) const override;

    /***** Section: Control-Flow Graph *****/

    control_flow::State& add_state(bool is_start_state = false, const DebugInfo& debug_info = DebugInfo());

    control_flow::State& add_state_before(
        const control_flow::State& state, bool is_start_state = false, const DebugInfo& debug_info = DebugInfo()
    );

    control_flow::State& add_state_after(
        const control_flow::State& state, bool connect_states = true, const DebugInfo& debug_info = DebugInfo()
    );

    control_flow::ReturnState& add_return_state(const std::string& data, const DebugInfo& debug_info = DebugInfo());

    control_flow::ReturnState& add_return_state_after(
        const control_flow::State& state, const std::string& data, const DebugInfo& debug_info = DebugInfo()
    );

    control_flow::ReturnState& add_constant_return_state(
        const std::string& data, const types::IType& type, const DebugInfo& debug_info = DebugInfo()
    );

    control_flow::ReturnState& add_constant_return_state_after(
        const control_flow::State& state,
        const std::string& data,
        const types::IType& type,
        const DebugInfo& debug_info = DebugInfo()
    );

    control_flow::InterstateEdge&
    add_edge(const control_flow::State& src, const control_flow::State& dst, const DebugInfo& debug_info = DebugInfo());

    control_flow::InterstateEdge& add_edge(
        const control_flow::State& src,
        const control_flow::State& dst,
        const symbolic::Condition condition,
        const DebugInfo& debug_info = DebugInfo()
    );

    control_flow::InterstateEdge& add_edge(
        const control_flow::State& src,
        const control_flow::State& dst,
        const control_flow::Assignments& assignments,
        const DebugInfo& debug_info = DebugInfo()
    );

    control_flow::InterstateEdge& add_edge(
        const control_flow::State& src,
        const control_flow::State& dst,
        const control_flow::Assignments& assignments,
        const symbolic::Condition condition,
        const DebugInfo& debug_info = DebugInfo()
    );

    void remove_edge(const control_flow::InterstateEdge& edge);

    std::tuple<control_flow::State&, control_flow::State&, control_flow::State&> add_loop(
        const control_flow::State& state,
        sdfg::symbolic::Symbol iterator,
        sdfg::symbolic::Expression init,
        sdfg::symbolic::Condition cond,
        sdfg::symbolic::Expression update,
        const DebugInfo& debug_info = DebugInfo()
    );

    /***** Section: Dataflow Graph *****/

    data_flow::AccessNode&
    add_access(control_flow::State& state, const std::string& data, const DebugInfo& debug_info = DebugInfo());

    data_flow::ConstantNode& add_constant(
        control_flow::State& state,
        const std::string& data,
        const types::IType& type,
        const DebugInfo& debug_info = DebugInfo()
    );

    data_flow::Tasklet& add_tasklet(
        control_flow::State& state,
        const data_flow::TaskletCode code,
        const std::string& output,
        const std::vector<std::string>& inputs,
        const DebugInfo& debug_info = DebugInfo()
    );

    data_flow::Memlet& add_memlet(
        control_flow::State& state,
        data_flow::DataFlowNode& src,
        const std::string& src_conn,
        data_flow::DataFlowNode& dst,
        const std::string& dst_conn,
        const data_flow::Subset& subset,
        const types::IType& base_type,
        const DebugInfo& debug_info
    );

    data_flow::Memlet& add_computational_memlet(
        control_flow::State& state,
        data_flow::AccessNode& src,
        data_flow::Tasklet& dst,
        const std::string& dst_conn,
        const data_flow::Subset& subset,
        const types::IType& base_type,
        const DebugInfo& debug_info = DebugInfo()
    );

    data_flow::Memlet& add_computational_memlet(
        control_flow::State& state,
        data_flow::Tasklet& src,
        const std::string& src_conn,
        data_flow::AccessNode& dst,
        const data_flow::Subset& subset,
        const types::IType& base_type,
        const DebugInfo& debug_info = DebugInfo()
    );

    data_flow::Memlet& add_computational_memlet(
        control_flow::State& state,
        data_flow::AccessNode& src,
        data_flow::Tasklet& dst,
        const std::string& dst_conn,
        const data_flow::Subset& subset,
        const DebugInfo& debug_info = DebugInfo()
    );

    data_flow::Memlet& add_computational_memlet(
        control_flow::State& state,
        data_flow::Tasklet& src,
        const std::string& src_conn,
        data_flow::AccessNode& dst,
        const data_flow::Subset& subset,
        const DebugInfo& debug_info = DebugInfo()
    );

    data_flow::Memlet& add_computational_memlet(
        control_flow::State& state,
        data_flow::AccessNode& src,
        data_flow::LibraryNode& dst,
        const std::string& dst_conn,
        const data_flow::Subset& subset,
        const types::IType& base_type,
        const DebugInfo& debug_info = DebugInfo()
    );

    data_flow::Memlet& add_computational_memlet(
        control_flow::State& state,
        data_flow::LibraryNode& src,
        const std::string& src_conn,
        data_flow::AccessNode& dst,
        const data_flow::Subset& subset,
        const types::IType& base_type,
        const DebugInfo& debug_info = DebugInfo()
    );

    data_flow::Memlet& add_reference_memlet(
        control_flow::State& state,
        data_flow::AccessNode& src,
        data_flow::AccessNode& dst,
        const data_flow::Subset& subset,
        const types::IType& base_type,
        const DebugInfo& debug_info = DebugInfo()
    );

    data_flow::Memlet& add_dereference_memlet(
        control_flow::State& state,
        data_flow::AccessNode& src,
        data_flow::AccessNode& dst,
        bool derefs_src,
        const types::IType& base_type,
        const DebugInfo& debug_info = DebugInfo()
    );

    template<typename T, typename... Args>
    data_flow::LibraryNode& add_library_node(control_flow::State& state, const DebugInfo& debug_info, Args... arguments) {
        static_assert(std::is_base_of<data_flow::LibraryNode, T>::value, "T must be a subclass of data_flow::LibraryNode");

        auto& dataflow = state.dataflow();
        auto vertex = boost::add_vertex(dataflow.graph_);
        dataflow.invalidate_topological_order();
        auto node = std::unique_ptr<T>(new T(this->new_element_id(), debug_info, vertex, dataflow, arguments...));
        auto res = dataflow.nodes_.insert({vertex, std::move(node)});

        return static_cast<data_flow::LibraryNode&>(*(res.first->second));
    };
};

} // namespace builder
} // namespace sdfg
