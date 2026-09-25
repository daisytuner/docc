#include "sdfg/builder/sdfg_builder.h"

#include "sdfg/codegen/utils.h"
#include "sdfg/types/utils.h"

namespace sdfg {
namespace builder {

Function& SDFGBuilder::function() const {
    return static_cast<Function&>(*this->sdfg_);
};

SDFGBuilder::SDFGBuilder(std::unique_ptr<SDFG>& sdfg)
    : FunctionBuilder(), sdfg_(std::move(sdfg)) {

      };

SDFGBuilder::SDFGBuilder(const std::string& name, FunctionType type)
    : FunctionBuilder(), sdfg_(new SDFG(name, type)) {

      };

SDFGBuilder::SDFGBuilder(const std::string& name, FunctionType type, const types::IType& return_type)
    : FunctionBuilder(), sdfg_(new SDFG(name, type, return_type)) {

      };

SDFG& SDFGBuilder::subject() const {
    return *this->sdfg_;
};

std::unique_ptr<SDFG> SDFGBuilder::move() {
    return std::move(this->sdfg_);
};

void SDFGBuilder::rename_container(const std::string& old_name, const std::string& new_name) const {
    FunctionBuilder::rename_container(old_name, new_name);

    for (auto& entry : this->sdfg_->states_) {
        entry.second->replace(symbolic::symbol(old_name), symbolic::symbol(new_name));
    }
    for (auto& entry : this->sdfg_->edges_) {
        entry.second->replace(symbolic::symbol(old_name), symbolic::symbol(new_name));
    }
}

/***** Section: Control-Flow Graph *****/

control_flow::State& SDFGBuilder::add_state(bool is_start_state, const DebugInfo& debug_info) {
    auto vertex = boost::add_vertex(this->sdfg_->graph_);
    auto res = this->sdfg_->states_.insert(
        {vertex,
         std::unique_ptr<control_flow::State>(new control_flow::State(this->new_element_id(), debug_info, vertex))}
    );

    assert(res.second);
    (*res.first).second->dataflow_->parent_ = (*res.first).second.get();

    if (is_start_state) {
        this->sdfg_->start_state_ = (*res.first).second.get();
    }

    return *(*res.first).second;
};

control_flow::State& SDFGBuilder::
    add_state_before(const control_flow::State& state, bool is_start_state, const DebugInfo& debug_info) {
    auto& new_state = this->add_state(false, debug_info);

    std::vector<const control_flow::InterstateEdge*> to_redirect;
    for (auto& e : this->sdfg_->in_edges(state)) {
        to_redirect.push_back(&e);
    }

    // Redirect control-flow
    for (auto edge : to_redirect) {
        this->add_edge(edge->src(), new_state, edge->condition());

        auto desc = edge->edge();
        this->sdfg_->edges_.erase(desc);
        boost::remove_edge(desc, this->sdfg_->graph_);
    }
    this->add_edge(new_state, state);

    if (is_start_state) {
        this->sdfg_->start_state_ = &new_state;
    }

    return new_state;
};

control_flow::State& SDFGBuilder::
    add_state_after(const control_flow::State& state, bool connect_states, const DebugInfo& debug_info) {
    auto& new_state = this->add_state(false, debug_info);

    std::vector<const control_flow::InterstateEdge*> to_redirect;
    for (auto& e : this->sdfg_->out_edges(state)) {
        to_redirect.push_back(&e);
    }

    // Redirect control-flow
    for (auto& edge : to_redirect) {
        this->add_edge(new_state, edge->dst(), edge->condition());

        auto desc = edge->edge();
        this->sdfg_->edges_.erase(desc);
        boost::remove_edge(desc, this->sdfg_->graph_);
    }
    if (connect_states) {
        this->add_edge(state, new_state);
    }

    return new_state;
};

control_flow::ReturnState& SDFGBuilder::add_return_state(const std::string& data, const DebugInfo& debug_info) {
    auto vertex = boost::add_vertex(this->sdfg_->graph_);
    auto res = this->sdfg_->states_.insert(
        {vertex,
         std::unique_ptr<
             control_flow::State>(new control_flow::ReturnState(this->new_element_id(), debug_info, vertex, data))}
    );

    assert(res.second);
    (*res.first).second->dataflow_->parent_ = (*res.first).second.get();

    return static_cast<control_flow::ReturnState&>(*(*res.first).second);
};

control_flow::ReturnState& SDFGBuilder::
    add_return_state_after(const control_flow::State& state, const std::string& data, const DebugInfo& debug_info) {
    auto& new_state = this->add_return_state(data, debug_info);

    std::vector<const control_flow::InterstateEdge*> to_redirect;
    for (auto& e : this->sdfg_->out_edges(state)) {
        to_redirect.push_back(&e);
    }

    // Redirect control-flow
    for (auto& edge : to_redirect) {
        this->add_edge(new_state, edge->dst(), edge->condition());

        auto desc = edge->edge();
        this->sdfg_->edges_.erase(desc);
        boost::remove_edge(desc, this->sdfg_->graph_);
    }
    this->add_edge(state, new_state);

    return new_state;
};

control_flow::ReturnState& SDFGBuilder::
    add_constant_return_state(const std::string& data, const types::IType& type, const DebugInfo& debug_info) {
    auto vertex = boost::add_vertex(this->sdfg_->graph_);
    auto res = this->sdfg_->states_.insert(
        {vertex,
         std::unique_ptr<
             control_flow::State>(new control_flow::ReturnState(this->new_element_id(), debug_info, vertex, data, type))
        }
    );

    assert(res.second);
    (*res.first).second->dataflow_->parent_ = (*res.first).second.get();

    return static_cast<control_flow::ReturnState&>(*(*res.first).second);
};

control_flow::ReturnState& SDFGBuilder::add_constant_return_state_after(
    const control_flow::State& state, const std::string& data, const types::IType& type, const DebugInfo& debug_info
) {
    auto& new_state = this->add_constant_return_state(data, type, debug_info);

    std::vector<const control_flow::InterstateEdge*> to_redirect;
    for (auto& e : this->sdfg_->out_edges(state)) {
        to_redirect.push_back(&e);
    }

    // Redirect control-flow
    for (auto& edge : to_redirect) {
        this->add_edge(new_state, edge->dst(), edge->condition());

        auto desc = edge->edge();
        this->sdfg_->edges_.erase(desc);
        boost::remove_edge(desc, this->sdfg_->graph_);
    }
    this->add_edge(state, new_state);

    return new_state;
};

control_flow::InterstateEdge& SDFGBuilder::
    add_edge(const control_flow::State& src, const control_flow::State& dst, const DebugInfo& debug_info) {
    return this->add_edge(src, dst, control_flow::Assignments{}, SymEngine::boolTrue, debug_info);
};

control_flow::InterstateEdge& SDFGBuilder::add_edge(
    const control_flow::State& src,
    const control_flow::State& dst,
    const symbolic::Condition condition,
    const DebugInfo& debug_info
) {
    return this->add_edge(src, dst, control_flow::Assignments{}, condition, debug_info);
};

control_flow::InterstateEdge& SDFGBuilder::add_edge(
    const control_flow::State& src,
    const control_flow::State& dst,
    const control_flow::Assignments& assignments,
    const DebugInfo& debug_info
) {
    return this->add_edge(src, dst, assignments, SymEngine::boolTrue, debug_info);
};

control_flow::InterstateEdge& SDFGBuilder::add_edge(
    const control_flow::State& src,
    const control_flow::State& dst,
    const control_flow::Assignments& assignments,
    const symbolic::Condition condition,
    const DebugInfo& debug_info
) {
    if (dynamic_cast<const control_flow::ReturnState*>(&src) != nullptr) {
        throw InvalidSDFGException("Cannot add edge from ReturnState");
    }

    for (auto& entry : assignments) {
        auto& lhs = entry.first;
        auto& type = this->function().type(lhs->get_name());
        if (type.type_id() == types::TypeID::Scalar) {
            if (!types::is_integer(type.primitive_type())) {
                throw InvalidSDFGException("Assignment - LHS: must be integer type");
            }
        } else if (type.type_id() == types::TypeID::Reference) {
            auto* reference = dynamic_cast<const sdfg::codegen::Reference*>(&type);
            assert(reference != nullptr);
            auto& referenced_type = reference->reference_type();
            if (referenced_type.type_id() != types::TypeID::Scalar) {
                throw InvalidSDFGException("Assignment - LHS: must be a reference to a scalar type");
            }
            if (!types::is_integer(referenced_type.primitive_type())) {
                throw InvalidSDFGException("Assignment - LHS: must be integer type");
            }
        } else {
            throw InvalidSDFGException("Assignment - LHS: must be scalar type or a reference thereof");
        }

        auto& rhs = entry.second;
        for (auto& atom : symbolic::atoms(rhs)) {
            if (symbolic::is_nullptr(atom)) {
                continue;
            }
            auto& atom_type = this->function().type(atom->get_name());

            // Scalar integers
            if (atom_type.type_id() == types::TypeID::Scalar) {
                if (!types::is_integer(atom_type.primitive_type())) {
                    throw InvalidSDFGException("Assignment - RHS: must evaluate to integer type");
                }
                continue;
            } else if (atom_type.type_id() == types::TypeID::Reference) {
                auto* reference = dynamic_cast<const sdfg::codegen::Reference*>(&atom_type);
                assert(reference != nullptr);
                auto& referenced_type = reference->reference_type();
                if (referenced_type.type_id() != types::TypeID::Scalar) {
                    throw InvalidSDFGException("Assignment - RHS: must be a reference to a scalar type");
                }
                if (!types::is_integer(referenced_type.primitive_type())) {
                    throw InvalidSDFGException("Assignment - RHS: must evaluate to integer type");
                }
                continue;
            } else if (atom_type.type_id() == types::TypeID::Pointer) {
                continue;
            } else {
                throw InvalidSDFGException("Assignment - RHS: must evaluate to scalar, reference or pointer type");
            }
        }
    }

    for (auto& atom : symbolic::atoms(condition)) {
        if (symbolic::is_nullptr(atom)) {
            continue;
        }
        auto& atom_type = this->function().type(atom->get_name());
        if (atom_type.type_id() != types::TypeID::Scalar && atom_type.type_id() != types::TypeID::Pointer) {
            throw InvalidSDFGException("Condition: must be integer type or pointer type");
        }
    }

    auto edge = boost::add_edge(src.vertex_, dst.vertex_, this->sdfg_->graph_);
    assert(edge.second);

    auto res = this->sdfg_->edges_.insert(
        {edge.first,
         std::unique_ptr<control_flow::InterstateEdge>(new control_flow::InterstateEdge(
             this->new_element_id(), debug_info, edge.first, src, dst, condition, assignments
         ))}
    );

    assert(res.second);

    return *(*res.first).second;
};

void SDFGBuilder::remove_edge(const control_flow::InterstateEdge& edge) {
    auto desc = edge.edge();
    this->sdfg_->edges_.erase(desc);

    boost::remove_edge(desc, this->sdfg_->graph_);
};

std::tuple<control_flow::State&, control_flow::State&, control_flow::State&> SDFGBuilder::add_loop(
    const control_flow::State& state,
    sdfg::symbolic::Symbol iterator,
    sdfg::symbolic::Expression init,
    sdfg::symbolic::Condition cond,
    sdfg::symbolic::Expression update,
    const DebugInfo& debug_info
) {
    // Init: iterator = init
    auto& init_state = this->add_state_after(state, true, debug_info);
    const graph::Edge init_edge_desc = (*this->sdfg_->in_edges(init_state).begin()).edge_;
    auto& init_edge = this->sdfg_->edges_[init_edge_desc];
    init_edge->assignments_.insert({iterator, init});

    // Final state
    auto& final_state = this->add_state_after(init_state, false, debug_info);

    // Init -> early_exit -> final
    auto& early_exit_state = this->add_state(false, debug_info);
    this->add_edge(init_state, early_exit_state, symbolic::Not(cond));
    this->add_edge(early_exit_state, final_state);

    // Init -> header -> body
    auto& header_state = this->add_state(false, debug_info);
    this->add_edge(init_state, header_state, cond);

    auto& body_state = this->add_state(false, debug_info);
    this->add_edge(header_state, body_state);

    auto& update_state = this->add_state(false, debug_info);
    this->add_edge(body_state, update_state, {{iterator, update}});

    // Back edge and exit edge
    this->add_edge(update_state, header_state, cond);
    this->add_edge(update_state, final_state, symbolic::Not(cond));

    return {init_state, body_state, final_state};
};

/***** Section: Dataflow Graph *****/

data_flow::AccessNode& SDFGBuilder::
    add_access(control_flow::State& state, const std::string& data, const DebugInfo& debug_info) {
    auto& dataflow = state.dataflow();
    auto vertex = boost::add_vertex(dataflow.graph_);
    auto res = dataflow.nodes_.insert(
        {vertex,
         std::unique_ptr<
             data_flow::AccessNode>(new data_flow::AccessNode(this->new_element_id(), debug_info, vertex, dataflow, data)
         )}
    );

    return static_cast<data_flow::AccessNode&>(*(res.first->second));
};

data_flow::ConstantNode& SDFGBuilder::add_constant(
    control_flow::State& state, const std::string& data, const types::IType& type, const DebugInfo& debug_info
) {
    auto& dataflow = state.dataflow();
    auto vertex = boost::add_vertex(dataflow.graph_);
    auto res = dataflow.nodes_.insert(
        {vertex,
         std::unique_ptr<data_flow::ConstantNode>(
             new data_flow::ConstantNode(this->new_element_id(), debug_info, vertex, dataflow, data, type)
         )}
    );

    return static_cast<data_flow::ConstantNode&>(*(res.first->second));
};

data_flow::Tasklet& SDFGBuilder::add_tasklet(
    control_flow::State& state,
    const data_flow::TaskletCode code,
    const std::string& output,
    const std::vector<std::string>& inputs,
    const DebugInfo& debug_info
) {
    auto& dataflow = state.dataflow();
    auto vertex = boost::add_vertex(dataflow.graph_);
    auto res = dataflow.nodes_.insert(
        {vertex,
         std::unique_ptr<data_flow::Tasklet>(
             new data_flow::Tasklet(this->new_element_id(), debug_info, vertex, dataflow, code, output, inputs)
         )}
    );

    return static_cast<data_flow::Tasklet&>(*(res.first->second));
};

data_flow::Memlet& SDFGBuilder::add_memlet(
    control_flow::State& state,
    data_flow::DataFlowNode& src,
    const std::string& src_conn,
    data_flow::DataFlowNode& dst,
    const std::string& dst_conn,
    const data_flow::Subset& subset,
    const types::IType& base_type,
    const DebugInfo& debug_info
) {
    auto& dataflow = state.dataflow();
    auto edge = boost::add_edge(src.vertex_, dst.vertex_, dataflow.graph_);
    auto res = dataflow.edges_.insert(
        {edge.first,
         std::unique_ptr<data_flow::Memlet>(new data_flow::Memlet(
             this->new_element_id(), debug_info, edge.first, dataflow, src, src_conn, dst, dst_conn, subset, base_type
         ))}
    );

    auto& memlet = static_cast<data_flow::Memlet&>(*(res.first->second));

    return memlet;
};

data_flow::Memlet& SDFGBuilder::add_computational_memlet(
    control_flow::State& state,
    data_flow::AccessNode& src,
    data_flow::Tasklet& dst,
    const std::string& dst_conn,
    const data_flow::Subset& subset,
    const types::IType& base_type,
    const DebugInfo& debug_info
) {
    return this->add_memlet(state, src, "void", dst, dst_conn, subset, base_type, debug_info);
};

data_flow::Memlet& SDFGBuilder::add_computational_memlet(
    control_flow::State& state,
    data_flow::Tasklet& src,
    const std::string& src_conn,
    data_flow::AccessNode& dst,
    const data_flow::Subset& subset,
    const types::IType& base_type,
    const DebugInfo& debug_info
) {
    return this->add_memlet(state, src, src_conn, dst, "void", subset, base_type, debug_info);
};

data_flow::Memlet& SDFGBuilder::add_computational_memlet(
    control_flow::State& state,
    data_flow::AccessNode& src,
    data_flow::Tasklet& dst,
    const std::string& dst_conn,
    const data_flow::Subset& subset,
    const DebugInfo& debug_info
) {
    const types::IType* src_type = nullptr;
    if (auto cnode = dynamic_cast<data_flow::ConstantNode*>(&src)) {
        src_type = &cnode->type();
    } else {
        src_type = &this->sdfg_->type(src.data());
    }
    auto base_type = types::infer_type(*this->sdfg_, *src_type, subset);
    if (base_type->type_id() != types::TypeID::Scalar) {
        throw InvalidSDFGException("Computational memlet must have a scalar type");
    }
    return this->add_memlet(state, src, "void", dst, dst_conn, subset, *src_type, debug_info);
};

data_flow::Memlet& SDFGBuilder::add_computational_memlet(
    control_flow::State& state,
    data_flow::Tasklet& src,
    const std::string& src_conn,
    data_flow::AccessNode& dst,
    const data_flow::Subset& subset,
    const DebugInfo& debug_info
) {
    auto& dst_type = this->function().type(dst.data());
    auto base_type = types::infer_type(this->function(), dst_type, subset);
    if (base_type->type_id() != types::TypeID::Scalar) {
        throw InvalidSDFGException("Computational memlet must have a scalar type");
    }
    return this->add_memlet(state, src, src_conn, dst, "void", subset, dst_type, debug_info);
};

data_flow::Memlet& SDFGBuilder::add_computational_memlet(
    control_flow::State& state,
    data_flow::AccessNode& src,
    data_flow::LibraryNode& dst,
    const std::string& dst_conn,
    const data_flow::Subset& subset,
    const types::IType& base_type,
    const DebugInfo& debug_info
) {
    return this->add_memlet(state, src, "void", dst, dst_conn, subset, base_type, debug_info);
};

data_flow::Memlet& SDFGBuilder::add_computational_memlet(
    control_flow::State& state,
    data_flow::LibraryNode& src,
    const std::string& src_conn,
    data_flow::AccessNode& dst,
    const data_flow::Subset& subset,
    const types::IType& base_type,
    const DebugInfo& debug_info
) {
    return this->add_memlet(state, src, src_conn, dst, "void", subset, base_type, debug_info);
};

data_flow::Memlet& SDFGBuilder::add_reference_memlet(
    control_flow::State& state,
    data_flow::AccessNode& src,
    data_flow::AccessNode& dst,
    const data_flow::Subset& subset,
    const types::IType& base_type,
    const DebugInfo& debug_info
) {
    return this->add_memlet(state, src, "void", dst, "ref", subset, base_type, debug_info);
};

data_flow::Memlet& SDFGBuilder::add_dereference_memlet(
    control_flow::State& state,
    data_flow::AccessNode& src,
    data_flow::AccessNode& dst,
    bool derefs_src,
    const types::IType& base_type,
    const DebugInfo& debug_info
) {
    if (derefs_src) {
        return this->add_memlet(state, src, "void", dst, "deref", {symbolic::zero()}, base_type, debug_info);
    } else {
        return this->add_memlet(state, src, "deref", dst, "void", {symbolic::zero()}, base_type, debug_info);
    }
};

} // namespace builder
} // namespace sdfg
