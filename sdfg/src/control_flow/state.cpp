#include "sdfg/control_flow/state.h"

#include "sdfg/sdfg.h"

namespace sdfg {
namespace control_flow {

State::State(size_t element_id, const DebugInfo& debug_info, const graph::Vertex vertex)
    : Element(element_id, debug_info), vertex_(vertex) {
    this->dataflow_ = std::make_unique<data_flow::DataFlowGraph>();
};

void State::validate(const Function& function) const {
    this->dataflow_->validate(function);
};

graph::Vertex State::vertex() const {
    return this->vertex_;
};

const data_flow::DataFlowGraph& State::dataflow() const {
    return *this->dataflow_;
};

data_flow::DataFlowGraph& State::dataflow() {
    return *this->dataflow_;
};

void State::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    this->dataflow_->replace(old_expression, new_expression);
};

ReturnState::ReturnState(size_t element_id, const DebugInfo& debug_info, const graph::Vertex vertex, const std::string& data)
    : State(element_id, debug_info, vertex), data_(data), type_(nullptr) {};

ReturnState::ReturnState(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    const std::string& data,
    const types::IType& type
)
    : State(element_id, debug_info, vertex), data_(data), type_(type.clone()) {};

const std::string& ReturnState::data() const {
    return this->data_;
};

const types::IType& ReturnState::type() const {
    return *(this->type_);
};

bool ReturnState::is_data() const {
    return type_ == nullptr;
};

bool ReturnState::is_constant() const {
    return type_ != nullptr;
};

void ReturnState::validate(const Function& function) const {
    State::validate(function);

    auto& sdfg = static_cast<const SDFG&>(function);
    if (sdfg.out_degree(*this) > 0) {
        throw InvalidSDFGException("ReturnState must not have outgoing transitions");
    }

    if (is_data()) {
        if (data_ != "" && !function.exists(data_)) {
            throw InvalidSDFGException("Return node with data '" + data_ + "' does not correspond to any container");
        }
        if (type_ != nullptr) {
            throw InvalidSDFGException("Return node with data cannot have a type");
        }
    } else if (is_constant()) {
        if (function.exists(data_)) {
            throw InvalidSDFGException(
                "Return node with constant data '" + data_ + "' cannot correspond to any container"
            );
        }
        if (type_ == nullptr) {
            throw InvalidSDFGException("Return node with constant data must have a type");
        }
    } else {
        throw InvalidSDFGException("Return node must be either data or constant");
    }
}

void ReturnState::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    State::replace(old_expression, new_expression);

    if (this->data_ == old_expression->__str__()) {
        this->data_ = new_expression->__str__();
    }
};

} // namespace control_flow
} // namespace sdfg
