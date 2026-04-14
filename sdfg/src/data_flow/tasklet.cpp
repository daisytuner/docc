#include "sdfg/data_flow/tasklet.h"

#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace data_flow {

Tasklet::Tasklet(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    DataFlowGraph& parent,
    const TaskletCode code,
    const std::string& output,
    const std::vector<std::string>& inputs
)
    : CodeNode(element_id, debug_info, vertex, parent, {output}, inputs), code_(code) {};

void Tasklet::validate(const Function& function) const {
    auto& graph = this->get_parent();

    // Validate: inputs match arity
    if (arity(this->code_) != this->inputs_.size()) {
        throw InvalidSDFGException(
            "Tasklet (Code: " + std::to_string(this->code_) + "): Invalid number of inputs. Expected " +
            std::to_string(arity(this->code_)) + ", got " + std::to_string(this->inputs_.size())
        );
    }

    // Validate: inputs match type of operation
    for (auto& iedge : graph.in_edges(*this)) {
        auto input_type = iedge.result_type(function);
        if (is_integer(this->code_) && !types::is_integer(input_type->primitive_type())) {
            throw InvalidSDFGException(
                "Tasklet (Code: " + std::to_string(this->code_) + "): Integer operation with non-integer input type"
            );
        }
        if (is_floating_point(this->code_) && !types::is_floating_point(input_type->primitive_type())) {
            throw InvalidSDFGException(
                "Tasklet (Code: " + std::to_string(this->code_) + "): Floating point operation with integer input type"
            );
        }
    }

    // Validate: Edges
    if (graph.in_degree(*this) != this->inputs_.size()) {
        throw InvalidSDFGException(
            "Tasklet (Code: " + std::to_string(this->code_) +
            "): Number of input edges does not match number of inputs."
        );
    }
    if (graph.out_degree(*this) != this->outputs_.size()) {
        throw InvalidSDFGException(
            "Tasklet (Code: " + std::to_string(this->code_) +
            "): Number of output edges does not match number of outputs."
        );
    }
}

TaskletCode Tasklet::code() const { return this->code_; };


bool Tasklet::is_assign() const { return this->code_ == TaskletCode::assign; }

bool Tasklet::is_trivial(const Function& function) const {
    if (!this->is_assign()) {
        return false;
    }

    auto& graph = this->get_parent();
    auto& iedge = *graph.in_edges(*this).begin();
    auto& oedge = *graph.out_edges(*this).begin();
    auto input_type = iedge.result_type(function);
    auto output_type = oedge.result_type(function);

    return input_type->primitive_type() == output_type->primitive_type();
}

bool Tasklet::is_cast(const Function& function) const {
    if (!this->is_assign()) {
        return false;
    }

    auto& graph = this->get_parent();
    auto& iedge = *graph.in_edges(*this).begin();
    auto& oedge = *graph.out_edges(*this).begin();
    auto input_type = iedge.result_type(function);
    auto output_type = oedge.result_type(function);

    return input_type->primitive_type() != output_type->primitive_type();
}

bool Tasklet::is_zext(const Function& function) const {
    if (!this->is_assign()) {
        return false;
    }

    auto& graph = this->get_parent();
    auto& iedge = *graph.in_edges(*this).begin();
    auto& oedge = *graph.out_edges(*this).begin();
    auto input_type = iedge.result_type(function);
    auto output_type = oedge.result_type(function);

    if (!types::is_unsigned(input_type->primitive_type()) || !types::is_unsigned(output_type->primitive_type())) {
        return false;
    }
    if (types::bit_width(output_type->primitive_type()) <= types::bit_width(input_type->primitive_type())) {
        return false;
    }

    return true;
}

bool Tasklet::is_sext(const Function& function) const {
    if (!this->is_assign()) {
        return false;
    }

    auto& graph = this->get_parent();
    auto& iedge = *graph.in_edges(*this).begin();
    auto& oedge = *graph.out_edges(*this).begin();
    auto input_type = iedge.result_type(function);
    auto output_type = oedge.result_type(function);

    if (types::is_unsigned(input_type->primitive_type()) || types::is_unsigned(output_type->primitive_type())) {
        return false;
    }
    if (types::bit_width(output_type->primitive_type()) <= types::bit_width(input_type->primitive_type())) {
        return false;
    }

    return true;
}

bool Tasklet::is_trunc(const Function& function) const {
    if (!this->is_assign()) {
        return false;
    }

    auto& graph = this->get_parent();
    auto& iedge = *graph.in_edges(*this).begin();
    auto& oedge = *graph.out_edges(*this).begin();
    auto input_type = iedge.result_type(function);
    auto output_type = oedge.result_type(function);

    if (!types::is_integer(input_type->primitive_type()) || !types::is_integer(output_type->primitive_type())) {
        return false;
    }
    if (types::is_unsigned(input_type->primitive_type()) != types::is_unsigned(output_type->primitive_type())) {
        return false;
    }
    if (types::bit_width(output_type->primitive_type()) >= types::bit_width(input_type->primitive_type())) {
        return false;
    }

    return true;
}

bool Tasklet::is_fptoui(const Function& function) const {
    if (!this->is_assign()) {
        return false;
    }

    auto& graph = this->get_parent();
    auto& iedge = *graph.in_edges(*this).begin();
    auto& oedge = *graph.out_edges(*this).begin();
    auto input_type = iedge.result_type(function);
    auto output_type = oedge.result_type(function);

    if (!types::is_floating_point(input_type->primitive_type()) || !types::is_unsigned(output_type->primitive_type())) {
        return false;
    }

    return true;
}

bool Tasklet::is_fptosi(const Function& function) const {
    if (!this->is_assign()) {
        return false;
    }

    auto& graph = this->get_parent();
    auto& iedge = *graph.in_edges(*this).begin();
    auto& oedge = *graph.out_edges(*this).begin();
    auto input_type = iedge.result_type(function);
    auto output_type = oedge.result_type(function);

    if (!types::is_floating_point(input_type->primitive_type()) || !types::is_signed(output_type->primitive_type())) {
        return false;
    }

    return true;
}

bool Tasklet::is_uitofp(const Function& function) const {
    if (!this->is_assign()) {
        return false;
    }

    auto& graph = this->get_parent();
    auto& iedge = *graph.in_edges(*this).begin();
    auto& oedge = *graph.out_edges(*this).begin();
    auto input_type = iedge.result_type(function);
    auto output_type = oedge.result_type(function);

    if (!types::is_unsigned(input_type->primitive_type()) || !types::is_floating_point(output_type->primitive_type())) {
        return false;
    }

    return true;
}

bool Tasklet::is_sitofp(const Function& function) const {
    if (!this->is_assign()) {
        return false;
    }

    auto& graph = this->get_parent();
    auto& iedge = *graph.in_edges(*this).begin();
    auto& oedge = *graph.out_edges(*this).begin();
    auto input_type = iedge.result_type(function);
    auto output_type = oedge.result_type(function);

    if (!types::is_signed(input_type->primitive_type()) || !types::is_floating_point(output_type->primitive_type())) {
        return false;
    }

    return true;
}

bool Tasklet::is_fpext(const Function& function) const {
    if (!this->is_assign()) {
        return false;
    }

    auto& graph = this->get_parent();
    auto& iedge = *graph.in_edges(*this).begin();
    auto& oedge = *graph.out_edges(*this).begin();
    auto input_type = iedge.result_type(function);
    auto output_type = oedge.result_type(function);

    if (!types::is_floating_point(input_type->primitive_type()) ||
        !types::is_floating_point(output_type->primitive_type())) {
        return false;
    }
    if (types::bit_width(output_type->primitive_type()) <= types::bit_width(input_type->primitive_type())) {
        return false;
    }

    return true;
}

bool Tasklet::is_fptrunc(const Function& function) const {
    if (!this->is_assign()) {
        return false;
    }

    auto& graph = this->get_parent();
    auto& iedge = *graph.in_edges(*this).begin();
    auto& oedge = *graph.out_edges(*this).begin();
    auto input_type = iedge.result_type(function);
    auto output_type = oedge.result_type(function);

    if (!types::is_floating_point(input_type->primitive_type()) ||
        !types::is_floating_point(output_type->primitive_type())) {
        return false;
    }
    if (types::bit_width(output_type->primitive_type()) >= types::bit_width(input_type->primitive_type())) {
        return false;
    }

    return true;
}

const std::string& Tasklet::output() const { return this->outputs_[0]; };

std::unique_ptr<DataFlowNode> Tasklet::clone(size_t element_id, const graph::Vertex vertex, DataFlowGraph& parent)
    const {
    return std::unique_ptr<Tasklet>(
        new Tasklet(element_id, this->debug_info_, vertex, parent, this->code_, this->outputs_.at(0), this->inputs_)
    );
};

void Tasklet::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {}

EdgeRemoveOption Tasklet::can_remove_out_edge(const data_flow::DataFlowGraph& graph, const Memlet* memlet) const {
    if (graph.out_edges_for_connector(*this, memlet->src_conn()).size() > 1) {
        return EdgeRemoveOption::Trivially;
    } else {
        // trick to allow removal of Tasklets, without having general mark-and-sweep logic to checkl tha that all
        // outputs are
        return EdgeRemoveOption::RemoveNodeAfter;
    }
}

} // namespace data_flow
} // namespace sdfg
