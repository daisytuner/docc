#include "sdfg/codegen/dispatchers/block_dispatcher.h"

#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/codegen/dispatchers/node_dispatcher_registry.h"
#include "sdfg/codegen/instrumentation/instrumentation_info.h"
#include "sdfg/types/structure.h"

namespace sdfg {
namespace codegen {

BlockDispatcher::BlockDispatcher(
    LanguageExtension& language_extension,
    StructuredSDFG& sdfg,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::Block& node,
    InstrumentationPlan& instrumentation_plan,
    ArgCapturePlan& arg_capture_plan
)
    : sdfg::codegen::
          NodeDispatcher(language_extension, sdfg, analysis_manager, node, instrumentation_plan, arg_capture_plan),
      node_(node) {

      };

void BlockDispatcher::dispatch_node(
    PrettyPrinter& main_stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory
) {
    if (node_.dataflow().nodes().empty()) {
        return;
    }

    DataFlowDispatcher dispatcher(this->language_extension_, sdfg_, node_.dataflow(), instrumentation_plan_);
    dispatcher.dispatch(main_stream, globals_stream, library_snippet_factory);
}

AssignmentDispatcher::AssignmentDispatcher(
    LanguageExtension& language_extension,
    StructuredSDFG& sdfg,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::AssignmentBlock& node,
    InstrumentationPlan& instrumentation_plan,
    ArgCapturePlan& arg_capture_plan
)
    : NodeDispatcher(language_extension, sdfg, analysis_manager, node, instrumentation_plan, arg_capture_plan),
      node_(node) {}

void AssignmentDispatcher::dispatch_node(
    PrettyPrinter& main_stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory
) {
    auto& assignments = node_.assignments();
    if (!assignments.empty()) {
        main_stream << "{" << std::endl;
        main_stream.changeIndent(+4);
        for (auto assign : assignments) {
            main_stream << language_extension_.expression(assign.first) << " = "
                        << language_extension_.expression(assign.second) << ";" << std::endl;
        }
        main_stream.changeIndent(-4);
        main_stream << "}" << std::endl;
    }
}

DataFlowDispatcher::DataFlowDispatcher(
    LanguageExtension& language_extension,
    const Function& sdfg,
    const data_flow::DataFlowGraph& data_flow_graph,
    const InstrumentationPlan& instrumentation_plan
)
    : language_extension_(language_extension), function_(sdfg), data_flow_graph_(data_flow_graph),
      instrumentation_plan_(instrumentation_plan) {

      };

void DataFlowDispatcher::
    dispatch(PrettyPrinter& stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory) {
    // Dispatch code nodes in topological order
    auto nodes = this->data_flow_graph_.topological_sort();
    for (auto& node : nodes) {
        if (auto tasklet = dynamic_cast<const data_flow::Tasklet*>(node)) {
            this->dispatch_tasklet(stream, *tasklet);
        } else if (auto libnode = dynamic_cast<const data_flow::LibraryNode*>(node)) {
            this->dispatch_library_node(stream, globals_stream, library_snippet_factory, *libnode);
        } else if (auto access_node = dynamic_cast<const data_flow::AccessNode*>(node)) {
            for (auto& edge : this->data_flow_graph_.out_edges(*access_node)) {
                if (edge.type() == data_flow::MemletType::Reference) {
                    this->dispatch_ref(stream, edge);
                } else if (edge.type() == data_flow::MemletType::Dereference_Src) {
                    this->dispatch_deref_src(stream, edge);
                } else if (edge.type() == data_flow::MemletType::Dereference_Dst) {
                    this->dispatch_deref_dst(stream, edge);
                }
            }
        } else {
            throw InvalidSDFGException("Codegen: Node type not supported");
        }
    }
};

void DataFlowDispatcher::dispatch_ref(PrettyPrinter& stream, const data_flow::Memlet& memlet) {
    stream << "{" << std::endl;
    stream.setIndent(stream.indent() + 4);

    auto& src = dynamic_cast<const data_flow::AccessNode&>(memlet.src());
    auto& dst = dynamic_cast<const data_flow::AccessNode&>(memlet.dst());

    auto& subset = memlet.subset();
    auto& base_type = memlet.base_type();

    stream << this->language_extension_.access_node(dst);
    stream << " = ";

    std::string src_name = this->language_extension_.access_node(src);
    if (dynamic_cast<const data_flow::ConstantNode*>(&src)) {
        stream << src_name;
        stream << this->language_extension_.subset(base_type, subset);
    } else {
        if (base_type.type_id() == types::TypeID::Pointer && !subset.empty()) {
            stream << "&";
            stream << "(" + this->language_extension_.type_cast(src_name, base_type) + ")";
            stream << this->language_extension_.subset(base_type, subset);
        } else {
            stream << "&";
            stream << src_name;
            stream << this->language_extension_.subset(base_type, subset);
        }
    }

    stream << ";";
    stream << std::endl;

    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
};

void DataFlowDispatcher::dispatch_deref_src(PrettyPrinter& stream, const data_flow::Memlet& memlet) {
    stream << "{" << std::endl;
    stream.setIndent(stream.indent() + 4);

    auto& src = dynamic_cast<const data_flow::AccessNode&>(memlet.src());
    auto& dst = dynamic_cast<const data_flow::AccessNode&>(memlet.dst());
    auto& dst_type = this->function_.type(dst.data());
    auto& base_type = static_cast<const types::Pointer&>(memlet.base_type());

    switch (dst_type.type_id()) {
        // first-class values
        case types::TypeID::Scalar:
        case types::TypeID::Pointer: {
            stream << this->language_extension_.access_node(dst);
            stream << " = ";
            stream << "*";

            std::string src_name = this->language_extension_.access_node(src);
            stream << "(" << this->language_extension_.type_cast(src_name, base_type) << ")";
            break;
        }
        // composite values
        case types::TypeID::Array:
        case types::TypeID::Structure: {
            // Memcpy
            std::string dst_name = this->language_extension_.access_node(dst);
            std::string src_name = this->language_extension_.access_node(src);
            stream << "memcpy(" << "&" << dst_name;
            stream << ", ";
            stream << "(" << src_name << ")";
            stream << ", ";
            stream << "sizeof " << dst_name;
            stream << ")";
            break;
        }
        case types::TypeID::Reference:
        case types::TypeID::Function: {
            throw InvalidSDFGException("Memlet: Dereference memlets cannot have reference or function destination types"
            );
        }
        case types::TypeID::Tensor: {
            throw InvalidSDFGException(
                "Memlet: Dereference memlets cannot have tensor destination types. Tensors must be lowered to pointers "
                "before code generation."
            );
        }
    }
    stream << ";" << std::endl;

    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
};

void DataFlowDispatcher::dispatch_deref_dst(PrettyPrinter& stream, const data_flow::Memlet& memlet) {
    stream << "{" << std::endl;
    stream.setIndent(stream.indent() + 4);

    auto& src = dynamic_cast<const data_flow::AccessNode&>(memlet.src());
    auto& dst = dynamic_cast<const data_flow::AccessNode&>(memlet.dst());
    const sdfg::types::IType* src_type;
    if (auto const_node = dynamic_cast<const data_flow::ConstantNode*>(&src)) {
        src_type = &const_node->type();
    } else {
        src_type = &this->function_.type(src.data());
    }
    auto& base_type = static_cast<const types::Pointer&>(memlet.base_type());

    switch (src_type->type_id()) {
        // first-class values
        case types::TypeID::Scalar:
        case types::TypeID::Pointer: {
            stream << "*";
            std::string dst_name = this->language_extension_.access_node(dst);
            stream << "(" << this->language_extension_.type_cast(dst_name, base_type) << ")";
            stream << " = ";

            stream << this->language_extension_.access_node(src);
            break;
        }
        // composite values
        case types::TypeID::Array:
        case types::TypeID::Structure: {
            // Memcpy
            std::string src_name = this->language_extension_.access_node(src);
            std::string dst_name = this->language_extension_.access_node(dst);
            stream << "memcpy(";
            stream << "(" << dst_name << ")";
            stream << ", ";
            stream << "&" << src_name;
            stream << ", ";
            stream << "sizeof " << src_name;
            stream << ")";
            break;
        }
        case types::TypeID::Function:
        case types::TypeID::Reference: {
            throw InvalidSDFGException("Memlet: Dereference memlets cannot have source of type Function or Reference");
        }
        case types::TypeID::Tensor: {
            throw InvalidSDFGException(
                "Memlet: Dereference memlets cannot have tensor source types. Tensors must be lowered to pointers "
                "before code generation."
            );
        }
    }
    stream << ";" << std::endl;

    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
};

void DataFlowDispatcher::dispatch_tasklet(PrettyPrinter& stream, const data_flow::Tasklet& tasklet) {
    stream << "{" << std::endl;
    stream.setIndent(stream.indent() + 4);

    bool is_unsigned = data_flow::is_unsigned(tasklet.code());

    for (auto* iedge : this->data_flow_graph_.in_edges_by_connector(tasklet)) {
        auto& src = dynamic_cast<const data_flow::AccessNode&>(iedge->src());
        std::string src_name = this->language_extension_.access_node(src);

        std::string conn = iedge->dst_conn();
        auto conn_type = iedge->result_type(this->function_);
        auto& conn_type_scalar = dynamic_cast<const types::Scalar&>(*conn_type);
        if (is_unsigned) {
            types::Scalar conn_type_unsigned(types::as_unsigned(conn_type_scalar.primitive_type()));
            stream << this->language_extension_.declaration(conn, conn_type_unsigned);
            stream << " = ";
        } else {
            stream << this->language_extension_.declaration(conn, conn_type_scalar);
            stream << " = ";
        }

        // Reinterpret cast for opaque pointers
        if (iedge->base_type().type_id() == types::TypeID::Pointer) {
            stream << "(" << this->language_extension_.type_cast(src_name, iedge->base_type()) << ")";
        } else {
            stream << src_name;
        }

        stream << this->language_extension_.subset(iedge->base_type(), iedge->subset()) << ";";
        stream << std::endl;
    }

    auto& oedge = *this->data_flow_graph_.out_edges(tasklet).begin();
    std::string out_conn = oedge.src_conn();
    auto out_conn_type = oedge.result_type(this->function_);
    auto& out_conn_type_scalar = dynamic_cast<const types::Scalar&>(*out_conn_type);
    if (is_unsigned) {
        types::Scalar out_conn_type_unsigned(types::as_unsigned(out_conn_type_scalar.primitive_type()));
        stream << this->language_extension_.declaration(out_conn, out_conn_type_unsigned);
        stream << ";" << std::endl;
    } else {
        stream << this->language_extension_.declaration(out_conn, out_conn_type_scalar);
        stream << ";" << std::endl;
    }


    stream << std::endl;
    stream << out_conn << " = ";
    stream << this->language_extension_.tasklet(tasklet) << ";" << std::endl;
    stream << std::endl;

    // Write back
    for (auto& oedge : this->data_flow_graph_.out_edges_by_connector(tasklet)) {
        auto& dst = dynamic_cast<const data_flow::AccessNode&>(oedge->dst());

        std::string dst_name = this->language_extension_.access_node(dst);

        // Reinterpret cast for opaque pointers
        if (oedge->base_type().type_id() == types::TypeID::Pointer) {
            stream << "(" << this->language_extension_.type_cast(dst_name, oedge->base_type()) << ")";
        } else {
            stream << dst_name;
        }

        stream << this->language_extension_.subset(oedge->base_type(), oedge->subset()) << " = ";
        stream << oedge->src_conn();
        stream << ";" << std::endl;
    }

    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
};

void DataFlowDispatcher::dispatch_library_node(
    PrettyPrinter& stream,
    PrettyPrinter& globals_stream,
    CodeSnippetFactory& library_snippet_factory,
    const data_flow::LibraryNode& libnode
) {
    auto dispatcher_id = libnode.code().value() + "::" + libnode.implementation_type().value();
    auto dispatcher_fn = LibraryNodeDispatcherRegistry::instance().get_library_node_dispatcher(dispatcher_id);
    if (dispatcher_fn) {
        auto dispatcher = dispatcher_fn(this->language_extension_, this->function_, this->data_flow_graph_, libnode);
        auto applied = dispatcher->begin_node(stream);

        bool should_instrument = this->instrumentation_plan_.should_instrument(libnode);
        std::optional<InstrumentationInfo> instrument_info;
        if (should_instrument) {
            instrument_info = dispatcher->instrumentation_info();
            this->instrumentation_plan_
                .begin_instrumentation(libnode, stream, language_extension_, instrument_info.value());
        }

        dispatcher->dispatch(stream, globals_stream, library_snippet_factory);

        if (should_instrument) {
            this->instrumentation_plan_
                .end_instrumentation(libnode, stream, language_extension_, instrument_info.value());
        }
        dispatcher->end_node(stream, applied);
    } else {
        throw std::runtime_error("No library node dispatcher found for library node id: " + dispatcher_id);
    }
};

std::string resolve_input_edge_to_expression(
    const data_flow::Memlet& iedge, const Function& function, LanguageExtension& language_extension
) {
    auto& src = iedge.src();
    std::string src_name;
    if (auto* access_node = dynamic_cast<const data_flow::AccessNode*>(&src)) {
        src_name = language_extension.access_node(*access_node);
    } else {
        throw InvalidSDFGException(
            "Edge does not start at access-node: " + std::to_string(iedge.element_id()) + ", but #" +
            std::to_string(src.element_id())
        );
    }
    std::string expr;
    if (dynamic_cast<const data_flow::ConstantNode*>(&src)) {
        expr = src_name;
    } else if (iedge.base_type().type_id() == types::TypeID::Pointer) {
        expr = "(" + language_extension.type_cast(src_name, iedge.base_type()) + ")";
    } else {
        expr = src_name;
    }
    expr += language_extension.subset(iedge.base_type(), iedge.subset());
    return expr;
}

LibraryNodeDispatcher::LibraryNodeDispatcher(
    LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const data_flow::LibraryNode& node
)
    : language_extension_(language_extension), function_(function), data_flow_graph_(data_flow_graph), node_(node) {};

void LibraryNodeDispatcher::
    dispatch(PrettyPrinter& stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory) {
    auto& graph = this->node_.get_parent();

    stream << "{" << std::endl;
    stream.setIndent(stream.indent() + 4);

    std::vector<DispatchInput> inputs;
    auto input_count = node_.inputs().size();
    inputs.reserve(input_count);
    auto in_edges = data_flow_graph_.in_edges_by_connector(node_);
    for (auto i = 0; i < input_count; ++i) {
        auto* iedge = in_edges.at(i);
        if (!iedge) {
            throw InvalidSDFGException(
                "On libNode #" + std::to_string(node_.element_id()) + ", input " + std::to_string(i) + ":" +
                node_.input(i) + " is unconnected!"
            );
        }
        auto expr = resolve_input_edge_to_expression(*iedge, this->function_, language_extension_);
        inputs.emplace_back(expr, *iedge, false);
    }

    std::vector<DispatchOutput> outputs;
    outputs.reserve(this->node_.outputs().size());

    // Define outputs
    for (auto i = 0; i < this->node_.outputs().size(); ++i) {
        auto& oconn = this->node_.output(i);
        auto oedges = data_flow_graph_.out_edges_for_connector(node_, oconn);
        std::unique_ptr<types::IType> oconn_type;
        for (auto& oedge : oedges) {
            auto edge_type = oedge->result_type(this->function_);
            if (!oconn_type) {
                oconn_type = std::move(edge_type);
            } else {
                if (oconn_type != edge_type) {
                    throw InvalidSDFGException(
                        "Output connector " + oconn + " on #" + std::to_string(node_.element_id()) +
                        " has different types"
                    );
                }
            }
        }
        outputs.emplace_back(nullptr, std::move(oconn_type), oedges.size() > 0);
    }

    CodegenOutput codegen = {
        stream,
        globals_stream,
        library_snippet_factory,
        language_extension_,
    };
    this->dispatch_code_with_edges(codegen, inputs, outputs);

    int i = 0;
    for (auto o_idx = 0; o_idx < node_.outputs().size(); o_idx++) {
        auto& oconn = node_.output(o_idx);
        auto oedges = data_flow_graph_.out_edges_for_connector(node_, oconn);
        auto& output = outputs.at(o_idx);
        if (output.local_name) {
            for (auto* edge : oedges) {
                copy_output(codegen, output, *edge);
            }
        } else if (oedges.size() > 0) {
            throw InvalidSDFGException(
                "Output connector " + oconn + " on #" + std::to_string(node_.element_id()) +
                " has no output, but out-edges"
            );
        }
    }

    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
}

void LibraryNodeDispatcher::require_locally_modifiable_var(CodegenOutput& out, DispatchInput& input) const {
    if (input.is_locally_modifiable) {
        return;
    }

    // create a local copy, if not already done

    auto& iedge = input.edge;
    std::string conn = iedge.dst_conn();
    auto conn_type = iedge.result_type(this->function_);

    out.stream << out.language_extension.declaration(conn, *conn_type);

    if (conn_type->type_id() == types::TypeID::Array ||
        (conn_type->type_id() == types::TypeID::Structure &&
         !static_cast<const types::Structure&>(*conn_type).is_pointer_like())) {
        // Handle array and structure types

        out.stream << ";" << std::endl;
        out.stream << "memcpy(" << "&" << conn << ", " << "&" << input.expr << ", sizeof " << conn << ");" << std::endl;
    } else {
        out.stream << " = " << input.expr << ";" << std::endl;
    }
    input.is_locally_modifiable = true;
}

void LibraryNodeDispatcher::pre_allocate_output(CodegenOutput& out, DispatchOutput& output, const std::string& var_name)
    const {
    out.stream << out.language_extension.declaration(var_name, *output.out_type) << ";" << std::endl;
    output.local_name = &var_name;
}

void LibraryNodeDispatcher::register_output(DispatchOutput& output, const std::string& result_identifier) const {
    output.local_name = &result_identifier;
}

void LibraryNodeDispatcher::copy_output(CodegenOutput& out, const DispatchOutput& output, const data_flow::Memlet& oedge)
    const {
    auto* dst = dynamic_cast<const data_flow::AccessNode*>(&oedge.dst());
    if (!dst) {
        throw InvalidSDFGException(
            "Output " + oedge.src_conn() + " does not end at access-node: " + std::to_string(oedge.element_id()) +
            ", but #" + std::to_string(dst->element_id())
        );
    }
    auto dst_name = this->language_extension_.access_node(*dst);

    auto& conn_type = output.out_type;
    auto conn = output.local_name;
    if (!conn) {
        throw InvalidSDFGException(
            "Output " + oedge.src_conn() + " does not exist on #" + std::to_string(node_.element_id())
        );
    }

    auto expr = dst_name;
    expr += this->language_extension_.subset(oedge.base_type(), oedge.subset());

    if (conn_type->type_id() == types::TypeID::Array ||
        (conn_type->type_id() == types::TypeID::Structure &&
         !static_cast<const types::Structure&>(*conn_type).is_pointer_like())) {
        // Handle array and structure types
        out.stream << "memcpy(" << "&" << expr << ", " << "&" << *output.local_name << ", sizeof " << conn

                   << ");" << std::endl;
    } else {
        out.stream << expr << " = " << *output.local_name << ";" << std::endl;
    }
}

void LibraryNodeDispatcher::dispatch_code_with_edges(
    CodegenOutput& out, std::vector<DispatchInput>& inputs, std::vector<DispatchOutput>& outputs
) {
    for (auto& in : inputs) {
        require_locally_modifiable_var(out, in);
    }

    for (auto o_idx = 0; o_idx < node_.outputs().size(); o_idx++) {
        auto& output = outputs.at(o_idx);
        auto& oconn = node_.output(o_idx);
        pre_allocate_output(out, output, oconn);
    }
    if (!inputs.empty() || !outputs.empty()) {
        out.stream << std::endl;
    }

    this->dispatch_code(out.stream, out.globals_stream, out.library_snippet_factory);

    out.stream << std::endl;
}

InstrumentationInfo LibraryNodeDispatcher::instrumentation_info() const {
    return InstrumentationInfo(
        node_.element_id(), std::string(node_.element_type()) + ":::" + node_.code().value(), TargetType_SEQUENTIAL
    );
};


} // namespace codegen
} // namespace sdfg
