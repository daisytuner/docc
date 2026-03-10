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
};

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

    // Define and initialize inputs
    for (auto& iedge : graph.in_edges_by_connector(this->node_)) {
        auto& src = dynamic_cast<const data_flow::AccessNode&>(iedge->src());
        std::string src_name = this->language_extension_.access_node(src);

        std::string conn = iedge->dst_conn();
        auto conn_type = iedge->result_type(this->function_);
        if (conn_type->type_id() == types::TypeID::Array ||
            (conn_type->type_id() == types::TypeID::Structure &&
             !static_cast<const types::Structure&>(*conn_type).is_pointer_like())) {
            // Handle array and structure types
            stream << this->language_extension_.declaration(conn, *conn_type) << ";" << std::endl;
            stream << "memcpy(" << "&" << conn << ", " << "&" << src_name
                   << this->language_extension_.subset(iedge->base_type(), iedge->subset()) << ", sizeof " << conn
                   << ");" << std::endl;
        } else {
            stream << this->language_extension_.declaration(conn, *conn_type);
            stream << " = ";

            // Reinterpret cast for opaque pointers
            if (dynamic_cast<const data_flow::ConstantNode*>(&src)) {
                stream << src_name;
            } else {
                if (iedge->base_type().type_id() == types::TypeID::Pointer) {
                    stream << "(" << this->language_extension_.type_cast(src_name, iedge->base_type()) << ")";
                } else {
                    stream << src_name;
                }
            }

            stream << this->language_extension_.subset(iedge->base_type(), iedge->subset()) << ";";
            stream << std::endl;
        }
    }

    // Define outputs
    for (auto& oedge : graph.out_edges_by_connector(this->node_)) {
        if (std::find(this->node_.inputs().begin(), this->node_.inputs().end(), oedge->src_conn()) !=
            this->node_.inputs().end()) {
            continue;
        }

        auto& dst = dynamic_cast<const data_flow::AccessNode&>(oedge->dst());
        std::string dst_name = this->language_extension_.access_node(dst);

        std::string conn = oedge->src_conn();
        auto conn_type = oedge->result_type(this->function_);
        if (conn_type->type_id() == types::TypeID::Array || conn_type->type_id() == types::TypeID::Structure) {
            // Handle array and structure types
            stream << this->language_extension_.declaration(conn, *conn_type) << ";" << std::endl;
            stream << "memcpy(" << "&" << conn << ", " << "&" << dst_name
                   << this->language_extension_.subset(oedge->base_type(), oedge->subset()) << ", sizeof " << conn
                   << ");" << std::endl;
        } else {
            stream << this->language_extension_.declaration(conn, *conn_type);
            stream << " = ";

            // Reinterpret cast for opaque pointers
            if (oedge->base_type().type_id() == types::TypeID::Pointer) {
                stream << "(" << this->language_extension_.type_cast(dst_name, oedge->base_type()) << ")";
            } else {
                stream << dst_name;
            }

            stream << this->language_extension_.subset(oedge->base_type(), oedge->subset()) << ";";
            stream << std::endl;
        }
    }

    stream << std::endl;

    this->dispatch_code(stream, globals_stream, library_snippet_factory);

    stream << std::endl;

    for (auto& oedge : this->data_flow_graph_.out_edges_by_connector(this->node_)) {
        auto& dst = dynamic_cast<const data_flow::AccessNode&>(oedge->dst());
        if (this->function_.is_external(dst.data())) {
            continue;
        }

        std::string dst_name = this->language_extension_.access_node(dst);

        auto result_type = oedge->result_type(this->function_);
        if (result_type->type_id() == types::TypeID::Array || result_type->type_id() == types::TypeID::Structure) {
            stream << "memcpy(" << "&" << dst_name
                   << this->language_extension_.subset(oedge->base_type(), oedge->subset()) << ", " << "&"
                   << oedge->src_conn() << ", sizeof " << oedge->src_conn() << ");" << std::endl;
        } else {
            stream << dst_name;
            stream << this->language_extension_.subset(oedge->base_type(), oedge->subset()) << " = ";
            stream << oedge->src_conn();
            stream << ";" << std::endl;
        }
    }

    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
}

InstrumentationInfo LibraryNodeDispatcher::instrumentation_info() const {
    return InstrumentationInfo(node_.element_id(), ElementType_Unknown, TargetType_SEQUENTIAL);
};


} // namespace codegen
} // namespace sdfg
