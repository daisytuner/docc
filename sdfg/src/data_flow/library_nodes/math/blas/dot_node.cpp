#include "sdfg/data_flow/library_nodes/math/blas/dot_node.h"
#include <stdexcept>
#include <string>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"

#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace math {
namespace blas {

DotNode::DotNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const data_flow::ImplementationType& implementation_type,
    const BLAS_Precision& precision,
    symbolic::Expression n,
    symbolic::Expression incx,
    symbolic::Expression incy
)
    : BLASNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_DOT,
          {"__out"},
          {"__x", "__y"},
          implementation_type,
          precision
      ),
      n_(n), incx_(incx), incy_(incy) {}

symbolic::Expression DotNode::n() const { return this->n_; };

symbolic::Expression DotNode::incx() const { return this->incx_; };

symbolic::Expression DotNode::incy() const { return this->incy_; };

symbolic::SymbolSet DotNode::symbols() const {
    symbolic::SymbolSet syms;

    for (auto& atom : symbolic::atoms(this->n_)) {
        syms.insert(atom);
    }
    for (auto& atom : symbolic::atoms(this->incx_)) {
        syms.insert(atom);
    }
    for (auto& atom : symbolic::atoms(this->incy_)) {
        syms.insert(atom);
    }

    return syms;
};

void DotNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    this->n_ = symbolic::subs(this->n_, old_expression, new_expression);
    this->incx_ = symbolic::subs(this->incx_, old_expression, new_expression);
    this->incy_ = symbolic::subs(this->incy_, old_expression, new_expression);
};

void DotNode::validate(const Function& function) const { BLASNode::validate(function); }

bool DotNode::expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& dataflow = this->get_parent();
    auto& block = static_cast<structured_control_flow::Block&>(*dataflow.get_parent());
    auto& parent = static_cast<structured_control_flow::Sequence&>(*block.get_parent());
    int index = parent.index(block);
    auto& transition = parent.at(index).second;

    const data_flow::Memlet* iedge_x = nullptr;
    const data_flow::Memlet* iedge_y = nullptr;
    for (const auto& iedge : dataflow.in_edges(*this)) {
        if (iedge.dst_conn() == "__x") {
            iedge_x = &iedge;
        } else if (iedge.dst_conn() == "__y") {
            iedge_y = &iedge;
        }
    }

    const data_flow::Memlet* oedge_res = nullptr;
    for (const auto& oedge : dataflow.out_edges(*this)) {
        if (oedge.src_conn() == "__out") {
            oedge_res = &oedge;
            break;
        }
    }

    // Check if legal
    auto& input_node_x = static_cast<const data_flow::AccessNode&>(iedge_x->src());
    auto& input_node_y = static_cast<const data_flow::AccessNode&>(iedge_y->src());
    auto& output_node_res = static_cast<const data_flow::AccessNode&>(oedge_res->dst());
    if (dataflow.in_degree(input_node_x) != 0 || dataflow.in_degree(input_node_y) != 0 ||
        dataflow.out_degree(output_node_res) != 0) {
        return false;
    }

    auto& new_sequence = builder.add_sequence_before(parent, block, transition.assignments(), block.debug_info());

    std::string loop_var = builder.find_new_name("_i");
    builder.add_container(loop_var, types::Scalar(types::PrimitiveType::UInt64));

    auto loop_indvar = symbolic::symbol(loop_var);
    auto loop_init = symbolic::integer(0);
    auto loop_condition = symbolic::Lt(loop_indvar, this->n_);
    auto loop_update = symbolic::add(loop_indvar, symbolic::integer(1));

    auto& loop =
        builder.add_for(new_sequence, loop_indvar, loop_condition, loop_init, loop_update, {}, block.debug_info());
    auto& body = loop.root();

    auto& new_block = builder.add_block(body);

    auto& res_in = builder.add_access(new_block, output_node_res.data());
    auto& res_out = builder.add_access(new_block, output_node_res.data());
    auto& x = builder.add_access(new_block, input_node_x.data());
    auto& y = builder.add_access(new_block, input_node_y.data());

    auto& tasklet = builder.add_tasklet(new_block, data_flow::TaskletCode::fp_fma, "__out", {"_in1", "_in2", "_in3"});

    builder.add_computational_memlet(
        new_block,
        x,
        tasklet,
        "_in1",
        {symbolic::mul(loop_indvar, this->incx_)},
        iedge_x->base_type(),
        iedge_x->debug_info()
    );
    builder.add_computational_memlet(
        new_block,
        y,
        tasklet,
        "_in2",
        {symbolic::mul(loop_indvar, this->incy_)},
        iedge_y->base_type(),
        iedge_y->debug_info()
    );
    builder
        .add_computational_memlet(new_block, res_in, tasklet, "_in3", {}, oedge_res->base_type(), oedge_res->debug_info());
    builder.add_computational_memlet(
        new_block, tasklet, "__out", res_out, {}, oedge_res->base_type(), oedge_res->debug_info()
    );

    // Clean up
    builder.remove_memlet(block, *iedge_x);
    builder.remove_memlet(block, *iedge_y);
    builder.remove_memlet(block, *oedge_res);
    builder.remove_node(block, input_node_x);
    builder.remove_node(block, input_node_y);
    builder.remove_node(block, output_node_res);
    builder.remove_node(block, *this);
    builder.remove_child(parent, index + 1);

    return true;
}

symbolic::Expression DotNode::flop() const {
    auto muls = this->n_;
    auto adds = symbolic::sub(this->n_, symbolic::one());
    return symbolic::add(muls, adds);
}

std::unique_ptr<data_flow::DataFlowNode> DotNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    auto node_clone = std::unique_ptr<DotNode>(new DotNode(
        element_id,
        this->debug_info(),
        vertex,
        parent,
        this->implementation_type_,
        this->precision_,
        this->n_,
        this->incx_,
        this->incy_
    ));
    return std::move(node_clone);
}

data_flow::PointerAccessType DotNode::pointer_access_type(int input_idx) const {
    if (input_idx == 0) {
        return data_flow::PointerAccessMeta::create_read_only(symbolic::mul(n_, incx_), true);
    } else if (input_idx == 1) {
        return data_flow::PointerAccessMeta::create_read_only(symbolic::mul(n_, incy_), true);
    } else {
        return BLASNode::pointer_access_type(input_idx);
    }
}

nlohmann::json DotNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const DotNode& gemm_node = static_cast<const DotNode&>(library_node);
    nlohmann::json j;

    serializer::JSONSerializer serializer;
    j["code"] = gemm_node.code().value();
    j["precision"] = gemm_node.precision();
    j["n"] = serializer.expression(gemm_node.n());
    j["incx"] = serializer.expression(gemm_node.incx());
    j["incy"] = serializer.expression(gemm_node.incy());

    return j;
}

data_flow::LibraryNode& DotNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    // Assertions for required fields
    assert(j.contains("element_id"));
    assert(j.contains("code"));
    assert(j.contains("debug_info"));

    auto code = j["code"].get<std::string>();
    if (code != LibraryNodeType_DOT.value()) {
        throw std::runtime_error("Invalid library node code");
    }

    // Extract debug info using JSONSerializer
    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    auto precision = j.at("precision").get<BLAS_Precision>();
    auto n = symbolic::parse(j.at("n"));
    auto incx = symbolic::parse(j.at("incx"));
    auto incy = symbolic::parse(j.at("incy"));

    auto implementation_type = j.at("implementation_type").get<std::string>();

    return builder.add_library_node<DotNode>(parent, debug_info, implementation_type, precision, n, incx, incy);
}

DotNodeDispatcher_BLAS::DotNodeDispatcher_BLAS(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const DotNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void DotNodeDispatcher_BLAS::dispatch_code_with_edges(
    codegen::CodegenOutput& out,
    std::vector<codegen::DispatchInput>& inputs,
    std::vector<codegen::DispatchOutput>& outputs
) {
    auto& dot_node = static_cast<const DotNode&>(this->node_);

    sdfg::types::Scalar base_type(types::PrimitiveType::Void);
    BLAS_Precision precision = dot_node.precision();
    switch (precision) {
        case BLAS_Precision::h:
            base_type = types::Scalar(types::PrimitiveType::Half);
            break;
        case BLAS_Precision::s:
            base_type = types::Scalar(types::PrimitiveType::Float);
            break;
        case BLAS_Precision::d:
            base_type = types::Scalar(types::PrimitiveType::Double);
            break;
        default:
            throw std::runtime_error("Invalid BLAS_Precision value");
    }

    out.library_snippet_factory.require_dependency(BLASLibDependency::instance());

    auto& output = outputs.at(0);
    pre_allocate_output(out, output, dot_node.output(0));

    out.stream << *output.local_name << " = ";
    out.stream << "cblas_" << BLAS_Precision_to_string(precision) << "dot(";
    out.stream.changeIndent(+4);
    out.stream << this->language_extension_.expression(dot_node.n());
    out.stream << ", ";
    out.stream << inputs.at(0).expr;
    out.stream << ", ";
    out.stream << this->language_extension_.expression(dot_node.incx());
    out.stream << ", ";
    out.stream << inputs.at(1).expr;
    out.stream << ", ";
    out.stream << this->language_extension_.expression(dot_node.incy());
    out.stream.changeIndent(-4);
    out.stream << ");" << std::endl;
}


} // namespace blas
} // namespace math
} // namespace sdfg
