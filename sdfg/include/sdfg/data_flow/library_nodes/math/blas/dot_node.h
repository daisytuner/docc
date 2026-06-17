#pragma once

#include "sdfg/data_flow/library_nodes/math/math_node.h"

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/serializer/json_serializer.h"

#include "sdfg/data_flow/library_nodes/math/math.h"

namespace sdfg {
namespace math {
namespace blas {

inline data_flow::LibraryNodeCode LibraryNodeType_DOT("DOT");

class DotNode : public BLASNode {
private:
    symbolic::Expression n_;
    symbolic::Expression incx_;
    symbolic::Expression incy_;

public:
    DotNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const data_flow::ImplementationType& implementation_type,
        const BLAS_Precision& precision,
        symbolic::Expression n,
        symbolic::Expression incx = symbolic::integer(1),
        symbolic::Expression incy = symbolic::integer(1)
    );

    BLAS_Layout layout() const;

    symbolic::Expression n() const;

    symbolic::Expression incx() const;

    symbolic::Expression incy() const;

    symbolic::SymbolSet symbols() const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    void replace(const symbolic::ExpressionMapping& replacements) override;

    void validate(const Function& function) const override;

    bool expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;

    data_flow::PointerAccessType pointer_access_type(int input_idx) const override;

    symbolic::Expression flop() const override;
};

class DotNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override;

    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override;
};

class DotNodeDispatcher_BLAS : public codegen::LibraryNodeDispatcher {
public:
    DotNodeDispatcher_BLAS(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const DotNode& node
    );

    void dispatch_code_with_edges(
        codegen::CodegenOutput& out,
        std::vector<codegen::DispatchInput>& inputs,
        std::vector<codegen::DispatchOutput>& outputs
    ) override;
};

} // namespace blas
} // namespace math
} // namespace sdfg
