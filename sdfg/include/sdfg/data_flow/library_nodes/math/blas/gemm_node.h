#pragma once

#include "sdfg/data_flow/library_nodes/math/math_node.h"

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/serializer/json_serializer.h"

#include "sdfg/data_flow/library_nodes/math/math.h"

namespace sdfg {
namespace math {
namespace blas {

inline data_flow::LibraryNodeCode LibraryNodeType_GEMM("GEMM");

class GEMMNode : public BLASNode {
private:
    BLAS_Layout layout_;
    BLAS_Transpose trans_a_;
    BLAS_Transpose trans_b_;

    symbolic::Expression m_;
    symbolic::Expression n_;
    symbolic::Expression k_;
    symbolic::Expression lda_;
    symbolic::Expression ldb_;
    symbolic::Expression ldc_;

public:
    GEMMNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const data_flow::ImplementationType& implementation_type,
        const BLAS_Precision& precision,
        const BLAS_Layout& layout,
        const BLAS_Transpose& trans_a,
        const BLAS_Transpose& trans_b,
        symbolic::Expression m,
        symbolic::Expression n,
        symbolic::Expression k,
        symbolic::Expression lda,
        symbolic::Expression ldb,
        symbolic::Expression ldc
    );

    static constexpr int A_INPUT_IDX = 0;
    static constexpr int B_INPUT_IDX = 1;
    static constexpr int C_INPUT_IDX = 2;
    static constexpr int ALPHA_INPUT_IDX = 3;
    static constexpr int BETA_INPUT_IDX = 4;

    BLAS_Layout layout() const;

    BLAS_Transpose trans_a() const;

    BLAS_Transpose trans_b() const;

    symbolic::Expression m() const;

    symbolic::Expression n() const;

    symbolic::Expression k() const;

    symbolic::Expression lda() const;

    symbolic::Expression ldb() const;

    symbolic::Expression ldc() const;

    void validate(const Function& function) const override;

    bool expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    symbolic::SymbolSet symbols() const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    void replace(const symbolic::ExpressionMapping& replacements) override;

    std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;

    std::string toStr() const override;

    symbolic::Expression flop() const override;

    symbolic::Expression flops(
        symbolic::Condition alpha_non_zero,
        symbolic::Condition alpha_non_ident,
        symbolic::Condition beta_non_zero,
        symbolic::Condition beta_non_ident
    ) const;

    static symbolic::Expression calc_matrix_access_range(
        const symbolic::Expression& outer_dim,
        const symbolic::Expression& inner_dim,
        const symbolic::Expression& line_size,
        BLAS_Transpose trans,
        BLAS_Layout layout
    );

    data_flow::PointerAccessType pointer_access_type(int input_idx) const override;
};

class GEMMNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override;

    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override;
};

class GEMMNodeDispatcher_BLAS : public codegen::LibraryNodeDispatcher {
public:
    GEMMNodeDispatcher_BLAS(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const GEMMNode& node
    );

    void dispatch_code_with_edges(
        codegen::CodegenOutput& out,
        std::vector<codegen::DispatchInput>& inputs,
        std::vector<codegen::DispatchOutput>& outputs
    ) override;
};

GEMMNode& add_gemm_node(
    builder::StructuredSDFGBuilder& builder,
    Block& block,
    const std::string& ptr_a,
    const std::string& ptr_b,
    const std::string& ptr_c,
    data_flow::AccessNode& alpha_node,
    data_flow::AccessNode& beta_node,
    const BLAS_Precision& precision,
    const BLAS_Layout& layout,
    const BLAS_Transpose& trans_a,
    const BLAS_Transpose& trans_b,
    symbolic::Expression& m,
    symbolic::Expression& n,
    symbolic::Expression& k,
    symbolic::Expression& lda,
    symbolic::Expression& ldb,
    symbolic::Expression& ldc,
    const types::IType& a_type,
    const types::IType& b_type,
    const types::IType& c_type,
    const types::IType& factor_type,
    DebugInfo debug_info,
    DebugInfo a_access_deb_info,
    DebugInfo b_access_deb_info,
    DebugInfo c_access_deb_info,
    DebugInfo a_edge_deb_info,
    DebugInfo b_edge_deb_info,
    DebugInfo c_edge_deb_info,
    data_flow::ImplementationType impl_type
);

GEMMNode& add_gemm_node(
    builder::StructuredSDFGBuilder& builder,
    Block& block,
    const std::string& ptr_a,
    const std::string& ptr_b,
    const std::string& ptr_c,
    data_flow::AccessNode& alpha_node,
    data_flow::AccessNode& beta_node,
    const BLAS_Precision& precision,
    const BLAS_Layout& layout,
    const BLAS_Transpose& trans_a,
    const BLAS_Transpose& trans_b,
    symbolic::Expression& m,
    symbolic::Expression& n,
    symbolic::Expression& k,
    symbolic::Expression& lda,
    symbolic::Expression& ldb,
    symbolic::Expression& ldc,
    const types::IType& ptr_type,
    const types::IType& factor_type,
    DebugInfo debug_info = DebugInfo(),
    data_flow::ImplementationType impl_type = ImplementationType_BLAS
);

} // namespace blas
} // namespace math
} // namespace sdfg
