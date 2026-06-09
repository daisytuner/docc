#pragma once

#include "sdfg/data_flow/library_nodes/math/blas/blas_node.h"
#include "sdfg/data_flow/library_nodes/math/math_node.h"

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/serializer/json_serializer.h"

#include "sdfg/data_flow/library_nodes/math/math.h"

namespace sdfg {
namespace math {
namespace blas {

inline data_flow::LibraryNodeCode LibraryNodeType_BatchedGEMM("BatchedGEMM");

class BatchedGEMMNode : public BLASNode {
private:
    BLAS_Layout layout_;
    BLAS_Transpose trans_a_;
    BLAS_Transpose trans_b_;

    symbolic::Expression batch_count_;
    symbolic::Expression m_;
    symbolic::Expression n_;
    symbolic::Expression k_;
    symbolic::Expression lda_;
    symbolic::Expression ldb_;
    symbolic::Expression ldc_;
    symbolic::Expression stride_a_;
    symbolic::Expression stride_b_;
    symbolic::Expression stride_c_;

public:
    BatchedGEMMNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const data_flow::ImplementationType& implementation_type,
        const BLAS_Precision& precision,
        const BLAS_Layout& layout,
        const BLAS_Transpose& trans_a,
        const BLAS_Transpose& trans_b,
        symbolic::Expression batch_count,
        symbolic::Expression m,
        symbolic::Expression n,
        symbolic::Expression k,
        symbolic::Expression lda,
        symbolic::Expression ldb,
        symbolic::Expression ldc,
        symbolic::Expression stride_a,
        symbolic::Expression stride_b,
        symbolic::Expression stride_c
    );

    static constexpr int A_INPUT_IDX = 0;
    static constexpr int B_INPUT_IDX = 1;
    static constexpr int C_INPUT_IDX = 2;
    static constexpr int ALPHA_INPUT_IDX = 3;
    static constexpr int BETA_INPUT_IDX = 4;

    BLAS_Layout layout() const;

    BLAS_Transpose trans_a() const;

    BLAS_Transpose trans_b() const;

    symbolic::Expression batch_count() const;

    symbolic::Expression m() const;

    symbolic::Expression n() const;

    symbolic::Expression k() const;

    symbolic::Expression lda() const;

    symbolic::Expression ldb() const;

    symbolic::Expression ldc() const;

    symbolic::Expression stride_a() const;

    symbolic::Expression stride_b() const;

    symbolic::Expression stride_c() const;

    void validate(const Function& function) const override;

    symbolic::SymbolSet symbols() const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;

    std::string toStr() const override;

    symbolic::Expression flop() const override;

    data_flow::PointerAccessType pointer_access_type(int input_idx) const override;

    bool expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;
};

class BatchedGEMMNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override;

    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override;
};

} // namespace blas
} // namespace math
} // namespace sdfg
