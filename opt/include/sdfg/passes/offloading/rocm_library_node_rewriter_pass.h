#pragma once

#include "sdfg/data_flow/library_nodes/math/blas/gemm_node.h"
#include "sdfg/data_flow/library_nodes/stdlib/memset.h"
#include "sdfg/passes/pass.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace rocm {

class RocmLibraryNodeRewriter : public visitor::StructuredSDFGVisitor {
public:
    RocmLibraryNodeRewriter(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);

    static std::string name() { return "RocmLibraryNodeRewriterPass"; };
    bool accept(structured_control_flow::Block& node) override;

private:
    std::optional<data_flow::ImplementationType>
    try_library_node_implementation(const data_flow::LibraryNode& lib_node, types::PrimitiveType data_type);

    std::optional<data_flow::ImplementationType>
    try_rocm_gemm_node_implementation(const math::blas::GEMMNode& gemm_node, types::PrimitiveType data_type);

    std::optional<data_flow::ImplementationType> try_memset_implementation(const stdlib::MemsetNode& memset_node);
};

typedef passes::VisitorPass<RocmLibraryNodeRewriter> RocmLibraryNodeRewriterPass;

} // namespace rocm
} // namespace sdfg
