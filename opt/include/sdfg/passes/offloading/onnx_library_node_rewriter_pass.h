#pragma once

#include "sdfg/passes/pass.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {

/**
 * @class ONNXLibraryNodeRewriter
 * @brief Pass that sets the implementation type to ONNX for all tensor nodes
 *
 * This pass visits all blocks in the SDFG and sets the implementation type
 * of tensor library nodes (ElementWiseUnaryNode, ElementWiseBinaryNode,
 * ReduceNode, ConvNode, TransposeNode, BroadcastNode) to ONNX.
 *
 * This allows tensor operations to be dispatched to the ONNX Runtime
 * instead of being expanded or using other backends.
 */
class ONNXLibraryNodeRewriter : public visitor::NonStoppingStructuredSDFGVisitor {
public:
    /**
     * @brief Constructs a new ONNXLibraryNodeRewriter pass
     * @param builder The structured SDFG builder
     * @param analysis_manager The analysis manager
     */
    ONNXLibraryNodeRewriter(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);

    /**
     * @brief Returns the name of the pass
     * @return The string "ONNXLibraryNodeRewriter"
     */
    static std::string name() { return "ONNXLibraryNodeRewriter"; }

    /**
     * @brief Visit entry point
     * @return true if any tensor nodes were modified
     */
    virtual bool visit() override;

    /**
     * @brief Accepts a block and sets ONNX implementation type for tensor nodes
     * @param block The block to process
     * @return true if any tensor nodes were modified
     */
    virtual bool accept(structured_control_flow::Block& block) override;
};

typedef VisitorPass<ONNXLibraryNodeRewriter> ONNXLibraryNodeRewriterPass;

} // namespace passes
} // namespace sdfg
