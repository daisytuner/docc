#pragma once

#include "sdfg/passes/pass.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg::passes {

/// Fold contiguous batch and row dimensions into one matrix when the right operand is shared.
class MatMulBatchFlattening : public visitor::NonStoppingStructuredSDFGVisitor {
public:
    using visitor::NonStoppingStructuredSDFGVisitor::NonStoppingStructuredSDFGVisitor;

    static std::string name() {
        return "MatMulBatchFlattening";
    }

    bool accept(structured_control_flow::Block& block) override;
};

using MatMulBatchFlatteningPass = VisitorPass<MatMulBatchFlattening>;

} // namespace sdfg::passes
