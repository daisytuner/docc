#pragma once

#include <string>

#include "sdfg/passes/pass.h"

namespace sdfg {
namespace passes {

/**
 * Records the byte size of pointer containers in their StorageType::allocation_size,
 * derived from the Tensor-typed memlets connecting them to library nodes.
 *
 * Must run before library node expansion, which drops the tensor layouts. The size is the
 * memory span of the layout (including gaps of non-contiguous layouts); multiple uses are
 * combined with max. Containers are skipped if any use depends on non-argument symbols,
 * and managed allocations are left untouched.
 */
class TensorAllocationSizeInference : public Pass {
public:
    std::string name() override {
        return "TensorAllocationSizeInference";
    }

    bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;
};

} // namespace passes
} // namespace sdfg
