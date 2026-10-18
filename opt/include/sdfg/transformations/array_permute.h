#pragma once

#include <string>
#include <vector>

#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

/**
 * @brief Stores a container with the dimensions of its layout permuted within a region of a sequence.
 *
 * The container is viewed as a row-major tensor of @p shape. Within the region, the children of @p parent from
 * @p first to @p last, it is stored with dimension `permutation[j]` as its `j`-th, e.g. `{1, 0}` transposes a matrix
 * and `{0, 2, 1}` each matrix of a batch. Every access in the region must be affine in induction variables of
 * constant ranges, such that its coordinates in the tensor are affine too.
 *
 * A transient whose elements only the region accesses is permuted in place. Otherwise the region accesses a
 * permuted copy, allocated, copied into and out of around the region with TensorCopyNodes as needed.
 */
class ArrayPermute : public Transformation {
    structured_control_flow::Sequence& parent_;
    structured_control_flow::ControlFlowNode& first_;
    structured_control_flow::ControlFlowNode& last_;
    std::string container_;
    std::vector<int64_t> shape_;
    std::vector<size_t> permutation_;
    std::string permuted_;

public:
    ArrayPermute(
        structured_control_flow::Sequence& parent,
        structured_control_flow::ControlFlowNode& first,
        structured_control_flow::ControlFlowNode& last,
        const std::string& container,
        const std::vector<int64_t>& shape,
        const std::vector<size_t>& permutation
    );

    std::string name() const override;

    bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    void to_json(nlohmann::json& j) const override;

    static ArrayPermute from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);

    /// The container the region accesses after apply(): the container itself if permuted in place, else its copy.
    const std::string& permuted_container() const;
};

} // namespace transformations
} // namespace sdfg
