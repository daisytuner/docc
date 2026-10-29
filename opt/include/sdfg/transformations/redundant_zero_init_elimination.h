#pragma once

#include <string>
#include <vector>

#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/transformations/transformation.h"
#include "sdfg/visitor/immutable_structured_sdfg_visitor.h"

namespace sdfg {
namespace gpu {
class GpuMmaFragmentLoadNode;
} // namespace gpu

namespace transformations {

/**
 * @brief Collects the uses of an output container within a GPU MMA loop nest.
 *
 * Visits all blocks below the nest root and records MMA matmul nodes, fragment stores to the container and
 * C-fragment loads of it. Any other use of the container, or a C-fragment load below anything but maps,
 * invalidates the result. visit() stops early once the result is invalid.
 */
class MmaNestOutputUses : public visitor::ImmutableStructuredSDFGVisitor {
public:
    struct CFragmentLoad {
        structured_control_flow::Block* block;
        gpu::GpuMmaFragmentLoadNode* node;
    };

private:
    structured_control_flow::StructuredLoop& nest_;
    std::string container_;

    bool valid_ = true;
    bool has_mma_ = false;
    bool has_store_ = false;
    std::vector<CFragmentLoad> c_loads_;

    /// True if only maps enclose @p block within the nest (including the nest root).
    bool only_maps_enclose(const structured_control_flow::Block& block) const;

public:
    MmaNestOutputUses(
        StructuredSDFG& sdfg,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::StructuredLoop& nest,
        const std::string& container
    );

    bool visit() override;

    bool accept(structured_control_flow::Block& node) override;

    /// The container is only used by fragment stores and C-fragment loads only nested in maps.
    bool valid() const {
        return valid_;
    }
    bool has_mma() const {
        return has_mma_;
    }
    bool has_store() const {
        return has_store_;
    }
    const std::vector<CFragmentLoad>& c_loads() const {
        return c_loads_;
    }
};

/**
 * @brief Eliminates a zero-initialization nest that is redundant before a GPU MMA matmul nest.
 *
 * Original:
 *   [init=zero] map i, j:  Y[i, j] = 0
 *   <MMA nest>:            ... MmaFragmentLoad(c <- Y) ... EltwiseAdd(acc, c) ... MmaFragmentStore(Y)
 *
 * After RedundantZeroInitElimination:
 *   <MMA nest>:            ... GpuMmaFill(c) ... EltwiseAdd(acc, c) ... MmaFragmentStore(Y)
 *
 * The C-fragment loads of Y in the MMA nest could only observe the zeros written by the init nest,
 * so they are replaced by zero fills and the init nest is removed.
 *
 * Prerequisites:
 * - The loop carries the zero-init annotation (ScheduleType_ZeroInit). The producer (naive MatMul/GEMM
 *   expansion) thereby certifies that the nest writes zeros to exactly the output region of the matmul
 *   nest directly following it.
 * - The nest is still a pure zero init, i.e. a perfect loop chain whose body is a single
 *   `constant 0 -> assign -> Y` block (it was not fused with other computation).
 * - The next loop nest in the same sequence contains GPU MMA matmul nodes and fragment stores to Y.
 *   Only data-transfer blocks (offloading nodes and their access nodes) that do not access Y may lie in
 *   between; these are inserted by offloading and are kept in place.
 * - Within that nest, Y is only used as the memory operand of fragment stores and of C-fragment loads,
 *   and the C-fragment loads are only nested in maps (no sequential loop could re-read stored partial results).
 *
 * Note: the MMA nests produced by GpuMmaEinsumTransform never use the data already contained in Y beyond
 * these C-fragment loads. If other MMA lowerings are matched in the future (e.g. an overwrite mode
 * without C-fragment loads, or nests reading Y differently), the checks may need to be extended.
 */
class RedundantZeroInitElimination : public Transformation {
    structured_control_flow::StructuredLoop& loop_;

public:
    explicit RedundantZeroInitElimination(structured_control_flow::StructuredLoop& loop);

    std::string name() const override;

    bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    void to_json(nlohmann::json& j) const override;

    static RedundantZeroInitElimination from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);
};

} // namespace transformations
} // namespace sdfg
