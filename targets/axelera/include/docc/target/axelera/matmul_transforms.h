#pragma once

#include <sdfg/structured_control_flow/structured_loop.h>
#include <sdfg/transformations/transformation.h>

namespace sdfg {
namespace axelera {

/**
 * Runs the float matmul C = A @ B that einsum detection finds in the loop nest @p loop on the MVM, which multiplies
 * int8 matrices: absmax reductions scale A and B per tensor, maps quantize them to int8, one AI core multiplies each
 * row of A with B (an @ref AxeleraMatMulNode per row), and a map scales the products back into C. A, B and C must
 * be dense row-major and C zero-initialized by the loop nest right before @p loop, as GEMM and MatMul expand.
 * With a @p k_panel smaller than K, the reduction runs in panels of that many rows of B, each loaded into the IMC in
 * turn; the first panel's products overwrite, the later ones accumulate. With several @p cores, each multiplies a
 * contiguous block of rows of A against its own IMC copy of B. Tiling and staging the row loop are left to further
 * transformations.
 */
class MvmEinsumTransform : public transformations::Transformation {
    structured_control_flow::StructuredLoop& loop_;
    size_t loop_element_id_;
    int64_t k_panel_;
    int64_t cores_;
    structured_control_flow::StructuredLoop* rows_ = nullptr;

public:
    /// @param k_panel rows of B per IMC load, a multiple of the PWORD lanes dividing K; 0 for all of K at once, or
    /// the largest such panel if the IMC cannot hold all of K
    /// @param cores AI cores sharing the rows of A, dividing M
    explicit MvmEinsumTransform(structured_control_flow::StructuredLoop& loop, int64_t k_panel = 0, int64_t cores = 1)
        : loop_(loop), loop_element_id_(loop.element_id()), k_panel_(k_panel), cores_(cores) {
    }

    std::string name() const override {
        return "AxeleraMvmEinsumTransform";
    }

    bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;
    void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;
    void to_json(nlohmann::json& j) const override;

    /// The loop over the rows of A in the AI core (valid after apply()).
    structured_control_flow::StructuredLoop* row_loop() const {
        return rows_;
    }
};

/**
 * Makes the B matrix of the single @ref AxeleraMatMulNode under @p loop resident in a new IMC buffer, loaded once
 * before the loop, so every iteration multiplies against the same weights. B must not move with any loop at or
 * below @p loop.
 */
class ImcResidency : public transformations::Transformation {
    structured_control_flow::StructuredLoop& loop_;
    std::string imc_container_;

public:
    explicit ImcResidency(structured_control_flow::StructuredLoop& loop) : loop_(loop) {
    }

    std::string name() const override {
        return "AxeleraImcResidency";
    }

    bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;
    void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;
    void to_json(nlohmann::json& j) const override;

    /// Name of the IMC buffer (valid after apply()).
    const std::string& imc_container() const {
        return imc_container_;
    }
};

} // namespace axelera
} // namespace sdfg
