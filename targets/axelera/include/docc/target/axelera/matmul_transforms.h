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
 * Tiling and staging the row loop are left to further transformations.
 */
class MvmEinsumTransform : public transformations::Transformation {
    structured_control_flow::StructuredLoop& loop_;
    size_t loop_element_id_;
    structured_control_flow::StructuredLoop* rows_ = nullptr;

public:
    explicit MvmEinsumTransform(structured_control_flow::StructuredLoop& loop)
        : loop_(loop), loop_element_id_(loop.element_id()) {
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
