#pragma once

#include <tuple>

#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

class KernelLocalStorage : public Transformation {
private:
    structured_control_flow::StructuredLoop& loop_;
    symbolic::Expression offset_;
    std::string container_;

public:
    KernelLocalStorage(
        structured_control_flow::StructuredLoop& loop, symbolic::Expression offset, const std::string& container
    );

    virtual std::string name() const override;

    static bool is_candidate(
        structured_control_flow::StructuredLoop& loop,
        const std::string& container,
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager
    );

    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        override;

    virtual void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual void to_json(nlohmann::json& j) const override;

    static KernelLocalStorage from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);

private:
    bool reads_container(std::string container, analysis::UsersView& body_users);
    bool uses_inner_indvar(analysis::UsersView& body_users);
    std::tuple<symbolic::Integer, symbolic::Integer, symbolic::Integer>
    dim_size(const std::vector<structured_control_flow::ControlFlowNode*> ancestors);

    std::tuple<symbolic::Symbol, symbolic::Symbol, symbolic::Symbol>
    dim_indvars(const std::vector<structured_control_flow::ControlFlowNode*> ancestors);
    std::tuple<bool, bool, bool>
    available_dims(std::vector<symbolic::Expression> subsets, analysis::AnalysisManager& analysis_manager);
};

} // namespace transformations
} // namespace sdfg
