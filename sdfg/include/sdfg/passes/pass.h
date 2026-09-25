#pragma once

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/sdfg_builder.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/options.h"

namespace sdfg {
namespace passes {

class Pass {
public:
    Pass() = default;
    // A pass is constructed for a specific set of options (non-owning; caller keeps them alive).
    explicit Pass(const Options& options) : options_(&options) {
    }

    virtual ~Pass() = default;

    virtual std::string name() = 0;

    // Registrable option specs for this pass; default: none.
    virtual std::vector<OptionSpec> options() {
        return {};
    }

    bool run(builder::SDFGBuilder& builder, bool create_report = false);

    bool
    run(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager, bool create_report = false
    );

    virtual bool run_pass(builder::SDFGBuilder& builder);

    virtual bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);

    virtual void invalidates(analysis::AnalysisManager& analysis_manager, bool applied);

protected:
    template<class T>
    T option(const OptionKey<T>& key, T fallback = T{}) const {
        return options_->get(key, fallback);
    }

private:
    // Options this pass was constructed for (non-owning).
    const Options* options_ = &Options::empty();
};

template<typename T>
class VisitorPass : public Pass {
    std::string name() override {
        return T::name();
    };

    bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override {
        T visitor(builder, analysis_manager);
        return visitor.visit();
    };
};

} // namespace passes
} // namespace sdfg
