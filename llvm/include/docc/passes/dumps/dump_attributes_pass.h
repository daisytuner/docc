#pragma once

#include <llvm/IR/PassManager.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>

#include <sdfg/analysis/analysis.h>
#include <sdfg/analysis/users.h>
#include <sdfg/structured_sdfg.h>
#include <string>

#include "docc/analysis/analysis.h"
#include "docc/analysis/attributes.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/code_node.h"
#include "sdfg/symbolic/symbolic.h"

namespace docc {
namespace passes {

class DumpAttributesPass : public llvm::PassInfoMixin<DumpAttributesPass> {
public:
    static bool available(analysis::AnalysisManager& AM) { return true; }

    llvm::PreservedAnalyses run(llvm::Module& Module, llvm::ModuleAnalysisManager& MAM, analysis::AnalysisManager& AM);
};

class AttributesAnalysis : public sdfg::analysis::Analysis {
private:
    docc::analysis::Attributes attributes_;

    sdfg::data_flow::AccessNode* get_in_access(sdfg::data_flow::CodeNode* node, const std::string& conn);
    sdfg::data_flow::AccessNode* get_out_access(sdfg::data_flow::CodeNode* node, const std::string& conn);

    void debug_print(const std::string& argument, const analysis::ArgumentAttributes& attrs);

    void set_argument_attributes(
        analysis::ArgumentAttributes& attrs,
        const std::string& copy_target,
        const std::string& copy_buffer,
        const sdfg::symbolic::Expression& copy_size,
        bool copy_in,
        bool copy_out,
        bool alloc,
        bool free
    );

protected:
    void run(sdfg::analysis::AnalysisManager& analysis_manager) override;

public:
    AttributesAnalysis(sdfg::StructuredSDFG& sdfg);

    std::string name() const override;

    const analysis::Attributes& get();

    static analysis::ArgumentAttributes empty();
};

} // namespace passes
} // namespace docc
