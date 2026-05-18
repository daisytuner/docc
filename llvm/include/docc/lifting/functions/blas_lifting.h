#include "docc/lifting/functions/function_lifting.h"
#include "docc/lifting/lift_report.h"

#include <sdfg/data_flow/library_nodes/math/math.h>

namespace docc {
namespace lifting {

class BLASLifting : public FunctionLifting {
private:
    static sdfg::data_flow::LibraryNodeCode library_node_code(const llvm::StringRef function_name) {
        if (function_name == "cblas_sdot") {
            return sdfg::math::blas::LibraryNodeType_DOT;
        } else if (function_name == "cblas_ddot") {
            return sdfg::math::blas::LibraryNodeType_DOT;
        } else if (function_name == "cblas_sgemm") {
            return sdfg::math::blas::LibraryNodeType_GEMM;
        } else if (function_name == "cblas_dgemm") {
            return sdfg::math::blas::LibraryNodeType_GEMM;
        }

        return sdfg::data_flow::LibraryNodeCode{""};
    }

    sdfg::control_flow::State& visit_dot(
        const llvm::BasicBlock* block, const llvm::CallBase* instruction, sdfg::control_flow::State& current_state
    );

    sdfg::control_flow::State& visit_gemm(
        const llvm::BasicBlock* block, const llvm::CallBase* instruction, sdfg::control_flow::State& current_state
    );

public:
    BLASLifting(
        llvm::TargetLibraryInfo& TLI,
        const llvm::DataLayout& DL,
        const llvm::Function& function,
        sdfg::FunctionType target_type,
        sdfg::builder::SDFGBuilder& builder,
        std::map<const llvm::BasicBlock*, std::set<const sdfg::control_flow::State*>>& state_mapping,
        std::map<const sdfg::control_flow::State*, std::set<const llvm::BasicBlock*>>& pred_mapping,
        std::unordered_map<const llvm::Value*, std::string>& constants_mapping,
        std::unordered_map<const llvm::Type*, std::string> anonymous_types_mapping
    )
        : FunctionLifting(
              TLI,
              DL,
              function,
              target_type,
              builder,
              state_mapping,
              pred_mapping,
              constants_mapping,
              anonymous_types_mapping
          ) {}

    sdfg::control_flow::State& visit(
        const llvm::BasicBlock* block, const llvm::CallBase* instruction, sdfg::control_flow::State& current_state
    ) override;

    static bool is_supported(llvm::Function& func) {
        if (!func.getName().starts_with("cblas_")) {
            return false;
        }

        return library_node_code(func.getName()) != sdfg::data_flow::LibraryNodeCode{""};
    };
};

} // namespace lifting
} // namespace docc
