#pragma once

#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/RegionInfo.h>

#include "docc/utils.h"

#include <sdfg/builder/sdfg_builder.h>
#include <sdfg/builder/structured_sdfg_builder.h>
#include <sdfg/structured_sdfg.h>

namespace docc {
namespace lifting {

class FunctionToSDFG {
private:
    llvm::Function& function_;
    llvm::FunctionAnalysisManager& FAM_;

    bool apply_on_linkonce_odr_;

    size_t sdfg_counter;

    std::unique_ptr<llvm::Region> expand_region(std::unique_ptr<llvm::Region>& R);

    // Entire function
    std::pair<bool, llvm::Value*> can_be_applied();

    // Single region
    std::pair<bool, llvm::Value*> can_be_applied(llvm::Region& region);

    // Entire function
    std::unique_ptr<sdfg::StructuredSDFG> apply();

    // Single region
    std::unique_ptr<sdfg::StructuredSDFG> apply(llvm::Region& region);

    std::unique_ptr<sdfg::StructuredSDFG> simplify(std::unique_ptr<sdfg::StructuredSDFG>& sdfg);

public:
    FunctionToSDFG(llvm::Function& function, llvm::FunctionAnalysisManager& FAM, bool apply_on_linkonce_odr);

    std::vector<std::unique_ptr<sdfg::StructuredSDFG>> run();

    static size_t loop_count(llvm::Function& function, llvm::Region& region, llvm::FunctionAnalysisManager& FAM) {
        auto& LI = FAM.getResult<llvm::LoopAnalysis>(function);
        size_t count = 0;
        for (auto& L : LI) {
            if (FunctionToSDFG::contains_loop(&region, L)) {
                count++;
            }
        }

        return count;
    };

    static bool contains_loop(const llvm::Region* R, const llvm::Loop* L) {
        if (!R->contains(L->getHeader())) {
            return false;
        }

        for (const llvm::BasicBlock* BB : L->blocks()) {
            if (!R->contains(BB)) {
                return false;
            }
        }

        return true;
    };

    /**
     * Checks whether the entire function should not be checked for lifting
     * Not called as part of [run]. Check before even instantiating.
     */
    static bool is_blacklisted(llvm::Function& F, bool apply_on_linkonce_odr);
};

} // namespace lifting
} // namespace docc
