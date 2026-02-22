#include "mlir/Conversion/ConvertToSDFG/ConvertToSDFG.h"

#include <llvm/Support/LogicalResult.h>

#include "mlir/Conversion/ArithToSDFG/ArithToSDFG.h"
#include "mlir/Conversion/FuncToSDFG/FuncToSDFG.h"
#include "mlir/Conversion/LinalgToSDFG/LinalgToSDFG.h"
#include "mlir/Conversion/TensorToSDFG/TensorToSDFG.h"
#include "mlir/Dialect/SDFG/IR/SDFG.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {

#define GEN_PASS_DEF_CONVERTTOSDFG
#include "mlir/Conversion/SDFGPasses.h.inc"

namespace {

struct ConvertToSDFG : public impl::ConvertToSDFGBase<ConvertToSDFG> {
    using ConvertToSDFGBase::ConvertToSDFGBase;

    void runOnOperation() override {
        auto pm = PassManager::on<ModuleOp>(&this->getContext());
        pm.addPass(createConvertFuncToSDFG());
        pm.addPass(createConvertArithToSDFG());
        pm.addPass(createConvertTensorToSDFG());
        pm.addPass(createConvertLinalgToSDFG());
        if (failed(this->runPipeline(pm, this->getOperation()))) {
            this->signalPassFailure();
        }
    }
};

} // namespace
} // namespace mlir
