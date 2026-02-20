#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sstream>
#include <string>

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/SDFG/TranslateToSDFG.h"

#include "sdfg/codegen/dispatchers/node_dispatcher_registry.h"
#include "sdfg/serializer/json_serializer.h"

namespace py = pybind11;

namespace {

class MLIRInitializer {
public:
    static MLIRInitializer& instance() {
        static MLIRInitializer inst;
        return inst;
    }

    mlir::DialectRegistry& registry() { return registry_; }

private:
    MLIRInitializer() {
        // Register all standard MLIR dialects
        mlir::registerAllDialects(registry_);
        // Register the SDFG dialect
        registry_.insert<mlir::sdfg::SDFGDialect>();
        // Register all passes
        mlir::registerAllPasses();
        mlir::registerSDFGConversionPasses();

        // Register SDFG code generators and serializers
        ::sdfg::codegen::register_default_dispatchers();
        ::sdfg::serializer::register_default_serializers();
    }

    mlir::DialectRegistry registry_;
};

class PyMLIRModule {
public:
    explicit PyMLIRModule(const std::string& mlir_text) {
        auto& initializer = MLIRInitializer::instance();
        context_ = std::make_unique<mlir::MLIRContext>(initializer.registry());
        context_->loadAllAvailableDialects();

        module_ = mlir::parseSourceString<mlir::ModuleOp>(mlir_text, context_.get());
        if (!module_) {
            throw std::runtime_error("Failed to parse MLIR module");
        }
    }

    std::string to_string() {
        std::string result;
        llvm::raw_string_ostream os(result);
        module_->print(os);
        return result;
    }

    std::string translate() {
        std::string result;
        llvm::raw_string_ostream os(result);
        if (mlir::failed(mlir::sdfg::translateToSDFG(module_->getOperation(), os))) {
            throw std::runtime_error("Failed to translate to SDFG code");
        }
        return result;
    }

    mlir::Operation* get_operation() { return module_->getOperation(); }

private:
    std::unique_ptr<mlir::MLIRContext> context_;
    mlir::OwningOpRef<mlir::ModuleOp> module_;
};

} // namespace

PYBIND11_MODULE(_sdfg_mlir, m) {
    m.doc() = "Native Python bindings for MLIR to SDFG conversion";

    // Expose the PyMLIRModule class
    py::class_<PyMLIRModule>(m, "MLIRModule")
        .def(py::init<const std::string&>(), py::arg("mlir_text"), "Create an MLIR module from MLIR text representation")
        .def("to_string", &PyMLIRModule::to_string, "Get the MLIR module as a string")
        .def("translate", &PyMLIRModule::translate, "Translate the SDFG dialect module to a serialized SDFG");
}
