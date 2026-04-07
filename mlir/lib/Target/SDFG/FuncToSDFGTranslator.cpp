#include "mlir/Target/SDFG/FuncToSDFGTranslator.h"

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/LogicalResult.h>
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/SDFG/SDFGTranslator.h"

namespace mlir {
namespace sdfg {

LogicalResult translateFuncFuncOp(SDFGTranslator& translator, func::FuncOp* func_op) {
    if (!translator.builder_empty()) {
        return func_op->emitOpError("Currently only one function is supported");
    }

    std::string sdfg_name = func_op->getSymName().data();
    translator.builder().subject().name("__docc_" + sdfg_name);
    translator.builder_empty(false);
    llvm::ScopedHashTableScope<Value, std::string> value_map_scope(translator.value_map());

    // Arguments
    for (auto arg : func_op->getRegion().getArguments()) {
        translator.get_or_create_container(arg, true);
    }

    // Create output argument containers for each result type (void return, output via pointers)
    auto function_type = func_op->getFunctionType();
    std::vector<std::string> output_arg_names;
    for (size_t i = 0; i < function_type.getNumResults(); i++) {
        std::string output_name = "_docc_ret_" + std::to_string(i);
        auto result_type = translator.convertType(function_type.getResult(i));
        if (!result_type) {
            return func_op->emitError("Could not convert result type ")
                   << function_type.getResult(i) << " to SDFG type";
        }
        // Output arguments are always Pointer types (caller allocates memory)
        ::sdfg::types::Scalar element_type(result_type->primitive_type());
        translator.builder().add_container(output_name, ::sdfg::types::Pointer(element_type), true);
        output_arg_names.push_back(output_name);
    }
    translator.set_output_args(output_arg_names);

    // Region
    translator.enter_sequence(translator.builder().subject().root());
    for (auto& op : func_op->getRegion().getOps()) {
        if (failed(translateOp(translator, &op))) {
            return failure();
        }
    }
    translator.exit_sequence(translator.builder().subject().root());

    return success();
}

LogicalResult translateFuncReturnOp(SDFGTranslator& translator, func::ReturnOp* return_op) {
    const auto& output_args = translator.output_args();
    std::vector<std::string> output_shapes;
    auto deb_info = translator.get_debug_info(return_op->getOperationName(), return_op->getLoc());

    for (size_t i = 0; i < return_op->getOperands().size(); i++) {
        auto operand = return_op->getOperand(i);
        auto return_container = translator.get_or_create_container(operand);
        bool isa_tensor = llvm::isa<TensorType>(operand.getType());

        if (isa_tensor) {
            auto tensor_type = llvm::dyn_cast<TensorType>(operand.getType());
            auto& tensor_info = translator.get_or_create_tensor_info(return_container, tensor_type);
            auto element_type = translator.convertType(operand.getType())->primitive_type();

            // Copy to output container (always in C-order)
            translator.copy_to_output(return_container, tensor_info, element_type, output_args[i], deb_info);

            // Store shape for metadata
            output_shapes.push_back(tensor_info.shape_str());
        } else {
            // Scalar: copy via simple assignment
            translator.copy_scalar_to_output(return_container, output_args[i], deb_info);
            output_shapes.push_back("");
        }
    }

    translator.handle_frees("", deb_info);

    // Add metadata for output args
    std::string output_args_str;
    for (size_t i = 0; i < output_args.size(); i++) {
        if (i > 0) output_args_str += ",";
        output_args_str += output_args[i];
    }
    translator.builder().subject().add_metadata("output_args", output_args_str);

    // Add shapes metadata per output
    for (size_t i = 0; i < output_shapes.size(); i++) {
        if (!output_shapes[i].empty()) {
            translator.builder().subject().add_metadata(output_args[i] + "_shape", output_shapes[i]);
        }
    }

    // Void return
    translator.builder().add_return(translator.insertion_point(), "", {}, deb_info);

    return success();
}

LogicalResult translateFuncOp(SDFGTranslator& translator, Operation* op) {
    return llvm::TypeSwitch<Operation*, LogicalResult>(op)
        .Case<func::FuncOp>([&](func::FuncOp func_op) { return translateFuncFuncOp(translator, &func_op); })
        .Case<func::ReturnOp>([&](func::ReturnOp return_op) { return translateFuncReturnOp(translator, &return_op); })
        .Default([&](Operation* op) {
            return op->emitError("Unknown operation from func dialect encountered: ") << op->getName();
        });
}

} // namespace sdfg
} // namespace mlir
