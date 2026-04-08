#include "mlir/Target/SDFG/TensorToSDFGTranslator.h"

#include <cstddef>
#include <cstdint>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/LogicalResult.h>

#include <memory>
#include <string>
#include <vector>
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Target/SDFG/SDFGTranslator.h"
#include "mlir/Target/SDFG/helper.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/tensor.h"

namespace mlir {
namespace sdfg {

LogicalResult translateTensorEmptyOp(SDFGTranslator& translator, tensor::EmptyOp* empty_op) {
    Value result = empty_op->getResult();
    auto result_tensor_type = llvm::dyn_cast<TensorType>(result.getType());
    auto deb_info = translator.get_debug_info(empty_op->getOperationName(), empty_op->getLoc());

    std::string container = translator.get_or_create_container(result);
    auto tensor_info = translator.get_or_create_tensor_info(container, result_tensor_type);

    auto element_type = translator.convertType(result_tensor_type.getElementType());
    auto sdfg_tensor = tensor_info.get_sdfg_tensor(static_cast<::sdfg::types::Scalar&>(*element_type));

    uint64_t size = 1;
    for (int64_t dim : tensor_info.shape()) {
        size *= dim;
    }
    translator.handle_malloc(
        container,
        ::sdfg::symbolic::mul(::sdfg::symbolic::integer(size), ::sdfg::symbolic::size_of_type(*element_type)),
        deb_info
    );

    return success();
}

LogicalResult translateTensorCollapseOp(SDFGTranslator& translator, tensor::CollapseShapeOp* collapse_op) {
    Value input = collapse_op->getSrc();
    Value result = collapse_op->getResult();
    auto deb_info = translator.get_debug_info(collapse_op->getOperationName(), collapse_op->getLoc());

    auto input_tensor_type = llvm::dyn_cast<TensorType>(input.getType());
    auto result_tensor_type = llvm::dyn_cast<TensorType>(result.getType());
    if (!input_tensor_type || !result_tensor_type) {
        return collapse_op->emitError("Input and output types must be ranked tensors");
    }

    auto in_container = translator.get_or_create_container(input);
    auto out_container = translator.get_or_create_container(result);

    translator.add_reference(in_container, out_container, deb_info);

    auto& in_tensor_info = translator.get_or_create_tensor_info(in_container, input_tensor_type);
    auto in_element_type = translator.convertType(input_tensor_type.getElementType());
    auto& in_scalar_type = static_cast<::sdfg::types::Scalar&>(*in_element_type);
    in_container = translator.store_in_c_order(in_container, in_tensor_info, in_scalar_type, deb_info);
    in_tensor_info = translator.get_or_create_tensor_info(in_container, input_tensor_type);

    auto new_shape = result_tensor_type.getShape();
    if (!in_tensor_info.is_reshape_valid(new_shape)) {
        return collapse_op->emitError("Collapse reshape is not valid (non-contiguous or mismatched element count)");
    }

    auto out_tensor_info = in_tensor_info.reshape(new_shape);
    translator.tensor_info_map().insert({out_container, out_tensor_info});

    return success();
}

LogicalResult translateTensorConcatOp(SDFGTranslator& translator, tensor::ConcatOp* concat_op) {
    OperandRange inputs = concat_op->getInputs();
    std::vector<std::string> inputs_container;
    inputs_container.reserve(inputs.size());
    std::vector<std::unique_ptr<::sdfg::types::Tensor>> inputs_sdfg_tensor;
    inputs_sdfg_tensor.reserve(inputs.size());
    for (Value input : inputs) {
        auto input_container = translator.get_or_create_container(input);
        inputs_container.push_back(input_container);
        auto input_tensor_type = llvm::dyn_cast<TensorType>(input.getType());
        auto& input_tensor_info = translator.get_or_create_tensor_info(input_container, input_tensor_type);
        auto input_element_type = translator.convertType(input_tensor_type.getElementType());
        auto& input_scalar_type = static_cast<::sdfg::types::Scalar&>(*input_element_type);
        auto input_sdfg_tensor = input_tensor_info.get_sdfg_tensor(input_scalar_type);
        inputs_sdfg_tensor.push_back(std::move(input_sdfg_tensor));
    }

    Value result = concat_op->getResult();
    auto result_container = translator.get_or_create_container(result);
    auto result_tensor_type = llvm::dyn_cast<TensorType>(result.getType());
    auto& result_tensor_info = translator.get_or_create_tensor_info(result_container, result_tensor_type);
    auto result_element_type = translator.convertType(result_tensor_type.getElementType());
    auto result_scalar_type = static_cast<::sdfg::types::Scalar&>(*result_element_type);
    auto result_sdfg_tensor = result_tensor_info.get_sdfg_tensor(result_scalar_type);
    auto deb_info = translator.get_debug_info(concat_op->getOperationName(), concat_op->getLoc());

    uint64_t dim = concat_op->getDim();
    if (dim >= result_tensor_info.shape().size()) {
        return concat_op->emitOpError("has dimension ")
               << dim << " but maximum dimension of result is " << result_tensor_info.shape().size();
    }

    // Allocation for concatenated container
    uint64_t size = 1;
    for (int64_t dim : result_tensor_info.shape()) {
        size *= dim;
    }
    translator.handle_malloc(
        result_container,
        ::sdfg::symbolic::mul(::sdfg::symbolic::integer(size), ::sdfg::symbolic::size_of_type(result_scalar_type)),
        deb_info
    );

    // Create maps
    auto& builder = translator.builder();
    ::sdfg::structured_control_flow::Sequence* current_seq = &translator.insertion_point();
    std::vector<std::string> indvars;
    ::sdfg::data_flow::Subset subset;
    for (size_t i = 0; i < result_tensor_info.shape().size(); i++) {
        int64_t dim = result_tensor_info.shape().at(i);
        auto indvar_container = builder.find_new_name("_i");
        builder.add_container(indvar_container, ::sdfg::types::Scalar(sdfg_index_type));
        indvars.push_back(indvar_container);
        subset.push_back(::sdfg::symbolic::symbol(indvar_container));
        auto indvar = ::sdfg::symbolic::symbol(indvar_container);
        auto bound = ::sdfg::symbolic::integer(dim);
        auto condition = ::sdfg::symbolic::Lt(indvar, bound);
        auto init = ::sdfg::symbolic::zero();
        auto update = ::sdfg::symbolic::add(indvar, ::sdfg::symbolic::one());

        auto& map = builder.add_map(
            *current_seq,
            indvar,
            condition,
            init,
            update,
            ::sdfg::structured_control_flow::ScheduleType_Sequential::create(),
            {},
            deb_info
        );
        current_seq = &map.root();
    }

    // Create if/else
    auto& if_else = builder.add_if_else(*current_seq, {}, deb_info);
    auto dim_indvar = subset.at(dim);

    // Create conditional copy for every input
    ::sdfg::symbolic::Expression offset = ::sdfg::symbolic::zero();
    for (size_t i = 0; i < inputs.size(); i++) {
        auto new_offset = ::sdfg::symbolic::add(offset, inputs_sdfg_tensor.at(i)->shape().at(dim));
        auto condition =
            ::sdfg::symbolic::And(::sdfg::symbolic::Ge(dim_indvar, offset), ::sdfg::symbolic::Lt(dim_indvar, new_offset));
        auto& sequence = builder.add_case(if_else, condition, deb_info);

        ::sdfg::data_flow::Subset offset_subset(subset);
        offset_subset[dim] = ::sdfg::symbolic::sub(dim_indvar, offset);

        auto& block = builder.add_block(sequence, {}, deb_info);
        auto& input_access = builder.add_access(block, inputs_container.at(i), deb_info);
        auto& result_access = builder.add_access(block, result_container, deb_info);
        auto& tasklet = builder.add_tasklet(block, ::sdfg::data_flow::TaskletCode::assign, "_out", {"_in"}, deb_info);
        builder.add_computational_memlet(
            block, input_access, tasklet, "_in", offset_subset, *inputs_sdfg_tensor.at(i), deb_info
        );
        builder.add_computational_memlet(block, tasklet, "_out", result_access, subset, *result_sdfg_tensor, deb_info);

        offset = new_offset;
    }

    return success();
}

LogicalResult translateTensorExpandOp(SDFGTranslator& translator, tensor::ExpandShapeOp* expand_op) {
    Value input = expand_op->getSrc();
    Value result = expand_op->getResult();
    auto deb_info = translator.get_debug_info(expand_op->getOperationName(), expand_op->getLoc());

    auto input_tensor_type = llvm::dyn_cast<TensorType>(input.getType());
    auto result_tensor_type = llvm::dyn_cast<TensorType>(result.getType());
    if (!input_tensor_type || !result_tensor_type) {
        return expand_op->emitError("Input and output types must be ranked tensors");
    }

    auto in_container = translator.get_or_create_container(input);
    auto out_container = translator.get_or_create_container(result);

    translator.add_reference(in_container, out_container, deb_info);

    auto& in_tensor_info = translator.get_or_create_tensor_info(in_container, input_tensor_type);
    auto in_element_type = translator.convertType(input_tensor_type.getElementType());
    auto& in_scalar_type = static_cast<::sdfg::types::Scalar&>(*in_element_type);
    in_container = translator.store_in_c_order(in_container, in_tensor_info, in_scalar_type, deb_info);
    in_tensor_info = translator.get_or_create_tensor_info(in_container, input_tensor_type);

    auto new_shape = result_tensor_type.getShape();
    if (!in_tensor_info.is_reshape_valid(new_shape)) {
        return expand_op->emitError("Expand reshape is not valid (non-contiguous or mismatched element count)");
    }

    auto out_tensor_info = in_tensor_info.reshape(new_shape);
    translator.tensor_info_map().insert({out_container, out_tensor_info});

    return success();
}

LogicalResult translateTensorExtractOp(SDFGTranslator& translator, tensor::ExtractOp* extract_op) {
    Value tensor = extract_op->getTensor();
    auto deb_info = translator.get_debug_info(extract_op->getOperationName(), extract_op->getLoc());

    auto tensor_container = translator.get_or_create_container(tensor);
    auto tensor_type = llvm::dyn_cast<TensorType>(tensor.getType());
    auto& tensor_info = translator.get_or_create_tensor_info(tensor_container, tensor_type);
    auto element_type = translator.convertType(tensor_type.getElementType());
    auto sdfg_tensor = tensor_info.get_sdfg_tensor(static_cast<::sdfg::types::Scalar&>(*element_type));

    OperandRange indices = extract_op->getIndices();
    ::sdfg::data_flow::Subset subset;
    subset.reserve(indices.size());
    for (Value index : indices) {
        subset.push_back(::sdfg::symbolic::symbol(translator.get_or_create_container(index)));
    }

    Value result = extract_op->getResult();
    auto result_container = translator.get_or_create_container(result);

    auto& builder = translator.builder();
    auto& block = builder.add_block(translator.insertion_point(), {}, deb_info);
    auto& tensor_access = builder.add_access(block, tensor_container, deb_info);
    auto& result_access = builder.add_access(block, result_container, deb_info);
    auto& tasklet = builder.add_tasklet(block, ::sdfg::data_flow::TaskletCode::assign, "_out", {"_in"}, deb_info);
    builder.add_computational_memlet(block, tensor_access, tasklet, "_in", subset, *sdfg_tensor, deb_info);
    builder.add_computational_memlet(block, tasklet, "_out", result_access, {}, deb_info);

    return success();
}

LogicalResult translateTensorPadOp(SDFGTranslator& translator, tensor::PadOp* pad_op) {
    Value source = pad_op->getSource();
    Value result = pad_op->getResult();
    auto deb_info = translator.get_debug_info(pad_op->getOperationName(), pad_op->getLoc());

    // Extract padding values
    std::vector<::sdfg::symbolic::Expression> low, high;
    auto static_low = pad_op->getStaticLow();
    auto static_high = pad_op->getStaticHigh();
    low.reserve(static_low.size());
    high.reserve(static_high.size());
    size_t i = 0;
    for (auto val : static_low) {
        if (val == INT64_MIN) {
            if (i >= pad_op->getLow().size()) {
                return pad_op->emitError("Index out of (non-static) low range: ") << i;
            }
            low.push_back(::sdfg::symbolic::symbol(translator.get_or_create_container(pad_op->getLow()[i++])));
        } else {
            low.push_back(::sdfg::symbolic::integer(val));
        }
    }
    i = 0;
    for (auto val : static_high) {
        if (val == INT64_MIN) {
            if (i >= pad_op->getHigh().size()) {
                return pad_op->emitError("Index out of (non-static) high range: ") << i;
            }
            high.push_back(::sdfg::symbolic::symbol(translator.get_or_create_container(pad_op->getHigh()[i++])));
        } else {
            high.push_back(::sdfg::symbolic::integer(val));
        }
    }

    auto& builder = translator.builder();
    auto source_container = translator.get_or_create_container(source);
    auto result_container = translator.get_or_create_container(result);

    auto source_tensor_type = llvm::dyn_cast<TensorType>(source.getType());
    auto& source_tensor_info = translator.get_or_create_tensor_info(source_container, source_tensor_type);
    auto result_tensor_type = llvm::dyn_cast<TensorType>(result.getType());
    auto& result_tensor_info = translator.get_or_create_tensor_info(result_container, result_tensor_type);

    auto source_element_type = translator.convertType(source_tensor_type.getElementType());
    auto source_sdfg_tensor =
        source_tensor_info.get_sdfg_tensor(static_cast<::sdfg::types::Scalar&>(*source_element_type));
    auto result_element_type = translator.convertType(result_tensor_type.getElementType());
    auto result_sdfg_tensor =
        result_tensor_info.get_sdfg_tensor(static_cast<::sdfg::types::Scalar&>(*result_element_type));

    // Allocation for padded container
    uint64_t size = 1;
    for (int64_t dim : result_tensor_info.shape()) {
        size *= dim;
    }
    translator.handle_malloc(
        result_container,
        ::sdfg::symbolic::mul(::sdfg::symbolic::integer(size), ::sdfg::symbolic::size_of_type(*result_element_type)),
        deb_info
    );

    // Create loops
    ::sdfg::structured_control_flow::Sequence* current_seq = &translator.insertion_point();
    std::vector<std::string> indvars;
    ::sdfg::data_flow::Subset result_subset, source_subset;
    ::sdfg::symbolic::Condition copy_condition = ::sdfg::symbolic::__true__();
    for (i = 0; i < result_tensor_info.shape().size(); i++) {
        int64_t dim = result_tensor_info.shape().at(i);
        auto indvar_container = builder.find_new_name("_i");
        builder.add_container(indvar_container, ::sdfg::types::Scalar(sdfg_index_type));
        indvars.push_back(indvar_container);
        auto indvar = ::sdfg::symbolic::symbol(indvar_container);
        result_subset.push_back(indvar);
        source_subset.push_back(::sdfg::symbolic::sub(indvar, low.at(i)));
        auto bound = ::sdfg::symbolic::integer(dim);
        auto condition = ::sdfg::symbolic::Lt(indvar, bound);
        auto init = ::sdfg::symbolic::zero();
        auto update = ::sdfg::symbolic::add(indvar, ::sdfg::symbolic::one());

        if (!::sdfg::symbolic::eq(low.at(i), ::sdfg::symbolic::zero()) ||
            !::sdfg::symbolic::eq(high.at(i), ::sdfg::symbolic::zero())) {
            copy_condition = ::sdfg::symbolic::
                And(copy_condition,
                    ::sdfg::symbolic::
                        And(::sdfg::symbolic::Ge(indvar, low.at(i)),
                            ::sdfg::symbolic::Lt(indvar, ::sdfg::symbolic::sub(bound, high.at(i)))));
        }

        auto& map = builder.add_map(
            *current_seq,
            indvar,
            condition,
            init,
            update,
            ::sdfg::structured_control_flow::ScheduleType_Sequential::create(),
            {},
            deb_info
        );
        current_seq = &map.root();
    }

    // Create if/else
    auto& if_else = builder.add_if_else(*current_seq, {}, deb_info);
    auto& copy_case =
        builder.add_case(if_else, ::sdfg::symbolic::Eq(copy_condition, ::sdfg::symbolic::__true__()), deb_info);
    auto& fill_case =
        builder.add_case(if_else, ::sdfg::symbolic::Eq(copy_condition, ::sdfg::symbolic::__false__()), deb_info);

    // Create copy case
    auto& copy_block = builder.add_block(copy_case, {}, deb_info);
    auto& source_access = builder.add_access(copy_block, source_container, deb_info);
    auto& copy_result_access = builder.add_access(copy_block, result_container, deb_info);
    auto& copy_tasklet =
        builder.add_tasklet(copy_block, ::sdfg::data_flow::TaskletCode::assign, "_out", {"_in"}, deb_info);
    builder.add_computational_memlet(
        copy_block, source_access, copy_tasklet, "_in", source_subset, *source_sdfg_tensor, deb_info
    );
    builder.add_computational_memlet(
        copy_block, copy_tasklet, "_out", copy_result_access, result_subset, *result_sdfg_tensor, deb_info
    );

    // Create block arguments
    Region& region = pad_op->getRegion();
    if (region.getBlocks().size() != 1) {
        return pad_op
                   ->emitOpError("Only exactly one block for the region of tensor.pad is currently supported but found "
                   )
               << region.getBlocks().size();
    }
    auto& block = region.getBlocks().front();
    if (block.getNumArguments() != indvars.size()) {
        return pad_op->emitOpError("number of block arguments != number of tensor dimensions: ")
               << block.getNumArguments() << " != " << indvars.size();
    }
    for (i = 0; i < block.getNumArguments(); i++) {
        BlockArgument argument = block.getArgument(i);
        auto argument_container = translator.get_or_create_container(argument);

        auto& fill_block = builder.add_block(fill_case, {}, deb_info);
        auto& indvar_access = builder.add_access(fill_block, indvars.at(i), deb_info);
        auto& argument_access = builder.add_access(fill_block, argument_container, deb_info);
        auto& tasklet =
            builder.add_tasklet(fill_block, ::sdfg::data_flow::TaskletCode::assign, "_out", {"_in"}, deb_info);
        builder.add_computational_memlet(fill_block, indvar_access, tasklet, "_in", {}, deb_info);
        builder.add_computational_memlet(fill_block, tasklet, "_out", argument_access, {}, deb_info);
    }

    // Translate operations in block until tensor.yield is reached
    translator.enter_sequence(fill_case);
    for (auto& op : block.getOperations()) {
        if (auto yield_op = llvm::dyn_cast_or_null<tensor::YieldOp>(op)) {
            // Create fill case
            auto yield_container = translator.get_or_create_container(yield_op.getValue());
            auto& fill_block = builder.add_block(translator.insertion_point(), {}, deb_info);
            auto& yield_access = builder.add_access(fill_block, yield_container, deb_info);
            auto& fill_result_access = builder.add_access(fill_block, result_container, deb_info);
            auto& tasklet =
                builder.add_tasklet(fill_block, ::sdfg::data_flow::TaskletCode::assign, "_out", {"_in"}, deb_info);
            builder.add_computational_memlet(fill_block, yield_access, tasklet, "_in", {}, deb_info);
            builder.add_computational_memlet(
                fill_block, tasklet, "_out", fill_result_access, result_subset, *result_sdfg_tensor, deb_info
            );
            break;
        } else {
            if (failed(translateOp(translator, &op))) {
                return failure();
            }
        }
    }
    translator.exit_sequence(fill_case);

    return success();
}

LogicalResult translateTensorOp(SDFGTranslator& translator, Operation* op) {
    return llvm::TypeSwitch<Operation*, LogicalResult>(op)
        .Case<tensor::CollapseShapeOp>([&](tensor::CollapseShapeOp collapse_op) {
            return translateTensorCollapseOp(translator, &collapse_op);
        })
        .Case<tensor::ConcatOp>([&](tensor::ConcatOp concat_op) {
            return translateTensorConcatOp(translator, &concat_op);
        })
        .Case<tensor::EmptyOp>([&](tensor::EmptyOp empty_op) { return translateTensorEmptyOp(translator, &empty_op); })
        .Case<tensor::ExpandShapeOp>([&](tensor::ExpandShapeOp expand_op) {
            return translateTensorExpandOp(translator, &expand_op);
        })
        .Case<tensor::ExtractOp>([&](tensor::ExtractOp extract_op) {
            return translateTensorExtractOp(translator, &extract_op);
        })
        .Case<tensor::PadOp>([&](tensor::PadOp pad_op) { return translateTensorPadOp(translator, &pad_op); })
        .Default([&](Operation* op) {
            return op->emitError("Unknown operation from tensor dialect encountered: ") << op->getName();
        });
}

} // namespace sdfg
} // namespace mlir
