#include "mlir/Target/SDFG/LinalgToSDFGTranslator.h"

#include <llvm-19/llvm/Support/Casting.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/LogicalResult.h>
#include <string>
#include <vector>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/SDFG/SDFGTranslator.h"
#include "mlir/Target/SDFG/helper.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/cmath_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/fill_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/tasklet_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/matmul_node.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/element.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/tensor.h"
#include "sdfg/types/type.h"

namespace mlir {
namespace sdfg {

template<typename ElemOp, ::sdfg::data_flow::TaskletCode fp_code, ::sdfg::data_flow::TaskletCode int_code>
LogicalResult translateLinalgElementwiseTaskletOp(SDFGTranslator& translator, ElemOp* add_op) {
    Value input1 = add_op->getInputs()[0];
    Value input2 = add_op->getInputs()[1];
    Value output = add_op->getOutputs()[0];
    Value result = add_op->getResultTensors()[0];

    auto& builder = translator.builder();
    auto input1_container = translator.get_or_create_container(input1);
    auto input2_container = translator.get_or_create_container(input2);
    auto output_container = translator.get_or_create_container(output);
    auto result_container = translator.get_or_create_container(result);

    auto result_tensor_type = llvm::dyn_cast<TensorType>(result.getType());
    auto tensor_info = translator.get_or_create_tensor_info(result_container, result_tensor_type);

    auto element_type = translator.convertType(result_tensor_type.getElementType());
    auto sdfg_tensor = tensor_info.get_sdfg_tensor(static_cast<::sdfg::types::Scalar&>(*element_type));

    ::sdfg::data_flow::TaskletCode code;
    if (::sdfg::types::is_floating_point(sdfg_tensor->primitive_type())) {
        code = fp_code;
    } else {
        code = int_code;
    }

    translator.add_reference(output_container, result_container);

    auto& block = builder.add_block(translator.insertion_point());
    auto& input1_access = builder.add_access(block, input1_container);
    auto& input2_access =
        *(input1_container == input2_container ? &input1_access : &builder.add_access(block, input2_container));
    auto& result_access = builder.add_access(block, result_container);
    auto& libnode = builder.add_library_node<::sdfg::math::tensor::TaskletTensorNode>(
        block,
        ::sdfg::DebugInfo(),
        code,
        std::vector<std::string>({"_out"}),
        std::vector<std::string>({"_in1", "_in2"}),
        sdfg_tensor->shape()
    );
    builder.add_computational_memlet(block, input1_access, libnode, "_in1", {}, *sdfg_tensor);
    builder.add_computational_memlet(block, input2_access, libnode, "_in2", {}, *sdfg_tensor);
    builder.add_computational_memlet(block, libnode, "_out", result_access, {}, *sdfg_tensor);

    return success();
}

template<typename ElemOp, ::sdfg::math::cmath::CMathFunction function>
LogicalResult translateLinalgElementwiseCMathOp(SDFGTranslator& translator, ElemOp* op) {
    Value input = op->getInputs()[0];
    Value output = op->getOutputs()[0];
    Value result = op->getResultTensors()[0];

    auto& builder = translator.builder();
    auto input_container = translator.get_or_create_container(input);
    auto output_container = translator.get_or_create_container(output);
    auto result_container = translator.get_or_create_container(result);

    auto result_tensor_type = llvm::dyn_cast<TensorType>(result.getType());
    auto tensor_info = translator.get_or_create_tensor_info(result_container, result_tensor_type);

    auto element_type = translator.convertType(result_tensor_type.getElementType());
    auto sdfg_tensor = tensor_info.get_sdfg_tensor(static_cast<::sdfg::types::Scalar&>(*element_type));

    translator.add_reference(output_container, result_container);

    auto& block = builder.add_block(translator.insertion_point());
    auto& input_access = builder.add_access(block, input_container);
    auto& result_access = builder.add_access(block, result_container);
    auto& libnode = builder.add_library_node<::sdfg::math::tensor::CMathTensorNode>(
        block,
        ::sdfg::DebugInfo(),
        function,
        std::vector<std::string>({"_out"}),
        std::vector<std::string>({"_in"}),
        sdfg_tensor->shape()
    );
    builder.add_computational_memlet(block, input_access, libnode, "_in", {}, *sdfg_tensor);
    builder.add_computational_memlet(block, libnode, "_out", result_access, {}, *sdfg_tensor);

    return success();
}

LogicalResult translateLinalgGenericOp(SDFGTranslator& translator, linalg::GenericOp* generic_op) {
    ArrayAttr iteration_types = generic_op->getIteratorTypes();
    ArrayAttr indexing_maps = generic_op->getIndexingMaps();
    OperandRange inputs = generic_op->getInputs();
    OperandRange outputs = generic_op->getOutputs();
    ResultRange results = generic_op->getResultTensors();
    if (inputs.size() + outputs.size() != indexing_maps.size()) {
        return generic_op->emitOpError("number of inputs + number of outputs != number of indexing_maps: ")
               << inputs.size() << " + " << outputs.size() << " != " << indexing_maps.size();
    }
    if (outputs.size() != results.size()) {
        return generic_op->emitOpError("number of outputs != number of results: ")
               << outputs.size() << " != " << results.size();
    }

    // Determine all dimensions
    std::vector<int64_t> dimensions;
    std::vector<AffineMap> affine_maps;
    dimensions.reserve(iteration_types.size());
    for (size_t i = 0; i < iteration_types.size(); i++) {
        // Determine candidates
        std::vector<int64_t> candidates;
        for (size_t j = 0; j < indexing_maps.size(); j++) {
            if (!llvm::dyn_cast_or_null<AffineMapAttr>(indexing_maps[j])) {
                return generic_op->emitOpError("Currently all indexing_maps must be of type AffineMapAttr");
            }
            auto affine_map_attr = llvm::dyn_cast<AffineMapAttr>(indexing_maps[j]);
            AffineMap affine_map = affine_map_attr.getAffineMap();
            if (affine_map.getNumDims() != iteration_types.size()) {
                return generic_op->emitOpError("Affine map ")
                       << affine_map_attr << " has dimension " << affine_map.getNumDims() << " but expected "
                       << iteration_types.size();
            }
            affine_maps.push_back(affine_map);

            // Skip scalars
            if (affine_map.getNumResults() == 0) {
                continue;
            }

            // Determine tensor type
            Type type;
            if (j < inputs.size()) {
                type = inputs[j].getType();
            } else {
                type = outputs[j - inputs.size()].getType();
            }
            if (!llvm::dyn_cast_or_null<TensorType>(type)) {
                return generic_op->emitOpError("Only supports scalars or tensors for inputs/outputs for now: ") << type;
            }
            auto tensor_type = llvm::dyn_cast<TensorType>(type);
            if (tensor_type.getShape().size() != affine_map.getNumResults()) {
                return generic_op->emitOpError("Mismatch between tensor shape ")
                       << tensor_type << " and number of affine results " << affine_map_attr;
            }

            // Iterate results of affine map
            AffineExpr dim_i = getAffineDimExpr(i, affine_map.getContext());
            for (size_t k = 0; k < affine_map.getNumResults(); k++) {
                // Replace i-th dimension with k-th tensor dimension in k-th result and simplify
                AffineExpr result = affine_map.getResult(k);
                AffineExpr tensor_dim = getAffineConstantExpr(tensor_type.getShape()[k], affine_map.getContext());
                AffineExpr candidate_expr = result.replace(dim_i, tensor_dim);
                AffineExpr simplified_candidate_expr =
                    simplifyAffineExpr(candidate_expr, affine_map.getNumDims(), affine_map.getNumSymbols());

                // If simplified expression is a constant, add it to candidates
                if (simplified_candidate_expr.getKind() != AffineExprKind::Constant) {
                    continue;
                }
                auto constant_expr = llvm::dyn_cast<AffineConstantExpr>(simplified_candidate_expr);
                candidates.push_back(constant_expr.getValue());
            }
        }

        // Check that all candidates are equal
        if (candidates.empty()) {
            return generic_op->emitOpError("Could not found candidates for dimension: ") << i;
        }
        int64_t dim = candidates.at(0);
        for (int64_t candidate : candidates) {
            if (candidate != dim) {
                return generic_op->emitOpError("Candidate mismatch for dimension ")
                       << i << ": " << candidate << " != " << dim;
            }
        }

        // Add dimension size
        dimensions.push_back(dim);
    }

    // Create containers
    auto& builder = translator.builder();
    std::vector<std::string> input_containers, output_containers, result_containers;
    input_containers.reserve(inputs.size());
    for (auto input : inputs) {
        input_containers.push_back(translator.get_or_create_container(input));
    }
    output_containers.reserve(outputs.size());
    for (auto output : outputs) {
        output_containers.push_back(translator.get_or_create_container(output));
    }
    result_containers.reserve(results.size());
    for (auto result : results) {
        result_containers.push_back(translator.get_or_create_container(result));
    }

    // Create references
    for (size_t i = 0; i < outputs.size(); i++) {
        translator.add_reference(output_containers.at(i), result_containers.at(i));
    }

    // Create loops
    ::sdfg::structured_control_flow::Sequence* current_seq = &translator.insertion_point();
    std::vector<::sdfg::symbolic::Symbol> indvars;
    for (size_t i = 0; i < iteration_types.size(); i++) {
        auto attr = iteration_types[i];
        if (!llvm::dyn_cast_or_null<linalg::IteratorTypeAttr>(attr)) {
            return generic_op->emitOpError("Expected a string for attribute in iteration_types: ") << attr;
        }
        auto iterator_type_attr = llvm::dyn_cast<linalg::IteratorTypeAttr>(attr);

        auto indvar_container = builder.find_new_name("_i");
        builder.add_container(indvar_container, ::sdfg::types::Scalar(::sdfg::types::PrimitiveType::Int64));
        auto indvar = ::sdfg::symbolic::symbol(indvar_container);
        indvars.push_back(indvar);
        auto condition = ::sdfg::symbolic::Lt(indvar, ::sdfg::symbolic::integer(dimensions.at(i)));
        auto init = ::sdfg::symbolic::zero();
        auto update = ::sdfg::symbolic::add(indvar, ::sdfg::symbolic::one());

        if (iterator_type_attr.getValue() == utils::IteratorType::reduction) {
            auto& for_loop = builder.add_for(*current_seq, indvar, condition, init, update);
            current_seq = &for_loop.root();
        } else {
            auto& map = builder.add_map(
                *current_seq,
                indvar,
                condition,
                init,
                update,
                ::sdfg::structured_control_flow::ScheduleType_Sequential::create()
            );
            current_seq = &map.root();
        }
    }

    // Create tensor mapping to scalars
    Region& region = generic_op->getRegion();
    if (region.getBlocks().size() != 1) {
        return generic_op->emitOpError(
                   "Only exactly one block for the region of linalg.generic is currently supported but found "
               )
               << region.getBlocks().size();
    }
    auto& block = region.getBlocks().front();
    if (block.getNumArguments() != inputs.size() + outputs.size()) {
        return generic_op->emitOpError("number of block arguments != number of inputs + number of outputs: ")
               << block.getNumArguments() << " != " << inputs.size() << " + " << outputs.size();
    }
    for (size_t i = 0; i < block.getNumArguments(); i++) {
        BlockArgument argument = block.getArgument(i);
        auto argument_container = translator.get_or_create_container(argument);
        auto outer_container = (i < inputs.size()) ? input_containers.at(i) : output_containers.at(inputs.size() - i);
        auto outer_type = (i < inputs.size()) ? inputs[i].getType() : outputs[inputs.size() - i].getType();

        auto& sdfg_block = builder.add_block(*current_seq);
        auto& outer_access = builder.add_access(sdfg_block, outer_container);
        auto& argument_access = builder.add_access(sdfg_block, argument_container);
        auto& tasklet = builder.add_tasklet(sdfg_block, ::sdfg::data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(sdfg_block, tasklet, "_out", argument_access, {});

        if (is_sdfg_primitive(outer_type)) {
            builder.add_computational_memlet(sdfg_block, outer_access, tasklet, "_in", {});
        } else if (is_tensor_of_sdfg_primitive(outer_type)) {
            auto tensor_type = llvm::dyn_cast<TensorType>(outer_type);
            auto tensor_info = translator.get_or_create_tensor_info(outer_container, tensor_type);

            auto element_type = translator.convertType(tensor_type.getElementType());
            auto sdfg_tensor = tensor_info.get_sdfg_tensor(static_cast<::sdfg::types::Scalar&>(*element_type));

            auto subset = affine_map_to_subset(affine_maps.at(i), indvars);
            builder.add_computational_memlet(sdfg_block, outer_access, tasklet, "_in", subset, *sdfg_tensor);
        } else {
            return generic_op
                ->emitOpError("Only primitive types or tensors are supported for linalg.generic block arguments");
        }
    }

    // Translate operations in block until linalg.yield is reached
    translator.enter_sequence(*current_seq);
    for (auto& op : block.getOperations()) {
        if (auto yield_op = llvm::dyn_cast_or_null<linalg::YieldOp>(op)) {
            OperandRange values = yield_op.getValues();
            if (values.size() != results.size()) {
                return yield_op->emitOpError("emits ")
                       << values.size() << " values but linalg.generic expects " << results.size();
            }

            for (size_t i = 0; i < values.size(); i++) {
                auto value_container = translator.get_or_create_container(values[i]);

                auto& sdfg_block = builder.add_block(translator.insertion_point());
                auto& value_access = builder.add_access(sdfg_block, value_container);
                auto& result_access = builder.add_access(sdfg_block, result_containers.at(i));
                auto& tasklet =
                    builder.add_tasklet(sdfg_block, ::sdfg::data_flow::TaskletCode::assign, "_out", {"_in"});
                builder.add_computational_memlet(sdfg_block, value_access, tasklet, "_in", {});

                auto tensor_type = llvm::dyn_cast<TensorType>(results[i].getType());
                auto tensor_info = translator.get_or_create_tensor_info(result_containers.at(i), tensor_type);

                auto element_type = translator.convertType(tensor_type.getElementType());
                auto sdfg_tensor = tensor_info.get_sdfg_tensor(static_cast<::sdfg::types::Scalar&>(*element_type));

                auto subset = affine_map_to_subset(affine_maps.at(inputs.size() + i), indvars);
                builder.add_computational_memlet(sdfg_block, tasklet, "_out", result_access, subset, *sdfg_tensor);
            }
            break;
        } else {
            if (failed(translateOp(translator, &op))) {
                return failure();
            }
        }
    }
    translator.exit_sequence(*current_seq);

    return success();
}

LogicalResult translateLinalgOp(SDFGTranslator& translator, Operation* op) {
    return llvm::TypeSwitch<Operation*, LogicalResult>(op)
        .Case<linalg::AddOp>([&](linalg::AddOp add_op) {
            return translateLinalgElementwiseTaskletOp<
                linalg::AddOp,
                ::sdfg::data_flow::TaskletCode::fp_add,
                ::sdfg::data_flow::TaskletCode::int_add>(translator, &add_op);
        })
        .Case<linalg::DivOp>([&](linalg::DivOp div_op) {
            return translateLinalgElementwiseTaskletOp<
                linalg::DivOp,
                ::sdfg::data_flow::TaskletCode::fp_div,
                ::sdfg::data_flow::TaskletCode::int_sdiv>(translator, &div_op);
        })
        .Case<linalg::MulOp>([&](linalg::MulOp div_op) {
            return translateLinalgElementwiseTaskletOp<
                linalg::MulOp,
                ::sdfg::data_flow::TaskletCode::fp_mul,
                ::sdfg::data_flow::TaskletCode::int_mul>(translator, &div_op);
        })
        .Case<linalg::SubOp>([&](linalg::SubOp div_op) {
            return translateLinalgElementwiseTaskletOp<
                linalg::SubOp,
                ::sdfg::data_flow::TaskletCode::fp_sub,
                ::sdfg::data_flow::TaskletCode::int_sub>(translator, &div_op);
        })
        .Case<linalg::AbsOp>([&](linalg::AbsOp abs_op) {
            return translateLinalgElementwiseCMathOp<
                linalg::AbsOp,
                ::sdfg::math::cmath::CMathFunction::fabs>(translator, &abs_op);
        })
        .Case<linalg::CeilOp>([&](linalg::CeilOp ceil_op) {
            return translateLinalgElementwiseCMathOp<
                linalg::CeilOp,
                ::sdfg::math::cmath::CMathFunction::ceil>(translator, &ceil_op);
        })
        .Case<linalg::ErfOp>([&](linalg::ErfOp erf_op) {
            return translateLinalgElementwiseCMathOp<
                linalg::ErfOp,
                ::sdfg::math::cmath::CMathFunction::erf>(translator, &erf_op);
        })
        .Case<linalg::ExpOp>([&](linalg::ExpOp exp_op) {
            return translateLinalgElementwiseCMathOp<
                linalg::ExpOp,
                ::sdfg::math::cmath::CMathFunction::exp>(translator, &exp_op);
        })
        .Case<linalg::FloorOp>([&](linalg::FloorOp floor_op) {
            return translateLinalgElementwiseCMathOp<
                linalg::FloorOp,
                ::sdfg::math::cmath::CMathFunction::floor>(translator, &floor_op);
        })
        .Case<linalg::LogOp>([&](linalg::LogOp log_op) {
            return translateLinalgElementwiseCMathOp<
                linalg::LogOp,
                ::sdfg::math::cmath::CMathFunction::log>(translator, &log_op);
        })
        .Case<linalg::MaxOp>([&](linalg::MaxOp max_op) {
            return translateLinalgElementwiseCMathOp<
                linalg::MaxOp,
                ::sdfg::math::cmath::CMathFunction::fmax>(translator, &max_op);
        })
        .Case<linalg::MinOp>([&](linalg::MinOp min_op) {
            return translateLinalgElementwiseCMathOp<
                linalg::MinOp,
                ::sdfg::math::cmath::CMathFunction::fmin>(translator, &min_op);
        })
        .Case<linalg::PowFOp>([&](linalg::PowFOp powf_op) {
            return translateLinalgElementwiseCMathOp<
                linalg::PowFOp,
                ::sdfg::math::cmath::CMathFunction::pow>(translator, &powf_op);
        })
        .Case<linalg::RoundOp>([&](linalg::RoundOp round_op) {
            return translateLinalgElementwiseCMathOp<
                linalg::RoundOp,
                ::sdfg::math::cmath::CMathFunction::round>(translator, &round_op);
        })
        .Case<linalg::SqrtOp>([&](linalg::SqrtOp sqrt_op) {
            return translateLinalgElementwiseCMathOp<
                linalg::SqrtOp,
                ::sdfg::math::cmath::CMathFunction::sqrt>(translator, &sqrt_op);
        })
        .Case<linalg::TanhOp>([&](linalg::TanhOp tanh_op) {
            return translateLinalgElementwiseCMathOp<
                linalg::TanhOp,
                ::sdfg::math::cmath::CMathFunction::tanh>(translator, &tanh_op);
        })
        .Case<linalg::GenericOp>([&](linalg::GenericOp generic_op) {
            return translateLinalgGenericOp(translator, &generic_op);
        })
        .Case<linalg::FillOp>([&](linalg::FillOp fill_op) { return translateLinalgFillOp(translator, &fill_op); })
        .Case<linalg::MatmulOp>([&](linalg::MatmulOp matmul_op) {
            return translateLinalgMatmulOp(translator, &matmul_op);
        })
        .Case<linalg::TransposeOp>([&](linalg::TransposeOp transpose_op) {
            return translateLinalgTransposeOp(translator, &transpose_op);
        })
        .Default([&](Operation* op) { return op->emitError("Unknown operation from linalg dialect encountered"); });
}

LogicalResult translateLinalgFillOp(SDFGTranslator& translator, linalg::FillOp* op) {
    auto& sequence = translator.insertion_point();

    Value value = op->value();
    Value output = op->output();
    Value result = op->result();

    auto value_container = translator.get_or_create_container(value);
    auto output_container = translator.get_or_create_container(output);
    auto result_container = translator.get_or_create_container(result);

    translator.add_reference(output_container, result_container);

    auto& block = translator.builder().add_block(sequence);

    auto result_type = dyn_cast_or_null<RankedTensorType>(result.getType());
    if (!result_type) {
        return op->emitError("Only ranked tensor result type is supported for now");
    }

    ::sdfg::types::Scalar base_type(translator.convertType(value.getType())->primitive_type());

    auto tensor_info = translator.get_or_create_tensor_info(translator.get_or_create_container(result), result_type);

    auto tensor_type = tensor_info.get_sdfg_tensor(base_type);


    auto& lib_node =
        translator.builder()
            .add_library_node<::sdfg::math::tensor::FillNode>(block, ::sdfg::DebugInfo(), tensor_type->shape());

    if (auto constant_op = dyn_cast_or_null<arith::ConstantOp>(value.getDefiningOp())) {
        auto& inaccess = translator.builder().add_constant(
            block, translator.convertTypedAttr(constant_op.getValue()), *translator.convertType(constant_op.getType())
        );
        translator.builder().add_computational_memlet(block, inaccess, lib_node, "X", {}, base_type);
    } else {
        auto& in_access = translator.builder().add_access(block, translator.get_or_create_container(value));
        translator.builder().add_computational_memlet(block, in_access, lib_node, "X", {}, base_type);
    }

    auto& out_access = translator.builder().add_access(block, translator.get_or_create_container(result));
    translator.builder().add_computational_memlet(block, lib_node, "Y", out_access, {}, *tensor_type);

    return success();
}

LogicalResult translateLinalgMatmulOp(SDFGTranslator& translator, linalg::MatmulOp* op) {
    auto& sequence = translator.insertion_point();

    auto output = op->getOutputs()[0];
    auto result = op->getResult(0);

    auto output_container = translator.get_or_create_container(output);
    auto result_container = translator.get_or_create_container(result);

    translator.add_reference(output_container, result_container);

    auto& block = translator.builder().add_block(sequence);

    // For now, only handle 2D matmul with no transposes or broadcasts
    auto lhs_type = dyn_cast_or_null<RankedTensorType>(op->getOperand(0).getType());
    auto rhs_type = dyn_cast_or_null<RankedTensorType>(op->getOperand(1).getType());
    auto output_type = dyn_cast_or_null<RankedTensorType>(op->getResult(0).getType());
    if (!lhs_type || !rhs_type || !output_type || lhs_type.getRank() != 2 || rhs_type.getRank() != 2 ||
        output_type.getRank() != 2) {
        return op->emitError("Only 2D matmul is supported for now");
    }

    auto in_container_lhs = translator.get_or_create_container(op->getOperand(0));
    auto in_container_rhs = translator.get_or_create_container(op->getOperand(1));
    auto out_container = translator.get_or_create_container(op->getResult(0));

    ::sdfg::data_flow::AccessNode* lhs_access = &translator.builder().add_access(block, in_container_lhs);
    ::sdfg::data_flow::AccessNode* rhs_access = &translator.builder().add_access(block, in_container_rhs);

    if (in_container_lhs == in_container_rhs) {
        rhs_access = lhs_access;
    } else {
        rhs_access = &translator.builder().add_access(block, in_container_rhs);
    }

    auto& tensor_info_lhs = translator.get_or_create_tensor_info(in_container_lhs, lhs_type);
    auto& tensor_info_rhs = translator.get_or_create_tensor_info(in_container_rhs, rhs_type);
    auto& tensor_info_out = translator.get_or_create_tensor_info(out_container, output_type);

    // check if offsets are 0 for all tensors since we don't support partial tensors for now
    if (tensor_info_lhs.offset() != 0 || tensor_info_rhs.offset() != 0 || tensor_info_out.offset() != 0) {
        return op->emitError("Only tensors with 0 offset are supported for now");
    }

    ::sdfg::symbolic::MultiExpression shape_lhs;
    for (auto entry : tensor_info_lhs.shape()) {
        shape_lhs.push_back(::sdfg::symbolic::integer(entry));
    }
    ::sdfg::symbolic::MultiExpression shape_rhs;
    for (auto entry : tensor_info_rhs.shape()) {
        shape_rhs.push_back(::sdfg::symbolic::integer(entry));
    }
    ::sdfg::symbolic::MultiExpression shape_out;
    for (auto entry : tensor_info_out.shape()) {
        shape_out.push_back(::sdfg::symbolic::integer(entry));
    }

    ::sdfg::symbolic::MultiExpression strides_lhs;
    for (auto entry : tensor_info_lhs.strides()) {
        strides_lhs.push_back(::sdfg::symbolic::integer(entry));
    }
    ::sdfg::symbolic::MultiExpression strides_rhs;
    for (auto entry : tensor_info_rhs.strides()) {
        strides_rhs.push_back(::sdfg::symbolic::integer(entry));
    }

    auto& libnode = translator.builder().add_library_node<::sdfg::math::tensor::MatMulNode>(
        block,
        ::sdfg::DebugInfo(),
        shape_lhs,
        shape_rhs,
        strides_lhs,
        strides_rhs,
        /*offset_a=*/::sdfg::symbolic::zero(),
        /*offset_b=*/::sdfg::symbolic::zero()
    );

    auto lhs_primitive_type = translator.convertType(lhs_type)->primitive_type();
    ::sdfg::types::Tensor lhs_tensor_type(lhs_primitive_type, shape_lhs, strides_lhs);
    auto rhs_primitive_type = translator.convertType(rhs_type)->primitive_type();
    ::sdfg::types::Tensor rhs_tensor_type(rhs_primitive_type, shape_rhs, strides_rhs);
    auto output_primitive_type = translator.convertType(output_type)->primitive_type();
    ::sdfg::types::Tensor output_tensor_type(output_primitive_type, shape_out);

    translator.builder().add_computational_memlet(block, *lhs_access, libnode, "A", {}, lhs_tensor_type);
    translator.builder().add_computational_memlet(block, *rhs_access, libnode, "B", {}, rhs_tensor_type);

    auto& write_access = translator.builder().add_access(block, out_container);

    translator.builder().add_computational_memlet(block, libnode, "Y", write_access, {}, output_tensor_type);

    return success();
}

LogicalResult translateLinalgTransposeOp(SDFGTranslator& translator, linalg::TransposeOp* op) {
    Value input = op->getInput();
    Value result = op->getResult()[0];

    // Check that input and output types are ranked tensors
    auto input_tensor_type = dyn_cast_or_null<TensorType>(input.getType());
    auto result_tensor_type = dyn_cast_or_null<TensorType>(result.getType());
    if (!input_tensor_type || !result_tensor_type) {
        return op->emitError("Input and output types must be ranked tensors");
    }

    auto permutation = op->getPermutation();

    auto in_container = translator.get_or_create_container(input);
    auto out_container = translator.get_or_create_container(result);

    translator.add_reference(in_container, out_container);

    // Compute and store tensor info for input and output tensors. This will be used for libnode generation later on.
    auto& in_tensor_info = translator.get_or_create_tensor_info(in_container, input_tensor_type);

    auto out_tensor_info = in_tensor_info.transpose(permutation);
    translator.tensor_info_map().insert({out_container, out_tensor_info});

    return success();
}

} // namespace sdfg
} // namespace mlir
