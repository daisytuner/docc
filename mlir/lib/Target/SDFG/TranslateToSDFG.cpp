#include "mlir/Target/SDFG/TranslateToSDFG.h"

#include <llvm/ADT/ScopedHashTable.h>
#include <llvm/ADT/TypeSwitch.h>
#include <memory>

#include "mlir/Dialect/SDFG/IR/SDFG.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include <sdfg/analysis/analysis.h>
#include <sdfg/builder/structured_sdfg_builder.h>
#include <sdfg/codegen/code_generators/c_code_generator.h>
#include <sdfg/codegen/instrumentation/arg_capture_plan.h>
#include <sdfg/codegen/instrumentation/instrumentation_plan.h>
#include <sdfg/data_flow/access_node.h>
#include <sdfg/data_flow/library_node.h>
#include <sdfg/data_flow/library_nodes/math/blas/blas_node.h>
#include <sdfg/data_flow/library_nodes/math/blas/gemm_node.h>
#include <sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/fill_node.h>
#include <sdfg/data_flow/tasklet.h>
#include <sdfg/element.h>
#include <sdfg/function.h>
#include <sdfg/serializer/json_serializer.h>
#include <sdfg/structured_control_flow/block.h>
#include <sdfg/structured_control_flow/sequence.h>
#include <sdfg/symbolic/symbolic.h>
#include <sdfg/types/scalar.h>
#include <sdfg/types/type.h>

namespace mlir {
namespace sdfg {

class SDFGTranslator {
    using Builder = ::sdfg::builder::StructuredSDFGBuilder;
    using Sequence = ::sdfg::structured_control_flow::Sequence;
    using Block = ::sdfg::structured_control_flow::Block;
    std::list<Builder> builders_;

    using ValueMap = llvm::ScopedHashTableScope<Value, std::string>;
    llvm::ScopedHashTable<Value, std::string> value_map_;
    size_t value_counter_ = 0;

public:
    std::string get_or_create_container(Builder& builder, Value val, bool argument = false) {
        if (!this->value_map_.count(val)) {
            this->value_map_.insert(val, "_" + std::to_string(value_counter_++));
        }
        std::string container = *this->value_map_.begin(val);
        auto type = convertType(val.getType());
        if (builder.subject().exists(container)) {
            assert(builder.subject().type(container) == *type);
            assert(!argument || builder.subject().is_argument(container));
        } else {
            builder.add_container(container, *type, argument);
        }
        return container;
    }

    std::unique_ptr<::sdfg::types::IType> convertType(const Type mlir_type) {
        if (mlir_type.isInteger(1)) {
            return std::make_unique<::sdfg::types::Scalar>(::sdfg::types::PrimitiveType::Bool);
        } else if (mlir_type.isSignedInteger(8) || mlir_type.isSignlessInteger(8)) {
            return std::make_unique<::sdfg::types::Scalar>(::sdfg::types::PrimitiveType::Int8);
        } else if (mlir_type.isSignedInteger(16) || mlir_type.isSignlessInteger(16)) {
            return std::make_unique<::sdfg::types::Scalar>(::sdfg::types::PrimitiveType::Int16);
        } else if (mlir_type.isSignedInteger(32) || mlir_type.isSignlessInteger(32)) {
            return std::make_unique<::sdfg::types::Scalar>(::sdfg::types::PrimitiveType::Int32);
        } else if (mlir_type.isSignedInteger(64) || mlir_type.isSignlessInteger(64)) {
            return std::make_unique<::sdfg::types::Scalar>(::sdfg::types::PrimitiveType::Int64);
        } else if (mlir_type.isSignedInteger(128) || mlir_type.isSignlessInteger(128)) {
            return std::make_unique<::sdfg::types::Scalar>(::sdfg::types::PrimitiveType::Int128);
        } else if (mlir_type.isUnsignedInteger(8)) {
            return std::make_unique<::sdfg::types::Scalar>(::sdfg::types::PrimitiveType::UInt8);
        } else if (mlir_type.isUnsignedInteger(16)) {
            return std::make_unique<::sdfg::types::Scalar>(::sdfg::types::PrimitiveType::UInt16);
        } else if (mlir_type.isUnsignedInteger(32)) {
            return std::make_unique<::sdfg::types::Scalar>(::sdfg::types::PrimitiveType::UInt32);
        } else if (mlir_type.isUnsignedInteger(64)) {
            return std::make_unique<::sdfg::types::Scalar>(::sdfg::types::PrimitiveType::UInt64);
        } else if (mlir_type.isUnsignedInteger(128)) {
            return std::make_unique<::sdfg::types::Scalar>(::sdfg::types::PrimitiveType::UInt128);
        } else if (mlir_type.isF16()) {
            return std::make_unique<::sdfg::types::Scalar>(::sdfg::types::PrimitiveType::Half);
        } else if (mlir_type.isBF16()) {
            return std::make_unique<::sdfg::types::Scalar>(::sdfg::types::PrimitiveType::BFloat);
        } else if (mlir_type.isF32()) {
            return std::make_unique<::sdfg::types::Scalar>(::sdfg::types::PrimitiveType::Float);
        } else if (mlir_type.isF64()) {
            return std::make_unique<::sdfg::types::Scalar>(::sdfg::types::PrimitiveType::Double);
        } else if (mlir_type.isF80()) {
            return std::make_unique<::sdfg::types::Scalar>(::sdfg::types::PrimitiveType::X86_FP80);
        } else if (mlir_type.isF128()) {
            return std::make_unique<::sdfg::types::Scalar>(::sdfg::types::PrimitiveType::FP128);
        } else if (auto vector_type = dyn_cast_or_null<VectorType>(mlir_type)) {
            auto base_type = this->convertType(vector_type.getElementType());
            if (!base_type) {
                return nullptr;
            }
            return std::make_unique<::sdfg::types::Pointer>(*base_type);
        } else if (auto tensor_type = dyn_cast_or_null<TensorType>(mlir_type)) {
            auto base_type = this->convertType(tensor_type.getElementType());
            if (!base_type) {
                return nullptr;
            }
            return std::make_unique<::sdfg::types::Pointer>(*base_type);
        }
        return nullptr;
    }

    std::string convertTypedAttr(const TypedAttr attr) {
        return llvm::TypeSwitch<TypedAttr, std::string>(attr)
            .Case<FloatAttr>([](FloatAttr attr) { return std::to_string(attr.getValue().convertToDouble()); })
            .Case<IntegerAttr>([](IntegerAttr attr) { return std::to_string(attr.getInt()); })
            .Default([](TypedAttr attr) { return ""; });
    }

    LogicalResult translate(Operation* op) {
        return llvm::TypeSwitch<Operation*, LogicalResult>(op)
            .Case<ModuleOp>([&](ModuleOp module_op) { return this->translateModuleOp(&module_op); })
            .Default([&](Operation* op) { return op->emitError("A module op is required here"); });
    }

    LogicalResult translateModuleOp(ModuleOp* module_op) {
        for (auto& op : module_op->getRegion().getOps()) {
            LogicalResult status = llvm::TypeSwitch<Operation*, LogicalResult>(&op)
                                       .Case<SDFGOp>([&](SDFGOp sdfg_op) { return this->translateSDFGOp(&sdfg_op); })
                                       .Default([&](Operation* op) {
                                           return op->emitError("A sdfg op is required here");
                                       });
            if (failed(status)) {
                return failure();
            }
        }
        return success();
    }

    LogicalResult translateSDFGOp(SDFGOp* sdfg_op) {
        std::string sdfg_name = sdfg_op->getSymName().data();
        sdfg_name = "__docc_" + sdfg_name;
        this->builders_.push_back(Builder(sdfg_name, ::sdfg::FunctionType_CPU));
        auto& builder = this->builders_.back();
        ValueMap value_map_scope(this->value_map_);

        // Return type
        auto function_type = sdfg_op->getFunctionType();
        assert(function_type.getNumResults() <= 1);
        if (function_type.getNumResults() == 1) {
            auto return_type = this->convertType(function_type.getResult(0));
            if (!return_type) {
                return sdfg_op->emitError("Could not convert type ") << function_type.getResult(0) << " to SDFG type";
            }
            builder.set_return_type(*return_type);
        }

        // Arguments
        for (auto arg : sdfg_op->getRegion().getArguments()) {
            this->get_or_create_container(builder, arg, true);
        }

        // Region
        for (auto& op : sdfg_op->getRegion().getOps()) {
            if (failed(this->translateOperation(builder, builder.subject().root(), &op))) {
                return failure();
            }
        }

        return success();
    }

    LogicalResult translateOperation(Builder& builder, Sequence& parent, Operation* op) {
        return llvm::TypeSwitch<Operation*, LogicalResult>(op)
            .Case<BlockOp>([&](BlockOp block_op) { return this->translateBlockOp(builder, parent, &block_op); })
            .Case<ReturnOp>([&](ReturnOp return_op) { return this->translateReturnOp(builder, parent, &return_op); })
            .Case<TensorConstantOp>([&](TensorConstantOp tensor_constant_op) {
                return this->translateTensorConstantOp(builder, parent, &tensor_constant_op);
            })
            .Default([&](Operation* op) {
                return op->emitError("Unknown control flow operation. Could not translate.");
            });
    }

    LogicalResult translateBlockOp(Builder& builder, Sequence& parent, BlockOp* block_op) {
        auto& block = builder.add_block(parent);
        std::unordered_map<mlir::Attribute::ImplType*, ::sdfg::data_flow::Tasklet*> tasklets;
        std::unordered_map<mlir::Attribute::ImplType*, ::sdfg::data_flow::LibraryNode*> libnodes;
        std::unordered_map<mlir::detail::ValueImpl*, ::sdfg::data_flow::AccessNode*> access_nodes;
        std::unordered_map<mlir::detail::ValueImpl*, ::sdfg::data_flow::ConstantNode*> constant_nodes;

        // Capture results
        std::vector<Value> results;
        for (auto value : block_op->getResults()) {
            results.push_back(value);
        }

        // Create all nodes
        for (auto& op : block_op->getRegion().getOps()) {
            LogicalResult result =
                llvm::TypeSwitch<Operation*, LogicalResult>(&op)
                    .Case<TaskletOp>([&](TaskletOp tasklet_op) {
                        auto code = static_cast<::sdfg::data_flow::TaskletCode>(tasklet_op.getCode());
                        std::vector<std::string> inputs;
                        switch (arity(tasklet_op.getCode())) {
                            case 1:
                                inputs.push_back("_in");
                                break;
                            case 2:
                                inputs.push_back("_in1");
                                inputs.push_back("_in2");
                                break;
                            case 3:
                                inputs.push_back("_in1");
                                inputs.push_back("_in2");
                                inputs.push_back("_in3");
                                break;
                            default:
                                return failure();
                        }
                        tasklets
                            .insert({tasklet_op.getLoc()->getImpl(), &builder.add_tasklet(block, code, "_out", inputs)}
                            );
                        return success();
                    })
                    .Case<FillOp>([&](FillOp fill_op) {
                        std::vector<::sdfg::symbolic::Expression> shape;
                        Type type = fill_op.getOutput().getType();
                        if (auto vector_type = dyn_cast<VectorType>(type)) {
                            for (int64_t dim : vector_type.getShape()) {
                                shape.push_back(::sdfg::symbolic::integer(dim));
                            }
                        } else if (auto tensor_type = dyn_cast<TensorType>(type)) {
                            for (int64_t dim : tensor_type.getShape()) {
                                shape.push_back(::sdfg::symbolic::integer(dim));
                            }
                        } else {
                            fill_op.emitOpError("does not use a vector or tensor");
                            return failure();
                        }
                        libnodes.insert(
                            {fill_op.getLoc()->getImpl(),
                             &builder.add_library_node<::sdfg::math::tensor::FillNode>(block, ::sdfg::DebugInfo(), shape)
                            }
                        );
                        return success();
                    })
                    .Case<MatmulOp>([&](MatmulOp matmul_op) {
                        ::sdfg::math::blas::BLAS_Precision precision;
                        ::sdfg::types::PrimitiveType precision_type;
                        ::sdfg::symbolic::Expression m, n, k, lda, ldb, ldc;
                        Type type = matmul_op.getOutput().getType();
                        if (auto vector_type = dyn_cast<VectorType>(type)) {
                            if (vector_type.getElementType().isF16()) {
                                precision = ::sdfg::math::blas::BLAS_Precision::h;
                                precision_type = ::sdfg::types::PrimitiveType::Half;
                            } else if (vector_type.getElementType().isF32()) {
                                precision = ::sdfg::math::blas::BLAS_Precision::s;
                                precision_type = ::sdfg::types::PrimitiveType::Float;
                            } else if (vector_type.getElementType().isF64()) {
                                precision = ::sdfg::math::blas::BLAS_Precision::d;
                                precision_type = ::sdfg::types::PrimitiveType::Double;
                            } else {
                                matmul_op
                                    ->emitOpError("has unsupported element type. Only f16, f32, and f64 are supported."
                                    );
                                return failure();
                            }
                            auto lhs_type = dyn_cast<VectorType>(matmul_op.getLhs().getType());
                            auto rhs_type = dyn_cast<VectorType>(matmul_op.getRhs().getType());
                            m = ::sdfg::symbolic::integer(lhs_type.getDimSize(0));
                            n = ::sdfg::symbolic::integer(rhs_type.getDimSize(1));
                            k = ::sdfg::symbolic::integer(lhs_type.getDimSize(1));
                            lda = ::sdfg::symbolic::integer(lhs_type.getDimSize(1));
                            ldb = ::sdfg::symbolic::integer(rhs_type.getDimSize(1));
                            ldc = ::sdfg::symbolic::integer(rhs_type.getDimSize(1));
                        } else if (auto tensor_type = dyn_cast<TensorType>(type)) {
                            if (tensor_type.getElementType().isF16()) {
                                precision = ::sdfg::math::blas::BLAS_Precision::h;
                                precision_type = ::sdfg::types::PrimitiveType::Half;
                            } else if (tensor_type.getElementType().isF32()) {
                                precision = ::sdfg::math::blas::BLAS_Precision::s;
                                precision_type = ::sdfg::types::PrimitiveType::Float;
                            } else if (tensor_type.getElementType().isF64()) {
                                precision = ::sdfg::math::blas::BLAS_Precision::d;
                                precision_type = ::sdfg::types::PrimitiveType::Double;
                            } else {
                                matmul_op
                                    ->emitOpError("has unsupported element type. Only f16, f32, and f64 are supported."
                                    );
                                return failure();
                            }
                            auto lhs_type = dyn_cast<TensorType>(matmul_op.getLhs().getType());
                            auto rhs_type = dyn_cast<TensorType>(matmul_op.getRhs().getType());
                            m = ::sdfg::symbolic::integer(lhs_type.getDimSize(0));
                            n = ::sdfg::symbolic::integer(rhs_type.getDimSize(1));
                            k = ::sdfg::symbolic::integer(lhs_type.getDimSize(1));
                            lda = ::sdfg::symbolic::integer(lhs_type.getDimSize(1));
                            ldb = ::sdfg::symbolic::integer(rhs_type.getDimSize(1));
                            ldc = ::sdfg::symbolic::integer(rhs_type.getDimSize(1));
                        } else {
                            matmul_op->emitError("has unsupported type. Only vectors and tensors are supported.");
                            return failure();
                        }
                        auto& libnode = builder.add_library_node<::sdfg::math::blas::GEMMNode>(
                            block,
                            ::sdfg::DebugInfo(),
                            ::sdfg::math::blas::ImplementationType_BLAS,
                            precision,
                            ::sdfg::math::blas::BLAS_Layout::RowMajor,
                            ::sdfg::math::blas::BLAS_Transpose::No,
                            ::sdfg::math::blas::BLAS_Transpose::No,
                            m,
                            n,
                            k,
                            lda,
                            ldb,
                            ldc
                        );
                        ::sdfg::types::Scalar one_type(precision_type);
                        auto& one = builder.add_constant(block, "1.0", one_type);
                        builder.add_computational_memlet(block, one, libnode, "__alpha", {}, one_type);
                        builder.add_computational_memlet(block, one, libnode, "__beta", {}, one_type);
                        libnodes.insert({matmul_op.getLoc()->getImpl(), &libnode});
                        return success();
                    })
                    .Case<MemletOp>([&](MemletOp memlet_op) {
                        if (dyn_cast_or_null<TaskletOp>(memlet_op.getInput().getDefiningOp()) ||
                            dyn_cast_or_null<FillOp>(memlet_op.getInput().getDefiningOp()) ||
                            dyn_cast_or_null<MatmulOp>(memlet_op.getInput().getDefiningOp())) {
                            access_nodes.insert(
                                {memlet_op.getOutput().getImpl(),
                                 &builder
                                      .add_access(block, this->get_or_create_container(builder, memlet_op.getOutput()))}
                            );
                        } else {
                            access_nodes.insert(
                                {memlet_op.getInput().getImpl(),
                                 &builder.add_access(block, this->get_or_create_container(builder, memlet_op.getInput()))
                                }
                            );
                        }
                        return success();
                    })
                    .Case<ConstantOp>([&](ConstantOp constant_op) {
                        TypedAttr attr = constant_op.getValue();
                        std::string constant = this->convertTypedAttr(attr);
                        if (constant.empty()) {
                            constant_op->emitOpError("could not convert constant value");
                            return failure();
                        }
                        constant_nodes.insert(
                            {constant_op.getResult().getImpl(),
                             &builder.add_constant(block, constant, *this->convertType(attr.getType()))}
                        );
                        return success();
                    })
                    .Case<YieldOp>([&](YieldOp yield_op) {
                        unsigned int num_operands = op.getNumOperands();
                        assert(num_operands == results.size());
                        for (unsigned int i = 0; i < num_operands; i++) {
                            this->value_map_
                                .insert(results.at(i), this->get_or_create_container(builder, yield_op.getOperand(i)));
                        }
                        return success();
                    })
                    .Default([&](Operation* op) {
                        return op->emitError("Unknown data flow operation. Could not translate.");
                    });
            if (failed(result)) {
                return failure();
            }
        }

        // Create all edges
        for (auto& op : block_op->getRegion().getOps()) {
            LogicalResult result =
                llvm::TypeSwitch<Operation*, LogicalResult>(&op)
                    .Case<TaskletOp>([&](TaskletOp tasklet_op) {
                        std::vector<std::string> inputs;
                        switch (arity(tasklet_op.getCode())) {
                            case 1:
                                inputs.push_back("_in");
                                break;
                            case 2:
                                inputs.push_back("_in1");
                                inputs.push_back("_in2");
                                break;
                            case 3:
                                inputs.push_back("_in1");
                                inputs.push_back("_in2");
                                inputs.push_back("_in3");
                                break;
                            default:
                                return failure();
                        }
                        for (unsigned int i = 0, e = tasklet_op.getNumOperands(); i < e; i++) {
                            if (auto memlet_op = dyn_cast_or_null<MemletOp>(tasklet_op.getOperand(i).getDefiningOp())) {
                                builder.add_computational_memlet(
                                    block,
                                    *access_nodes.at(memlet_op.getInput().getImpl()),
                                    *tasklets.at(tasklet_op.getLoc()->getImpl()),
                                    inputs.at(i),
                                    {}
                                );
                            } else if (auto constant_op =
                                           dyn_cast_or_null<ConstantOp>(tasklet_op.getOperand(i).getDefiningOp())) {
                                builder.add_computational_memlet(
                                    block,
                                    *constant_nodes.at(constant_op.getResult().getImpl()),
                                    *tasklets.at(tasklet_op.getLoc()->getImpl()),
                                    inputs.at(i),
                                    {}
                                );
                            } else {
                                return failure();
                            }
                        }
                        return success();
                    })
                    .Case<FillOp>([&](FillOp fill_op) {
                        if (auto memlet_op = dyn_cast_or_null<MemletOp>(fill_op.getInput().getDefiningOp())) {
                            builder.add_computational_memlet(
                                block,
                                *access_nodes.at(memlet_op.getInput().getImpl()),
                                *libnodes.at(fill_op.getLoc()->getImpl()),
                                "X",
                                {},
                                *this->convertType(fill_op.getInput().getType())
                            );
                        } else if (auto constant_op = dyn_cast_or_null<ConstantOp>(fill_op.getInput().getDefiningOp()
                                   )) {
                            builder.add_computational_memlet(
                                block,
                                *constant_nodes.at(constant_op.getResult().getImpl()),
                                *libnodes.at(fill_op.getLoc()->getImpl()),
                                "X",
                                {},
                                *this->convertType(fill_op.getInput().getType())
                            );
                        } else {
                            return failure();
                        }
                        return success();
                    })
                    .Case<MatmulOp>([&](MatmulOp matmul_op) {
                        if (auto memlet_op = dyn_cast_or_null<MemletOp>(matmul_op.getLhs().getDefiningOp())) {
                            builder.add_computational_memlet(
                                block,
                                *access_nodes.at(memlet_op.getInput().getImpl()),
                                *libnodes.at(matmul_op.getLoc()->getImpl()),
                                "__A",
                                {},
                                *this->convertType(matmul_op.getLhs().getType())
                            );
                        } else if (auto constant_op = dyn_cast_or_null<ConstantOp>(matmul_op.getLhs().getDefiningOp()
                                   )) {
                            builder.add_computational_memlet(
                                block,
                                *constant_nodes.at(constant_op.getResult().getImpl()),
                                *libnodes.at(matmul_op.getLoc()->getImpl()),
                                "__A",
                                {},
                                *this->convertType(matmul_op.getLhs().getType())
                            );
                        } else {
                            return failure();
                        }
                        if (auto memlet_op = dyn_cast_or_null<MemletOp>(matmul_op.getRhs().getDefiningOp())) {
                            builder.add_computational_memlet(
                                block,
                                *access_nodes.at(memlet_op.getInput().getImpl()),
                                *libnodes.at(matmul_op.getLoc()->getImpl()),
                                "__B",
                                {},
                                *this->convertType(matmul_op.getRhs().getType())
                            );
                        } else if (auto constant_op = dyn_cast_or_null<ConstantOp>(matmul_op.getRhs().getDefiningOp()
                                   )) {
                            builder.add_computational_memlet(
                                block,
                                *constant_nodes.at(constant_op.getResult().getImpl()),
                                *libnodes.at(matmul_op.getLoc()->getImpl()),
                                "__B",
                                {},
                                *this->convertType(matmul_op.getRhs().getType())
                            );
                        } else {
                            return failure();
                        }
                        if (auto memlet_op = dyn_cast_or_null<MemletOp>(matmul_op.getResInput().getDefiningOp())) {
                            builder.add_computational_memlet(
                                block,
                                *access_nodes.at(memlet_op.getInput().getImpl()),
                                *libnodes.at(matmul_op.getLoc()->getImpl()),
                                "__C",
                                {},
                                *this->convertType(matmul_op.getResInput().getType())
                            );
                        } else if (auto constant_op =
                                       dyn_cast_or_null<ConstantOp>(matmul_op.getResInput().getDefiningOp())) {
                            builder.add_computational_memlet(
                                block,
                                *constant_nodes.at(constant_op.getResult().getImpl()),
                                *libnodes.at(matmul_op.getLoc()->getImpl()),
                                "__C",
                                {},
                                *this->convertType(matmul_op.getResInput().getType())
                            );
                        } else {
                            return failure();
                        }
                        return success();
                    })
                    .Case<MemletOp>([&](MemletOp memlet_op) {
                        if (auto tasklet_op = dyn_cast_or_null<TaskletOp>(memlet_op.getInput().getDefiningOp())) {
                            builder.add_computational_memlet(
                                block,
                                *tasklets.at(tasklet_op.getLoc()->getImpl()),
                                "_out",
                                *access_nodes.at(memlet_op.getOutput().getImpl()),
                                {}
                            );
                        } else if (auto fill_op = dyn_cast_or_null<FillOp>(memlet_op.getInput().getDefiningOp())) {
                            builder.add_computational_memlet(
                                block,
                                *libnodes.at(fill_op.getLoc()->getImpl()),
                                "Y",
                                *access_nodes.at(memlet_op.getOutput().getImpl()),
                                {},
                                *this->convertType(memlet_op.getOutput().getType())
                            );
                        } else if (auto matmul_op = dyn_cast_or_null<MatmulOp>(memlet_op.getInput().getDefiningOp())) {
                            builder.add_computational_memlet(
                                block,
                                *libnodes.at(matmul_op.getLoc()->getImpl()),
                                "__C",
                                *access_nodes.at(memlet_op.getOutput().getImpl()),
                                {},
                                *this->convertType(memlet_op.getOutput().getType())
                            );
                        }
                        return success();
                    })
                    .Case<ConstantOp>([&](ConstantOp constant_op) {
                        // Has per definition no input
                        return success();
                    })
                    .Case<YieldOp>([&](YieldOp yield_op) { return success(); })
                    .Default([&](Operation* op) {
                        return op->emitError("Unknown data flow operation. Could not translate.");
                    });
            if (failed(result)) {
                return failure();
            }
        }

        return success();
    }

    LogicalResult translateReturnOp(Builder& builder, Sequence& parent, ReturnOp* return_op) {
        builder.add_return(parent, this->get_or_create_container(builder, return_op->getOperand()));
        return success();
    }

    template<typename T>
    std::string convertResourceElementsToString(DenseResourceElementsAttr attr) {
        if (::mlir::detail::DenseResourceElementsAttrBase<T> typed_attr =
                dyn_cast_or_null<::mlir::detail::DenseResourceElementsAttrBase<T>>(attr)) {
            std::optional<ArrayRef<T>> array_ref = typed_attr.tryGetAsArrayRef();
            if (!array_ref) {
                return "";
            }
            std::stringstream stream;
            stream << "{";
            bool first = true;
            for (T val : *array_ref) {
                if (first) {
                    first = false;
                } else {
                    stream << ", ";
                }
                stream << std::to_string(val);
            }
            stream << "}";
            return stream.str();
        }
        return "";
    }

    LogicalResult translateTensorConstantOp(Builder& builder, Sequence& parent, TensorConstantOp* tensor_constant_op) {
        std::string constant;
        if (auto dense_resource_attr = dyn_cast<DenseResourceElementsAttr>(tensor_constant_op->getValues())) {
            Type element_type = dense_resource_attr.getElementType();
            if (element_type.isInteger(1)) {
                constant = this->convertResourceElementsToString<bool>(dense_resource_attr);
            } else if (element_type.isSignedInteger(8) || element_type.isSignlessInteger(8)) {
                constant = this->convertResourceElementsToString<int8_t>(dense_resource_attr);
            } else if (element_type.isSignedInteger(16) || element_type.isSignlessInteger(16)) {
                constant = this->convertResourceElementsToString<int16_t>(dense_resource_attr);
            } else if (element_type.isSignedInteger(32) || element_type.isSignlessInteger(32)) {
                constant = this->convertResourceElementsToString<int32_t>(dense_resource_attr);
            } else if (element_type.isSignedInteger(64) || element_type.isSignlessInteger(64)) {
                constant = this->convertResourceElementsToString<int64_t>(dense_resource_attr);
            } else if (element_type.isUnsignedInteger(8)) {
                constant = this->convertResourceElementsToString<uint8_t>(dense_resource_attr);
            } else if (element_type.isUnsignedInteger(16)) {
                constant = this->convertResourceElementsToString<uint16_t>(dense_resource_attr);
            } else if (element_type.isUnsignedInteger(32)) {
                constant = this->convertResourceElementsToString<uint32_t>(dense_resource_attr);
            } else if (element_type.isUnsignedInteger(64)) {
                constant = this->convertResourceElementsToString<uint64_t>(dense_resource_attr);
            } else if (element_type.isF32()) {
                constant = this->convertResourceElementsToString<float>(dense_resource_attr);
            } else if (element_type.isF64()) {
                constant = this->convertResourceElementsToString<double>(dense_resource_attr);
            }
        }
        if (constant.empty()) {
            return tensor_constant_op->emitOpError("could not convert constant. dense type required.");
        }
        Value result = tensor_constant_op->getResult();
        auto result_type = this->convertType(result.getType());

        auto& block = builder.add_block(parent);
        auto& constant_access = builder.add_constant(block, constant, *result_type);
        auto& result_access = builder.add_access(block, this->get_or_create_container(builder, result));
        auto& tasklet = builder.add_tasklet(block, ::sdfg::data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_memlet(block, constant_access, "void", tasklet, "_in", {}, *result_type, ::sdfg::DebugInfo());
        builder.add_memlet(block, tasklet, "_out", result_access, "void", {}, *result_type, ::sdfg::DebugInfo());

        return success();
    }

    LogicalResult emitCode(raw_ostream& os) {
        for (auto& builder : this->builders_) {
            ::sdfg::serializer::JSONSerializer serializer;
            auto json = serializer.serialize(builder.subject());
            os << json.dump(4) << "\n";
        }
        return success();
    }
};

LogicalResult translateToSDFG(Operation* op, raw_ostream& os) {
    SDFGTranslator translator;
    if (failed(translator.translate(op))) {
        return emitError(op->getLoc(), "Could not translate to SDFG");
    }

    if (failed(translator.emitCode(os))) {
        return emitError(op->getLoc(), "Could not generate code");
    }

    return success();
}

void registerToSDFGTranslation() {
    TranslateFromMLIRRegistration registration(
        "mlir-to-sdfg",
        "translate MLIR in SDFG dialect to SDFG",
        [](Operation* op, raw_ostream& os) { return translateToSDFG(op, os); },
        [](mlir::DialectRegistry& registry) { registry.insert<mlir::sdfg::SDFGDialect>(); }
    );
}

} // namespace sdfg
} // namespace mlir
