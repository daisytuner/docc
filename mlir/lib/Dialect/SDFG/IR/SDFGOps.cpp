#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/LogicalResult.h>

#include "mlir/Dialect/SDFG/IR/SDFG.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LLVM.h"

// SDFG Dialect Operations
#define GET_OP_CLASSES
#include "mlir/Dialect/SDFG/IR/SDFGOps.cpp.inc"

// SDFG Dialect Enum Attributes
#include "mlir/Dialect/SDFG/IR/SDFGOpsEnums.cpp.inc"

namespace mlir {
namespace sdfg {

LogicalResult argument_depends_on_memlet(Value val) {
    return llvm::TypeSwitch<Operation*, LogicalResult>(val.getDefiningOp())
        .Case<MemletOp>([](MemletOp memlet_op) { return success(); })
        .Case<ConstantOp>([](ConstantOp constant_op) { return success(); })
        .Default([](Operation* op) { return failure(); });
}

//===----------------------------------------------------------------------===//
// SDFGOp
//===----------------------------------------------------------------------===//

void SDFGOp::print(OpAsmPrinter& p) {
    bool is_external = this->getBody().empty();
    p << " ";

    // Print name as symbol
    p.printSymbolName(this->getSymName());

    // Print function signature: Arguments
    p << " (";
    ArrayRef<Type> arg_types = this->getFunctionType().getInputs();
    for (size_t i = 0, e = arg_types.size(); i < e; ++i) {
        if (i > 0) {
            p << ", ";
        }
        if (is_external) {
            p.printType(arg_types[i]);
        } else {
            p.printRegionArgument(this->getBody().getArgument(i));
        }
    }
    p << ")";

    // Print function signature: Result type
    ArrayRef<Type> result_types = this->getFunctionType().getResults();
    if (!result_types.empty()) {
        assert(result_types.size() == 1);
        p << " -> ";
        p.printType(result_types.front());
    }

    // Print optional function body
    if (!is_external) {
        p << " ";
        p.printRegion(this->getBody(), false, true, true);
    }
}

ParseResult SDFGOp::parse(OpAsmParser& parser, OperationState& result) {
    auto& builder = parser.getBuilder();

    // Parse name as symbol
    StringAttr nameAttr;
    if (parser.parseSymbolName(nameAttr, "sym_name", result.attributes)) {
        return failure();
    }

    // Parse function signature: Arguments
    SMLoc signature_loc = parser.getCurrentLocation();
    llvm::SmallVector<OpAsmParser::Argument> entry_args;
    auto parseArgElem = [&]() -> ParseResult {
        OpAsmParser::Argument argument;
        auto arg_present = parser.parseOptionalArgument(argument, true, false);
        if (arg_present.has_value()) {
            if (failed(arg_present.value())) {
                return failure(); // Present but malformed.
            }

            // Reject this if the preceding argument was missing a name.
            if (!entry_args.empty() && entry_args.back().ssaName.name.empty()) {
                return parser.emitError(argument.ssaName.location, "expected type instead of SSA identifier");
            }
        } else {
            argument.ssaName.location = parser.getCurrentLocation();
            // Oterwise we just have a type list without SSA names. Reject this if the preceding argument had a name.
            if (!entry_args.empty() && !entry_args.back().ssaName.name.empty()) {
                return parser.emitError(argument.ssaName.location, "expected SSA identifier");
            }

            if (parser.parseType(argument.type) || parser.parseOptionalLocationSpecifier(argument.sourceLoc)) {
                return failure();
            }
        }
        entry_args.push_back(argument);
        return success();
    };
    if (parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren, parseArgElem)) {
        return failure();
    }

    // Parse function signature: Result type
    bool has_result_type = succeeded(parser.parseOptionalArrow());
    SmallVector<Type> result_types;
    if (has_result_type) {
        Type result_type;
        if (parser.parseType(result_type)) {
            return failure();
        }
        result_types.push_back(result_type);
    }

    // Add function signature
    SmallVector<Type> arg_types;
    arg_types.reserve(entry_args.size());
    for (auto& arg : entry_args) {
        arg_types.push_back(arg.type);
    }
    Type function_type = builder.getFunctionType(arg_types, result_types);
    if (!function_type) {
        return parser.emitError(signature_loc, "failed to construct function type");
    }
    result.addAttribute("function_type", TypeAttr::get(function_type));

    // Parse optional function body
    auto* body = result.addRegion();
    SMLoc body_loc = parser.getCurrentLocation();
    OptionalParseResult parse_result = parser.parseOptionalRegion(*body, entry_args, false);
    if (parse_result.has_value()) {
        if (failed(*parse_result)) {
            return failure();
        }
        // Function body was parsed, make sure ist not empty.
        if (body->empty()) {
            return parser.emitError(body_loc, "expected non-empty SDFG body");
        }
    }

    return success();
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

LogicalResult ReturnOp::verify() {
    auto sdfg_op = cast<SDFGOp>(this->getParentOp());
    const auto& results = sdfg_op.getFunctionType().getResults();
    auto operand = this->getOperand();
    if (operand) {
        if (results.size() == 0) {
            return this->emitOpError("has operand, but enclosing SDFG (@")
                   << sdfg_op.getSymName() << ") has no result type";
        }
        assert(results.size() == 1);
        if (results.front() != operand.getType()) {
            return this->emitError() << "type of return operand (" << operand.getType()
                                     << ") doesn't match SDFG result type (" << results.front() << ") in SDFG @"
                                     << sdfg_op.getSymName();
        }
    } else {
        if (results.size() == 0) {
            return success();
        } else if (results.size() == 1) {
            return this->emitOpError("has no operand, but enclosing SDFG (@")
                   << sdfg_op.getSymName() << ") has a result type";
        }
    }
    return success();
}

//===----------------------------------------------------------------------===//
// BlockOp
//===----------------------------------------------------------------------===//

void BlockOp::build(OpBuilder& builder, OperationState& state, TypeRange results) {
    Region* body = state.addRegion();
    state.addTypes(results);
    OpBuilder::InsertionGuard g(builder);
    BlockOp::ensureTerminator(*body, builder, state.location, results.size());
}

void BlockOp::build(
    OpBuilder& builder,
    OperationState& state,
    TypeRange resultTypes,
    ValueRange operands,
    ::llvm::ArrayRef<NamedAttribute> attributes
) {
    assert(operands.size() == 0u && "mismatched number of parameters");
    state.addOperands(operands);
    state.addAttributes(attributes);
    Region* body = state.addRegion();
    state.addTypes(resultTypes);
    OpBuilder::InsertionGuard g(builder);
    BlockOp::ensureTerminator(*body, builder, state.location, resultTypes.size());
}

void BlockOp::print(OpAsmPrinter& p) {
    // Print optional result types
    auto results = this->getResults();
    size_t results_size = results.size();
    if (results_size > 0) {
        p << " -> ";
        if (results_size > 1) {
            p << "(";
        }
        for (size_t i = 0; i < results_size; i++) {
            if (i > 0) {
                p << ", ";
            }
            p.printType(results[i].getType());
        }
        if (results_size > 1) {
            p << ")";
        }
    }

    // Print optional function body
    if (!this->getBody().empty()) {
        p << " ";
        p.printRegion(this->getBody(), false, true, true);
    }
}

ParseResult BlockOp::parse(OpAsmParser& parser, OperationState& result) {
    // Parse optional result types
    if (parser.parseOptionalArrowTypeList(result.types)) {
        return failure();
    }

    // Parse optional body
    auto* body = result.addRegion();
    SMLoc body_loc = parser.getCurrentLocation();
    OptionalParseResult parse_result = parser.parseOptionalRegion(*body);
    if (parse_result.has_value()) {
        if (failed(*parse_result)) {
            return failure();
        }
        // Function body was parsed, make sure ist not empty.
        if (body->empty()) {
            return parser.emitError(body_loc, "expected non-empty block body");
        }

        BlockOp::ensureTerminator(*body, parser.getBuilder(), result.location, result.types.size());
    }

    return success();
}

void BlockOp::ensureTerminator(Region& region, OpBuilder& builder, Location loc, size_t num_results) {
    OpBuilder::InsertionGuard guard(builder);
    if (region.empty()) {
        builder.createBlock(&region);
    }

    Block& block = region.back();
    if (!block.empty() && block.back().hasTrait<OpTrait::IsTerminator>()) {
        return;
    }

    builder.setInsertionPointToEnd(&block);
    OperationState state(loc, YieldOp::getOperationName());
    if (num_results == 0) {
        YieldOp::build(builder, state);
    } else {
        SmallVector<Value> all_operands;
        for (auto& op : block.getOperations()) {
            for (unsigned int i = 0, e = op.getNumResults(); i < e; i++) {
                all_operands.push_back(op.getResult(i));
            }
        }
        size_t num_all_operands = all_operands.size();
        if (num_all_operands <= num_results) {
            YieldOp::build(builder, state, all_operands);
        } else {
            SmallVector<Value> operands;
            operands.reserve(num_results);
            for (size_t i = num_all_operands - num_results; i < num_all_operands; i++) {
                operands.push_back(all_operands[i]);
            }
            YieldOp::build(builder, state, operands);
        }
    }
    builder.insert(Operation::create(state));
}

void BlockOp::ensureTerminator(Region& region, Builder& builder, Location loc, size_t num_results) {
    OpBuilder op_builder(builder.getContext());
    BlockOp::ensureTerminator(region, op_builder, loc, num_results);
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

LogicalResult YieldOp::verify() {
    auto block_op = cast<BlockOp>(this->getParentOp());
    const auto& results = block_op->getResults();
    size_t num_operands = this->getNumOperands();
    if (num_operands != results.size()) {
        return emitOpError("has ") << num_operands << " operands, but enclosing block yields " << results.size();
    } else {
        for (size_t i = 0; i < num_operands; i++) {
            if (this->getOperand(i).getType() != results[i].getType()) {
                return emitError() << "type of yield operand " << i << " (" << this->getOperand(i).getType()
                                   << ") doesn't match enclosing block result type (" << results[i].getType() << ")";
            }
        }
    }
    return success();
}

//===----------------------------------------------------------------------===//
// TaskletOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult TaskletOp::verify() {
    size_t code_arity = arity(this->getCode());
    size_t num_operands = this->getNumOperands();
    if (code_arity != num_operands) {
        return this->emitOpError() << "expects " << code_arity << " operands, but got " << num_operands;
    }
    for (auto op : this->getOperands()) {
        if (failed(argument_depends_on_memlet(op))) {
            return this->emitOpError() << "operand " << op << " does not depend on memlet or constant";
        }
    }
    return success();
}

//===----------------------------------------------------------------------===//
// FillOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult FillOp::verify() {
    // if (this->getInput().getType() != this->getOutput().getType()) {
    //     return this->emitOpError() << "input type (" << this->getInput().getType() << ") is not the same as output
    //     type ("
    //                              << this->getOutput().getType() << ")";
    // }
    if (failed(argument_depends_on_memlet(this->getInput()))) {
        return this->emitOpError() << "does not depend on memlet or constant";
    }
    return success();
}

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult MatmulOp::verify() {
    // TODO check type
    if (failed(argument_depends_on_memlet(this->getResInput()))) {
        return this->emitOpError() << "result input does not depend on memlet or constant";
    }
    if (failed(argument_depends_on_memlet(this->getLhs()))) {
        return this->emitOpError() << "lhs does not depend on memlet or constant";
    }
    if (failed(argument_depends_on_memlet(this->getRhs()))) {
        return this->emitOpError() << "rhs does not depend on memlet or constant";
    }
    return success();
}

//===----------------------------------------------------------------------===//
// ReshapeMemletOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult ReshapeMemletOp::verify() {
    auto input_type = cast<TensorType>(this->getInput().getType());
    auto output_type = cast<TensorType>(this->getOutput().getType());

    // Verify element types match
    if (input_type.getElementType() != output_type.getElementType()) {
        return this->emitOpError() << "input element type (" << input_type.getElementType()
                                   << ") must match output element type (" << output_type.getElementType() << ")";
    }

    // Verify total number of elements is preserved
    if (input_type.hasStaticShape() && output_type.hasStaticShape()) {
        int64_t input_num_elements = input_type.getNumElements();
        int64_t output_num_elements = output_type.getNumElements();

        if (input_num_elements != output_num_elements) {
            return this->emitOpError() << "reshape cannot change the total number of elements: input has "
                                       << input_num_elements << " elements but output has " << output_num_elements;
        }
    }

    return success();
}

//===----------------------------------------------------------------------===//
// FlipMemletOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult FlipMemletOp::verify() {
    auto input_type = cast<TensorType>(this->getInput().getType());
    auto output_type = cast<TensorType>(this->getOutput().getType());

    // Verify input and output types match (flip doesn't change shape or element type)
    if (input_type != output_type) {
        return this->emitOpError() << "input type (" << input_type << ") must match output type (" << output_type
                                   << ")";
    }

    // Verify axes are valid
    if (input_type.hasRank()) {
        int64_t rank = input_type.getRank();
        for (int64_t axis : this->getAxes()) {
            if (axis < 0 || axis >= rank) {
                return this->emitOpError() << "axis " << axis << " is out of bounds for tensor with rank " << rank;
            }
        }
    }

    return success();
}

//===----------------------------------------------------------------------===//
// TransposeMemletOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult TransposeMemletOp::verify() {
    auto input_type = cast<TensorType>(this->getInput().getType());
    auto output_type = cast<TensorType>(this->getOutput().getType());

    // Verify element types match
    if (input_type.getElementType() != output_type.getElementType()) {
        return this->emitOpError() << "input element type (" << input_type.getElementType()
                                   << ") must match output element type (" << output_type.getElementType() << ")";
    }

    ArrayRef<int64_t> permutation = this->getPermutation();

    // Verify permutation length matches input rank
    if (input_type.hasRank()) {
        int64_t rank = input_type.getRank();
        if (static_cast<int64_t>(permutation.size()) != rank) {
            return this->emitOpError() << "permutation length (" << permutation.size()
                                       << ") must match input tensor rank (" << rank << ")";
        }

        // Verify permutation is a valid permutation (each index 0..rank-1 appears exactly once)
        SmallVector<bool> seen(rank, false);
        for (int64_t idx : permutation) {
            if (idx < 0 || idx >= rank) {
                return this->emitOpError() << "permutation index " << idx << " is out of bounds for rank " << rank;
            }
            if (seen[idx]) {
                return this->emitOpError() << "permutation index " << idx << " appears more than once";
            }
            seen[idx] = true;
        }

        // Verify output shape matches permuted input shape
        if (output_type.hasRank()) {
            ArrayRef<int64_t> input_shape = input_type.getShape();
            ArrayRef<int64_t> output_shape = output_type.getShape();
            for (int64_t i = 0; i < rank; ++i) {
                int64_t expected_dim = input_shape[permutation[i]];
                if (expected_dim != ShapedType::kDynamic && output_shape[i] != ShapedType::kDynamic &&
                    output_shape[i] != expected_dim) {
                    return this->emitOpError() << "output dimension " << i << " has size " << output_shape[i]
                                               << " but expected " << expected_dim << " based on permutation";
                }
            }
        }
    }

    return success();
}

} // namespace sdfg
} // namespace mlir
