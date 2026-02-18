#ifndef SDFG_OPS
#define SDFG_OPS

include "mlir/Dialect/SDFG/IR/SDFGBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/Traits.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

// Generell SDFG operation
class SDFG_Op<string mnemonic, list<Trait> traits = []> : Op<SDFG_Dialect, mnemonic, traits>;

// SDFG operation that represents a structured SDFG
def SDFG_SDFGOp : SDFG_Op<"sdfg", [NoTerminator, IsolatedFromAbove, HasParent<"ModuleOp">]> {
    let summary = "Strutured Stateful Data Flow Multigraph";
    let description = [{
        A dataflow-based graph representation for programs.
    }];

    let arguments = (ins SymbolNameAttr:$sym_name,
                         TypeAttrOf<FunctionType>:$function_type);
    let regions = (region AnyRegion:$body);

    let hasCustomAssemblyFormat = 1;
}

// SDFG operations that represents a return
def SDFG_ReturnOp : SDFG_Op<"return", [Pure, ReturnLike, Terminator, HasParent<"SDFGOp">]> {
    let summary = "Represents a return statement that exits the SDFG";
    let description = [{
        Can return an operand or nothing depending on the result type of the SDFG.
    }];

    let arguments = (ins Optional<AnyType>:$operand);

    let assemblyFormat = "attr-dict ($operand^ `:` type($operand))?";

    let hasVerifier = 1;
}

// SDFG operation that represents a tensor constant
def SDFG_TensorConstantOp : SDFG_Op<
    "tensor_constant",
    [Pure, ConstantLike, AllTypesMatch<["values", "result"]>, ParentOneOf<["SDFGOp"]>]
> {
    let summary = "Operation for tensor constants";

    let arguments = (ins ElementsAttrInterface:$values);
    let results = (outs TensorOf<SDFG_PrimitiveType.allowedTypes>:$result);

    let assemblyFormat = "attr-dict $values";
}

// SDFG operation that holds data flow nodes
def SDFG_BlockOp : SDFG_Op<"block", [SingleBlock, SingleBlockImplicitTerminator<"YieldOp">, ParentOneOf<["SDFGOp"]>]> {
    let summary = "A basic block with dataflow computations";
    let description = [{
        A Block is the fundamental computation unit in a StructuredSDFG. It represents a basic block that contains
        dataflow nodes describing the computations to be performed. These can be constants, memlets, tasklets, and
        library nodes.
    }];

    let results = (outs Variadic<AnyType>:$results);
    let regions = (region MaxSizedRegion<1>:$body);

    let builders = [
        OpBuilder<(ins "TypeRange":$results)>,
        OpBuilder<(ins "TypeRange":$resultTypes,
                       "ValueRange":$operands,
                       CArg<"::llvm::ArrayRef<NamedAttribute>", "{}">:$attributes)>
    ];

    let hasCustomAssemblyFormat = 1;
    let skipDefaultBuilders = 1;

    let extraClassDeclaration = [{
        static void ensureTerminator(Region& region, OpBuilder& builder, Location loc, size_t num_results);
        static void ensureTerminator(Region& region, Builder& builder, Location loc, size_t num_results);
    }];
}

// SDFG operation that yields a result
def SDFG_YieldOp : SDFG_Op<"yield", [Pure, ReturnLike, Terminator, ParentOneOf<["BlockOp"]>]> {
    let summary = "Block yield and termination operation";
    let description = [{
        Yields an SSA value from a block.
    }];

    let arguments = (ins Variadic<AnyType>:$operands);

    let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
    let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

    let hasVerifier = 1;
}

// SDFG data flow node that represents a constant literal value
def SDFG_ConstantOp : SDFG_Op<"constant", [AllTypesMatch<["value", "result"]>, HasParent<"BlockOp">]> {
    let summary = "Represents a constant literal value";

    let arguments = (ins TypedAttrInterface:$value);
    let results = (outs SDFG_PrimitiveType:$result);

    let assemblyFormat = "attr-dict $value";
}

// SDFG data flow node that represents a simple computation operation
def SDFG_TaskletOp : SDFG_Op<"tasklet", [HasParent<"BlockOp">]> {
    let summary = "Simple computational operations";
    let description = [{
        Tasklets represent primitive computational operations like arithmetic, logic, and type conversions. They are the
        fundamental building blocks for computation in the SDFG.
    }];

    let arguments = (ins SDFG_TaskletCodeAttr:$code, Variadic<SDFG_PrimitiveType>:$inputs);
    let results = (outs SDFG_PrimitiveType:$output);

    let assemblyFormat = "$code `,` $inputs attr-dict `:` `(` type($inputs) `)` `->` type($output)";

    let hasVerifier = 1;
}

// SDFG data flow node that represents a memlet
def SDFG_MemletOp : SDFG_Op<"memlet", [Pure, HasParent<"BlockOp">]> {
    let summary = "Edges representing data movement";

    let arguments = (ins AnyType:$input);
    let results = (outs AnyType:$output);

    let assemblyFormat = "attr-dict $input `:` type($input) `->` type($output)";
}

// LibraryNodes
class SDFG_UnaryLibraryNode<string mnemonic> : SDFG_Op<mnemonic, [Pure, HasParent<"BlockOp">]> {
    let arguments = (ins SDFG_Type:$input);
    let results = (outs SDFG_Type:$output);

    let assemblyFormat = "attr-dict $input `:` type($input) `->` type($output)";
}
class SDFG_BinaryLibraryNode<string mnemonic> : SDFG_Op<mnemonic, [Pure, HasParent<"BlockOp">]> {
    let arguments = (ins SDFG_Type:$lhs, SDFG_Type:$rhs);
    let results = (outs SDFG_Type:$output);

    let assemblyFormat = "attr-dict $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `->` type($output)";
}

// SDFG data flow node that represents tensor fill operation
def SDFG_FillOp : SDFG_UnaryLibraryNode<"fill"> {
    let summary = "Tensor fill operation";

    let hasVerifier = 1;
}

// SDFG data flow node that represents matrix-matrix multiplication
def SDFG_MatmulOp : SDFG_Op<"matmul", [Pure, HasParent<"BlockOp">]> {
    let summary = "Tensor matmul operation";

    let arguments = (ins SDFG_Type:$res_input, SDFG_Type:$lhs, SDFG_Type:$rhs);
    let results = (outs SDFG_Type:$output);

    let assemblyFormat = [{
        attr-dict $res_input `,` $lhs `,` $rhs `:` type($res_input) `,` type($lhs) `,` type($rhs) `->` type($output)
    }];

    let hasVerifier = 1;
}

// SDFG data flow node that represents a transpose
def SDFG_TransposeOp : SDFG_Op<"transpose", [Pure, HasParent<"BlockOp">]> {
    let summary = "Tensor transpose operation";

    let arguments = (ins SDFG_Type:$res_input, SDFG_Type:$input);
    let results = (outs SDFG_Type:$output);

    let assemblyFormat = [{
        attr-dict $res_input `,` $input `:` type($res_input) `,` type($input) `->` type($output)
    }];

    let hasVerifier = 1;
}

#endif // SDFG_OPS
