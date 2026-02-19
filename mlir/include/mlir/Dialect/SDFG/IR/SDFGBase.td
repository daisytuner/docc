#ifndef SDFG_BASE
#define SDFG_BASE

include "mlir/IR/DialectBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/CommonTypeConstraints.td"

def SDFG_Dialect : Dialect {
    let name = "sdfg";
    let cppNamespace = "::mlir::sdfg";

    let summary = "A SDFG MLIR dialect";
    let description = [{
        MLIR dialect for generating stateful dataflow multigraphs (SDFGs).
    }];
    let dependentDialects = [
        "arith::ArithDialect"
    ];

    // let useDefaultTypePrinterParser = 1;
}

/**
 * Enumeration of all primitive data types supported by sdfglib
 */
def SDFG_PrimitiveType : AnyTypeOf<[
        // Void, ///< Void type (0 bits)
        I1, ///< Boolean type (1 bit)
        I8, ///< 8-bit signed integer
        I16, ///< 16-bit signed integer
        I32, ///< 32-bit signed integer
        I64, ///< 64-bit signed integer
        I128, ///< 128-bit signed integer
        UI8, ///< 8-bit unsigned integer
        UI16, ///< 16-bit unsigned integer
        UI32, ///< 32-bit unsigned integer
        UI64, ///< 64-bit unsigned integer
        UI<128>, ///< 128-bit unsigned integer
        F16, ///< 16-bit floating-point (IEEE 754 half precision)
        BF16, ///< 16-bit brain floating-point
        F32, ///< 32-bit floating-point (IEEE 754 single precision)
        F64, ///< 64-bit floating-point (IEEE 754 double precision)
        F80, ///< 80-bit extended precision floating-point (x86)
        F128, ///< 128-bit floating-point (IEEE 754 quadruple precision)
        // PPC_FP128 ///< 128-bit floating-point (PowerPC double-double)
    ], "SDFG primitive type">;

def SDFG_Type : AnyTypeOf<[
        SDFG_PrimitiveType,
        VectorOf<SDFG_PrimitiveType.allowedTypes>,
        TensorOf<SDFG_PrimitiveType.allowedTypes>
    ], "SDFG type">;

/**
 * Operation codes for tasklet computations
 *
 * Defines all primitive operations that tasklets can perform, including
 * arithmetic, logical, comparison, and type conversion operations.
 */
def SDFG_TaskletCodeAttr : I64EnumAttr<
        "TaskletCode", "Operation codes for tasklet computations",
        [
            I64EnumAttrCase<"assign", 0>, ///< Assignment/copy operation (can be cast if types differ)
            // Floating-point-specific operations
            // Operations
            I64EnumAttrCase<"fp_neg", 1>, ///< Floating-point negation
            I64EnumAttrCase<"fp_add", 2>, ///< Floating-point addition
            I64EnumAttrCase<"fp_sub", 3>, ///< Floating-point subtraction
            I64EnumAttrCase<"fp_mul", 4>, ///< Floating-point multiplication
            I64EnumAttrCase<"fp_div", 5>, ///< Floating-point division
            I64EnumAttrCase<"fp_rem", 6>, ///< Floating-point remainder
            I64EnumAttrCase<"fp_fma", 7>, ///< Floating-point fused multiply-add
            // Comparisions
            I64EnumAttrCase<"fp_oeq", 8>, ///< Floating-point ordered equal
            I64EnumAttrCase<"fp_one", 9>, ///< Floating-point ordered not equal
            I64EnumAttrCase<"fp_oge", 10>, ///< Floating-point ordered greater or equal
            I64EnumAttrCase<"fp_ogt", 11>, ///< Floating-point ordered greater than
            I64EnumAttrCase<"fp_ole", 12>, ///< Floating-point ordered less or equal
            I64EnumAttrCase<"fp_olt", 13>, ///< Floating-point ordered less than
            I64EnumAttrCase<"fp_ord", 14>, ///< Floating-point ordered (neither operand is NaN)
            I64EnumAttrCase<"fp_ueq", 15>, ///< Floating-point unordered equal
            I64EnumAttrCase<"fp_une", 16>, ///< Floating-point unordered not equal
            I64EnumAttrCase<"fp_ugt", 17>, ///< Floating-point unordered greater than
            I64EnumAttrCase<"fp_uge", 18>, ///< Floating-point unordered greater or equal
            I64EnumAttrCase<"fp_ult", 19>, ///< Floating-point unordered less than
            I64EnumAttrCase<"fp_ule", 20>, ///< Floating-point unordered less or equal
            I64EnumAttrCase<"fp_uno", 21>, ///< Floating-point unordered (at least one operand is NaN)
            // Integer-specific operations
            // Operations
            I64EnumAttrCase<"int_add", 22>, ///< Integer addition
            I64EnumAttrCase<"int_sub", 23>, ///< Integer subtraction
            I64EnumAttrCase<"int_mul", 24>, ///< Integer multiplication
            I64EnumAttrCase<"int_sdiv", 25>, ///< Signed integer division
            I64EnumAttrCase<"int_srem", 26>, ///< Signed integer remainder
            I64EnumAttrCase<"int_udiv", 27>, ///< Unsigned integer division
            I64EnumAttrCase<"int_urem", 28>, ///< Unsigned integer remainder
            I64EnumAttrCase<"int_and", 29>, ///< Bitwise AND
            I64EnumAttrCase<"int_or", 30>, ///< Bitwise OR
            I64EnumAttrCase<"int_xor", 31>, ///< Bitwise XOR
            I64EnumAttrCase<"int_shl", 32>, ///< Shift left
            I64EnumAttrCase<"int_ashr", 33>, ///< Arithmetic shift right (sign-extend)
            I64EnumAttrCase<"int_lshr", 34>, ///< Logical shift right (zero-extend)
            I64EnumAttrCase<"int_smin", 35>, ///< Signed minimum
            I64EnumAttrCase<"int_smax", 36>, ///< Signed maximum
            I64EnumAttrCase<"int_scmp", 37>, ///< Signed comparison (-1, 0, 1)
            I64EnumAttrCase<"int_umin", 38>, ///< Unsigned minimum
            I64EnumAttrCase<"int_umax", 39>, ///< Unsigned maximum
            I64EnumAttrCase<"int_ucmp", 40>, ///< Unsigned comparison (-1, 0, 1)
            // Comparisions
            I64EnumAttrCase<"int_eq", 41>, ///< Integer equal
            I64EnumAttrCase<"int_ne", 42>, ///< Integer not equal
            I64EnumAttrCase<"int_sge", 43>, ///< Signed greater or equal
            I64EnumAttrCase<"int_sgt", 44>, ///< Signed greater than
            I64EnumAttrCase<"int_sle", 45>, ///< Signed less or equal
            I64EnumAttrCase<"int_slt", 46>, ///< Signed less than
            I64EnumAttrCase<"int_uge", 47>, ///< Unsigned greater or equal
            I64EnumAttrCase<"int_ugt", 48>, ///< Unsigned greater than
            I64EnumAttrCase<"int_ule", 49>, ///< Unsigned less or equal
            I64EnumAttrCase<"int_ult", 50>, ///< Unsigned less than
            I64EnumAttrCase<"int_abs", 51> ///< Integer absolute value
        ]> {
    let cppNamespace = "::mlir::sdfg";
}

#endif // SDFG_BASE