set(LLVM_TARGET_DEFINITIONS LinalgCustomOps.td)
add_mlir_dialect(LinalgCustomOps linalg_custom)
add_mlir_doc(LinalgCustomOps LinalgCustomDialect Dialects/ -gen-dialect-doc)