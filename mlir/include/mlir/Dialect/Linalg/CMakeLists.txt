add_subdirectory(IR)

set(LLVM_TARGET_DEFINITIONS CustomPasses.td)
mlir_tablegen(CustomPasses.h.inc -gen-pass-decls -name LinalgCustom)
add_public_tablegen_target(MLIRLinalgCustomPassIncGen)

add_mlir_doc(CustomPasses LinalgCustomPasses ./ -gen-pass-doc)