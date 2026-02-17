get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
    ${dialect_libs}
    ${conversion_libs}
    MLIROptLib
    MLIRPass
    MLIRArithDialect
    MLIRSDFGDialect
)

add_llvm_executable(docc-mlir-opt
    docc-mlir-opt.cpp
    DEPENDS ${LIBS}
)

llvm_update_compile_flags(docc-mlir-opt)
target_link_libraries(docc-mlir-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(docc-mlir-opt)