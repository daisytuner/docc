get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

set(LIBS
    ${dialect_libs}
    MLIROptLib
    MLIRPass
    MLIRSDFGDialect
    MLIRLspServerLib
)

add_llvm_executable(docc-mlir-lsp-server
    docc-mlir-lsp-server.cpp
    DEPENDS ${LIBS}
)

llvm_update_compile_flags(docc-mlir-lsp-server)
target_link_libraries(docc-mlir-lsp-server PRIVATE ${LIBS})

mlir_check_all_link_libraries(docc-mlir-lsp-server)