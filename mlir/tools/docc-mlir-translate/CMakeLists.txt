set(LLVM_LINK_COMPONENTS Support)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

set(LIBS
    ${dialect_libs}
    ${translation_libs}
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRTranslateLib
    MLIRSupport
)

set_source_files_properties(docc-mlir-translate.cpp PROPERTIES COMPILE_FLAGS -fexceptions)
add_llvm_executable(docc-mlir-translate
    docc-mlir-translate.cpp
    DEPENDS ${LIBS}
)

llvm_update_compile_flags(docc-mlir-translate)
target_link_libraries(docc-mlir-translate PRIVATE ${LIBS})

mlir_check_all_link_libraries(docc-mlir-translate)