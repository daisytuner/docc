set(TARGET_SDFG_FILES
    BuiltinToSDFGTranslator.cpp
    FuncToSDFGTranslator.cpp
    SDFGTranslator.cpp
    TranslateToSDFG.cpp
)

set_source_files_properties(${TARGET_SDFG_FILES} PROPERTIES COMPILE_FLAGS -fexceptions)
add_mlir_translation_library(MLIRTargetSDFG
    ${TARGET_SDFG_FILES}

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/mlir/Target/SDFG

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport
    MLIRTranslateLib
    MLIRFuncDialect
    MLIRArithDialect
    MLIRLinalgDialect
    MLIRTensorDialect
    sdfglib
)