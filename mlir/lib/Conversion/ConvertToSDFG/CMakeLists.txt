add_mlir_conversion_library(MLIRConvertToSDFG
    ConvertToSDFG.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/mlir/Conversion/ConvertToSDFG

    DEPENDS
    SDFGConversionPassIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRArithDialect
    MLIRFuncDialect
    MLIRSDFGDialect
    MLIRTensorDialect
)