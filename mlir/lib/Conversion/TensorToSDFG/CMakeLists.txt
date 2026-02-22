add_mlir_conversion_library(MLIRTensorToSDFG
    TensorToSDFG.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/mlir/Conversion/TensorToSDFG

    DEPENDS
    SDFGConversionPassIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRTensorDialect
    MLIRSDFGDialect
)