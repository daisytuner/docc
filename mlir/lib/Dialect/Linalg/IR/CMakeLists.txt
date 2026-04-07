add_mlir_dialect_library(MLIRLinalgCustomDialect
    LinalgCustomOps.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/mlir/Dialect/Linalg

    DEPENDS
    MLIRLinalgCustomOpsIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRLinalgDialect
)