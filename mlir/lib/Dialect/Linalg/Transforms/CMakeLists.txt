add_mlir_dialect_library(MLIRLinalgCustomTransforms
    CustomSpecialize.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/mlir/Dialect/Linalg

    DEPENDS
    MLIRLinalgCustomDialect
    MLIRLinalgCustomPassIncGen

    LINK_LIBS PUBLIC
    MLIRArithDialect
    MLIRIR
    MLIRLinalgCustomDialect
    MLIRLinalgDialect
    MLIRMathDialect
    MLIRPass
    MLIRTransforms
)