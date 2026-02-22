find_package(pybind11 CONFIG REQUIRED)
find_package(nlohmann_json REQUIRED)

set(PYTHON_BINDING_SOURCES
    bindings.cpp
)

pybind11_add_module(sdfg-mlir ${PYTHON_BINDING_SOURCES})

set_target_properties(sdfg-mlir PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    OUTPUT_NAME "_sdfg_mlir"
)

# Link against MLIR libraries
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

target_link_libraries(sdfg-mlir PRIVATE
    ${dialect_libs}
    ${conversion_libs}
    ${translation_libs}
    MLIRPass
    MLIRParser
    MLIRSDFGDialect
    MLIRTargetSDFG
    MLIRConvertToSDFG
    MLIRArithToSDFG
    MLIRFuncToSDFG
    MLIRLinalgToSDFG
    MLIRTensorToSDFG
    sdfg::sdfg
    sdfg::opt
    sdfg::rpc
    nlohmann_json::nlohmann_json
)

target_include_directories(sdfg-mlir PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include
)

# Install the Python module to the docc/mlir package directory
install(TARGETS sdfg-mlir DESTINATION docc/mlir)
