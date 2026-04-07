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

# Link against only the MLIR libraries we need (reduces wheel size significantly)
target_link_libraries(sdfg-mlir PRIVATE
    # Core MLIR
    MLIRIR
    MLIRSupport
    MLIRParser
    MLIRPass

    # Only the dialects we use
    MLIRArithDialect
    MLIRControlFlowDialect
    MLIRFuncDialect
    MLIRLinalgCustomDialect
    MLIRLinalgDialect
    MLIRMathDialect
    MLIRTensorDialect

    # Linalg passes (for convert())
    MLIRLinalgCustomTransforms

    # SDFG target and libraries
    MLIRTargetSDFG
    sdfg::sdfg
    sdfg::opt
    sdfg::rpc
    nlohmann_json::nlohmann_json
)

target_include_directories(sdfg-mlir PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include
)

add_custom_command(TARGET sdfg-mlir POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E create_symlink
        "$<TARGET_FILE:sdfg-mlir>"
        "${CMAKE_CURRENT_SOURCE_DIR}/../docc/mlir/$<TARGET_FILE_NAME:sdfg-mlir>"
        COMMENT "Creating symlink to docc.mlir._sdfg_mlir in src dir"
)

# Install the Python module to the docc/mlir package directory
install(TARGETS sdfg-mlir DESTINATION docc/mlir COMPONENT docc_mlir)
