cmake_minimum_required(VERSION 3.26)

find_package(pybind11 CONFIG REQUIRED)
find_package(nlohmann_json REQUIRED)

# Define module
set(PYTHON_SOURCES
    bindings/bindings.cpp
    bindings/py_structured_sdfg.cpp
    bindings/builder/py_structured_sdfg_builder.cpp
    bindings/control_flow/py_control_flow.cpp
    bindings/data_flow/py_cmath.cpp
    bindings/data_flow/py_code_node.cpp
    bindings/data_flow/py_data_flow_graph.cpp
    bindings/data_flow/py_data_flow_node.cpp
    bindings/data_flow/py_memlet.cpp
    bindings/data_flow/py_tasklet.cpp
    bindings/targets/target_mapping.cpp
    bindings/transformations/py_transformations.cpp
    bindings/types/py_types.cpp
)

pybind11_add_module(sdfg-py ${PYTHON_SOURCES})
set_target_properties(sdfg-py PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    OUTPUT_NAME "_sdfg"
)

add_custom_command(TARGET sdfg-py POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E create_symlink
        "$<TARGET_FILE:sdfg-py>"
        "${CMAKE_CURRENT_SOURCE_DIR}/docc/sdfg/$<TARGET_FILE_NAME:sdfg-py>"
        COMMENT "Creating symlink to docc.sdfg._sdfg in src dir"
)

target_include_directories(sdfg-py
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/bindings>
)

target_link_libraries(sdfg-py PRIVATE sdfg::sdfg sdfg::opt sdfg::rpc docc::c-compile nlohmann_json::nlohmann_json ${CMAKE_DL_LIBS})

if(TARGET docc::target::et)
    target_link_libraries(sdfg-py PRIVATE docc::target::et)
endif()

# Install artifacts for the Python wheel
install(TARGETS sdfg-py DESTINATION docc/sdfg COMPONENT docc_python)
