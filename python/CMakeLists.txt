cmake_minimum_required(VERSION 3.26)

find_package(pybind11 CONFIG REQUIRED)
find_package(nlohmann_json REQUIRED)

# Define module
set(PYTHON_SOURCES
    bindings/bindings.cpp
    bindings/py_structured_sdfg.cpp
    bindings/builder/py_structured_sdfg_builder.cpp
    bindings/data_flow/py_cmath.cpp
    bindings/data_flow/py_tasklet.cpp
    bindings/types/py_types.cpp
)

pybind11_add_module(sdfg-py ${PYTHON_SOURCES})
set_target_properties(sdfg-py PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    OUTPUT_NAME "_sdfg"
)
target_include_directories(sdfg-py
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/bindings>
)

target_link_libraries(sdfg-py PRIVATE sdfg::sdfg sdfg::opt sdfg::rpc nlohmann_json::nlohmann_json ${CMAKE_DL_LIBS})

if(TARGET docc::target::et)
    target_link_libraries(sdfg-py PRIVATE docc::target::et)
endif()

# Install artifacts for the Python wheel
install(TARGETS sdfg-py DESTINATION docc/sdfg COMPONENT docc_python)
