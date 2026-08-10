cmake_minimum_required(VERSION 3.26)

find_package(pybind11 CONFIG REQUIRED)
find_package(nlohmann_json REQUIRED)

# Define module
set(PYTHON_SOURCES
    bindings/bindings.cpp
    bindings/builder/py_structured_sdfg_builder.cpp
    bindings/control_flow/py_control_flow.cpp
    bindings/data_flow/py_cmath.cpp
    bindings/data_flow/py_code_node.cpp
    bindings/data_flow/py_data_flow_graph.cpp
    bindings/data_flow/py_data_flow_node.cpp
    bindings/data_flow/py_memlet.cpp
    bindings/data_flow/py_tasklet.cpp
    bindings/metrics/docc_metrics.cpp
    bindings/metrics/py_metrics.cpp
    bindings/passes/py_passes.cpp
    bindings/py_structured_sdfg.cpp
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

# Mirror the shared RTL into the source tree (docc/lib) so editable / in-build
# test runs discover it via docc.benchmarks.rtl._find_rtl_lib without a wheel
# install. A custom target is used because daisy_rtl lives in another directory.
# The symlink resolves to the same build artifact the compiled SDFGs load via
# their rpath, so python and the artifacts share one RTL instance.
if(DAISY_RTL_SHARED)
    add_custom_target(docc_link_rtl ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/docc/lib"
        COMMAND ${CMAKE_COMMAND} -E create_symlink
            "$<TARGET_FILE:daisy_rtl>"
            "${CMAKE_CURRENT_SOURCE_DIR}/docc/lib/$<TARGET_FILE_NAME:daisy_rtl>"
        COMMENT "Linking daisy_rtl into docc/lib for python discovery"
    )
    add_dependencies(docc_link_rtl daisy_rtl)
endif()

target_include_directories(sdfg-py
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/bindings>
)

if(DEFINED DOCC_CXX_COMPILER)
    target_compile_definitions(sdfg-py PRIVATE
        DOCC_CXX_COMPILER="${DOCC_CXX_COMPILER}"
    )
endif()

target_link_libraries(sdfg-py PRIVATE
    ${DAISY_STACKTRACE_LIBS}
    sdfg::sdfg
    sdfg::opt
    sdfg::rpc
    docc::c-compile
    nlohmann_json::nlohmann_json
    ${CMAKE_DL_LIBS}
)

if(TARGET docc::target::et)
    target_link_libraries(sdfg-py PRIVATE docc::target::et)
endif()

if(TARGET docc::target::tenstorrent)
    target_link_libraries(sdfg-py PRIVATE docc::target::tenstorrent)
endif()

# Install artifacts for the Python wheel
install(TARGETS sdfg-py DESTINATION docc/sdfg COMPONENT docc_python)

# Ship the RTL trace JSON schema as package data for docc.benchmarks. The canonical
# copy lives in rtl/schema (single source of truth, also consumed by the C++ RTL
# tests); installing it here places it next to docc/benchmarks/rtl.py in the wheel.
install(FILES ${PROJECT_SOURCE_DIR}/rtl/schema/daisy_trace.schema.json
    DESTINATION docc/benchmarks
    COMPONENT docc_python
)
