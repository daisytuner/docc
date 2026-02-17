include(GNUInstallDirs)

if(NOT DEFINED PARENT_PROJECT_INSTALL_PREFIX)
    set(PARENT_PROJECT_INSTALL_PREFIX "" CACHE STRING
           "Path prefix (trailing /!) for installing sdfglib artifacts. For example 'lib/docc-0.1.6/'" )
endif()

# Enable packaging of symengine
set(_CMAKE_INSTALL_DEFAULT_COMPONENT_NAME ${CMAKE_INSTALL_DEFAULT_COMPONENT_NAME})
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME sdfglib-development)

set(WITH_SYMENGINE_THREAD_SAFE ON CACHE BOOL "override" FORCE)
set(WITH_SYMENGINE_RCP ON CACHE BOOL "override" FORCE)

add_subdirectory(3rdParty/symengine)
# Restore install dirs for sub-projects to avoid conflicts
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME ${_CMAKE_INSTALL_DEFAULT_COMPONENT_NAME})

# NLohmann json
find_package(nlohmann_json REQUIRED)


# Find the ISL library
find_package(PkgConfig REQUIRED)
pkg_check_modules(ISL REQUIRED IMPORTED_TARGET isl)

set(SOURCE_FILES
    src/analysis/analysis.cpp
    src/analysis/arguments_analysis.cpp
    src/analysis/assumptions_analysis.cpp
    src/analysis/control_flow_analysis.cpp
    src/analysis/data_dependency_analysis.cpp
    src/analysis/escape_analysis.cpp
    src/analysis/flop_analysis.cpp
    src/analysis/dominance_analysis.cpp
    src/analysis/loop_analysis.cpp
    src/analysis/mem_access_range_analysis.cpp
    src/analysis/memlet_delinearization_analysis.cpp
    src/analysis/reference_analysis.cpp
    src/analysis/scope_analysis.cpp
    src/analysis/type_analysis.cpp
    src/analysis/users.cpp
    src/builder/function_builder.cpp
    src/builder/sdfg_builder.cpp
    src/builder/structured_sdfg_builder.cpp
    src/codegen/utils.cpp
    src/codegen/code_snippet_factory.cpp
    src/codegen/instrumentation/arg_capture_plan.cpp
    src/codegen/instrumentation/instrumentation_info.cpp
    src/codegen/instrumentation/instrumentation_plan.cpp
    src/codegen/language_extensions/c_language_extension.cpp
    src/codegen/language_extensions/cpp_language_extension.cpp
    src/codegen/language_extensions/cuda_language_extension.cpp
    src/codegen/code_generators/c_code_generator.cpp
    src/codegen/code_generators/c_style_base_code_generator.cpp
    src/codegen/code_generators/cpp_code_generator.cpp
    src/codegen/code_generators/cuda_code_generator.cpp
    src/codegen/dispatchers/node_dispatcher.cpp
    src/codegen/dispatchers/node_dispatcher_registry.cpp
    src/codegen/dispatchers/block_dispatcher.cpp
    src/codegen/dispatchers/if_else_dispatcher.cpp
    src/codegen/dispatchers/sequence_dispatcher.cpp
    src/codegen/dispatchers/for_dispatcher.cpp
    src/codegen/dispatchers/map_dispatcher.cpp
    src/codegen/dispatchers/while_dispatcher.cpp
    src/codegen/loop_report.cpp
    src/control_flow/interstate_edge.cpp
    src/control_flow/state.cpp
    src/cutouts/cutout_serializer.cpp
    src/cutouts/cutouts.cpp
    src/data_flow/access_node.cpp
    src/data_flow/code_node.cpp
    src/data_flow/data_flow_graph.cpp
    src/data_flow/data_flow_node.cpp
    src/data_flow/library_node.cpp
    src/data_flow/memlet.cpp
    src/data_flow/tasklet.cpp
    src/data_flow/library_nodes/barrier_local_node.cpp
    src/data_flow/library_nodes/call_node.cpp
    src/data_flow/library_nodes/invoke_node.cpp
    src/data_flow/library_nodes/metadata_node.cpp
    src/data_flow/library_nodes/math/math_node.cpp
    src/data_flow/library_nodes/math/blas/blas_node.cpp
    src/data_flow/library_nodes/math/blas/dot_node.cpp
    src/data_flow/library_nodes/math/blas/gemm_node.cpp
    src/data_flow/library_nodes/math/cmath/cmath_node.cpp
    src/data_flow/library_nodes/math/tensor/tensor_node.cpp
    src/data_flow/library_nodes/math/tensor/elementwise_node.cpp
    src/data_flow/library_nodes/math/tensor/reduce_node.cpp
    src/data_flow/library_nodes/math/tensor/broadcast_node.cpp
    src/data_flow/library_nodes/math/tensor/conv_node.cpp
    src/data_flow/library_nodes/math/tensor/transpose_node.cpp
    src/data_flow/library_nodes/math/tensor/elementwise_ops/abs_node.cpp
    src/data_flow/library_nodes/math/tensor/elementwise_ops/add_node.cpp
    src/data_flow/library_nodes/math/tensor/elementwise_ops/cast_node.cpp
    src/data_flow/library_nodes/math/tensor/elementwise_ops/minimum_node.cpp
    src/data_flow/library_nodes/math/tensor/elementwise_ops/maximum_node.cpp
    src/data_flow/library_nodes/math/tensor/elementwise_ops/div_node.cpp
    src/data_flow/library_nodes/math/tensor/elementwise_ops/elu_node.cpp
    src/data_flow/library_nodes/math/tensor/elementwise_ops/exp_node.cpp
    src/data_flow/library_nodes/math/tensor/elementwise_ops/erf_node.cpp
    src/data_flow/library_nodes/math/tensor/elementwise_ops/fill_node.cpp
    src/data_flow/library_nodes/math/tensor/elementwise_ops/hard_sigmoid_node.cpp
    src/data_flow/library_nodes/math/tensor/elementwise_ops/leaky_relu_node.cpp
    src/data_flow/library_nodes/math/tensor/elementwise_ops/mul_node.cpp
    src/data_flow/library_nodes/math/tensor/elementwise_ops/pow_node.cpp
    src/data_flow/library_nodes/math/tensor/elementwise_ops/relu_node.cpp
    src/data_flow/library_nodes/math/tensor/elementwise_ops/sigmoid_node.cpp
    src/data_flow/library_nodes/math/tensor/elementwise_ops/sqrt_node.cpp
    src/data_flow/library_nodes/math/tensor/elementwise_ops/sub_node.cpp
    src/data_flow/library_nodes/math/tensor/elementwise_ops/tanh_node.cpp
    src/data_flow/library_nodes/math/tensor/reduce_ops/sum_node.cpp
    src/data_flow/library_nodes/math/tensor/reduce_ops/mean_node.cpp
    src/data_flow/library_nodes/math/tensor/reduce_ops/std_node.cpp
    src/data_flow/library_nodes/math/tensor/reduce_ops/max_node.cpp
    src/data_flow/library_nodes/math/tensor/reduce_ops/min_node.cpp
    src/data_flow/library_nodes/math/tensor/reduce_ops/softmax_node.cpp
    src/data_flow/library_nodes/stdlib/alloca.cpp
    src/data_flow/library_nodes/stdlib/calloc.cpp
    src/data_flow/library_nodes/stdlib/free.cpp
    src/data_flow/library_nodes/stdlib/malloc.cpp
    src/data_flow/library_nodes/stdlib/memcpy.cpp
    src/data_flow/library_nodes/stdlib/memmove.cpp
    src/data_flow/library_nodes/stdlib/memset.cpp
    src/data_flow/library_nodes/stdlib/trap.cpp
    src/data_flow/library_nodes/stdlib/unreachable.cpp
    src/deepcopy/structured_sdfg_deep_copy.cpp
    src/graph/graph.cpp
    src/optimization_report/optimization_report.cpp
    src/passes/pass.cpp
    src/passes/pipeline.cpp
    src/passes/statistics.cpp
    src/passes/dataflow/dead_data_elimination.cpp
    src/passes/dataflow/dead_reference_elimination.cpp
    src/passes/dataflow/constant_propagation.cpp
    src/passes/dataflow/tasklet_fusion.cpp
    src/passes/dataflow/trivial_array_elimination.cpp
    src/passes/dataflow/trivial_reference_conversion.cpp
    src/passes/dataflow/reference_propagation.cpp
    src/passes/dataflow/byte_reference_elimination.cpp
    src/passes/dataflow/memlet_linearization.cpp
    src/passes/dataflow/tensor_to_pointer_conversion.cpp
    src/passes/memory/allocation_management.cpp
    src/passes/memory/insert_frees.cpp
    src/passes/schedules/expansion_pass.cpp
    src/passes/symbolic/symbol_evolution.cpp
    src/passes/symbolic/symbol_propagation.cpp
    src/passes/symbolic/symbol_promotion.cpp
    src/passes/symbolic/type_minimization.cpp
    src/passes/structured_control_flow/block_fusion.cpp
    src/passes/structured_control_flow/common_assignment_elimination.cpp
    src/passes/structured_control_flow/condition_elimination.cpp
    src/passes/structured_control_flow/dead_cfg_elimination.cpp
    src/passes/structured_control_flow/for2map.cpp
    src/passes/structured_control_flow/pointer_evolution.cpp
    src/passes/structured_control_flow/loop_normalization.cpp
    src/passes/structured_control_flow/sequence_fusion.cpp
    src/passes/structured_control_flow/unify_loop_exits.cpp
    src/passes/structured_control_flow/while_to_for_conversion.cpp
    src/passes/debug_info_propagation.cpp
    src/serializer/json_serializer.cpp
    src/structured_control_flow/block.cpp
    src/structured_control_flow/control_flow_node.cpp
    src/structured_control_flow/for.cpp
    src/structured_control_flow/if_else.cpp
    src/structured_control_flow/map.cpp
    src/structured_control_flow/return.cpp
    src/structured_control_flow/sequence.cpp
    src/structured_control_flow/structured_loop.cpp
    src/structured_control_flow/while.cpp
    src/symbolic/assumptions.cpp
    src/symbolic/conjunctive_normal_form.cpp
    src/symbolic/extreme_values.cpp
    src/symbolic/maps.cpp
    src/symbolic/polynomials.cpp
    src/symbolic/series.cpp
    src/symbolic/sets.cpp
    src/symbolic/symbolic.cpp
    src/symbolic/utils.cpp
    src/transformations/loop_distribute.cpp
    src/transformations/loop_interchange.cpp
    src/transformations/loop_skewing.cpp
    src/transformations/loop_tiling.cpp
    src/transformations/out_local_storage.cpp
    src/transformations/recorder.cpp
    src/transformations/replayer.cpp
    src/transformations/utils.cpp
    src/types/array.cpp
    src/types/function.cpp
    src/types/pointer.cpp
    src/types/scalar.cpp
    src/types/structure.cpp
    src/types/tensor.cpp
    src/types/utils.cpp
    src/visitor/immutable_structured_sdfg_visitor.cpp
    src/visitor/structured_sdfg_visitor.cpp
    src/visualizer/dot_visualizer.cpp
    src/visualizer/visualizer.cpp
    src/element.cpp
    src/function.cpp
    src/sdfg.cpp
    src/structured_sdfg.cpp
)

# Original Library
add_library(sdfglib
    ${SOURCE_FILES}
)
add_library(sdfg::sdfg ALIAS sdfglib)
add_library(sdfglib::sdfglib ALIAS sdfglib)
target_include_directories(sdfglib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/3rdParty/symengine>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/3rdParty/symengine>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/3rdParty/symengine/symengine/utilities/cereal/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/3rdParty/symengine/symengine/utilities/cereal/include>
)

if(SDFG_ENABLE_SANITIZER)
  target_compile_options(sdfglib PUBLIC ${_SAN_COMPILE_OPTS})
  target_link_options   (sdfglib PUBLIC ${_SAN_LINK_OPTS})
endif()

# Add coverage flags if enabled
if(SDFG_ENABLE_COVERAGE)
  target_compile_options(sdfglib PRIVATE
    -fprofile-instr-generate
    -fcoverage-mapping
    -O0
    -g
    -fno-inline
  )
  target_link_options(sdfglib PRIVATE
    -fprofile-instr-generate
  )
endif()


target_link_libraries(sdfglib PUBLIC
    nlohmann_json::nlohmann_json
    $<BUILD_INTERFACE:Boost::graph>
    PkgConfig::ISL
    symengine
)
set_target_properties(sdfglib PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

## INSTALLATION

install(TARGETS sdfglib
    EXPORT sdfglibTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT sdfglib-development
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT sdfglib-development
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT sdfglib-development
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(DIRECTORY include/sdfg DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT sdfglib-development)

## CMAKE CONFIGURATION

install(EXPORT sdfglibTargets
    NAMESPACE sdfglib::
    FILE sdfglibTargets.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sdfglib
    COMPONENT sdfglib-development
)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/sdfglibConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    ${CMAKE_CURRENT_LIST_DIR}/cmake/sdfglibConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/sdfglibConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sdfglib
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/sdfglibConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/sdfglibConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sdfglib
    COMPONENT sdfglib-development
)

if(SDFG_BUILD_TESTS)
    set(INSTALL_GTEST OFF)
    add_subdirectory(tests)
endif()

# Documentation
option(SDFG_BUILD_DOCS "Build documentation using Doxygen" OFF)
if(SDFG_BUILD_DOCS)
    find_package(Doxygen)
    if(DOXYGEN_FOUND)
        set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile)
        set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

        # Copy Doxyfile to build directory
        configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

        # Add custom target for documentation generation
        add_custom_target(docs
            COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Generating API documentation with Doxygen"
            VERBATIM
        )

        message(STATUS "Doxygen found. Use 'make docs' to generate documentation.")
    else()
        message(WARNING "Doxygen not found. Documentation will not be generated.")
    endif()
endif()
