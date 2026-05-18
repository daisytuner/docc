cmake_minimum_required(VERSION 3.26)

file(READ "${CMAKE_SOURCE_DIR}/VERSION" PROJECT_VERSION)
string(STRIP "${PROJECT_VERSION}" PROJECT_VERSION)
set(DOCC_LLVM_VERSION "${PROJECT_VERSION}")

project(docc-llvm VERSION ${PROJECT_VERSION} DESCRIPTION "Daisytuner Optimizing Compiler Collection")

option(RELEASE_PACKAGE "Use the plain version, instead of a version suffixed with git hash" OFF)

include("${CMAKE_SOURCE_DIR}/cmake/PackageDependencies.cmake")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "Setup for packaging docc-llvm releases")
# Versioned prefix inside the system prefix: /usr/<prefix>...
set(PARENT_PROJECT_INSTALL_PREFIX "lib/docc-llvm-${DOCC_LLVM_VERSION}/")

install(FILES "${CMAKE_SOURCE_DIR}/cmake/PackageDependencies.cmake"
        DESTINATION "${PARENT_PROJECT_INSTALL_PREFIX}/cmake"
        COMPONENT docc-dev)

# include the all components
if(NOT TARGET docc::core)
    find_package(sdfg CONFIG REQUIRED)
endif()
if(NOT TARGET docc::opt)
    find_package(sdfgopt CONFIG REQUIRED)
endif()
if(NOT TARGET docc::rpc)
    find_package(sdfgrpc CONFIG REQUIRED)
endif()
if(DOCC_BUILD_TARGET_TENSTORRENT AND (NOT TARGET docc::target::tenstorrent))
    find_package(docc-target-tenstorrent CONFIG REQUIRED)
endif()
if(NOT TARGET docc::c-compile)
    find_package(docc-c-compile CONFIG REQUIRED)
endif()

# LLVM dependency
find_package(LLVM 19.1 REQUIRED CONFIG)
if (LLVM_VERSION VERSION_GREATER_EQUAL "20.0.0")
  message(FATAL_ERROR "LLVM version too new: ${LLVM_VERSION}")
endif()
if (LLVM_VERSION VERSION_LESS "19.0.0")
  message(FATAL_ERROR "LLVM version too old: ${LLVM_VERSION}")
endif()

# LLVM cmake configurations
include(AddLLVM)

# LLVM paths
add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})

llvm_map_components_to_libnames(LLVM_LIBS support core irreader analysis transformutils scalaropts ipo)

find_package(nlohmann_json REQUIRED)

set(SOURCE_FILES
  src/analysis/analysis.cpp
  src/analysis/global_cfg_analysis.cpp
  src/analysis/sdfg_registry.cpp
  src/lifting/functions/blas_lifting.cpp
  src/lifting/functions/function_lifting.cpp
  src/lifting/functions/intrinsic_lifting.cpp
  src/lifting/functions/libfunc_lifting.cpp
  src/lifting/function_to_sdfg.cpp
  src/lifting/lift_report.cpp
  src/lifting/lifting.cpp
  src/lifting/utils.cpp
  src/passes/code_generation/code_generation_pass.cpp
  src/passes/sdfg_simplify_pass.cpp
  src/passes/dumps/dump_attributes_pass.cpp
  src/passes/dumps/dump_sdfg_pass.cpp
  src/passes/dumps/global_cfg_dump_pass.cpp
  src/passes/dumps/opt_report_pass.cpp
  src/passes/dumps/pass_report_collector.cpp
  src/passes/inlining/argument_expansion_pass.cpp
  src/passes/scheduling/docc_backend_context.cpp
  src/passes/scheduling/einsum_pass.cpp
  src/passes/scheduling/normalization_pass.cpp
  src/passes/scheduling/scheduling_pass.cpp
  src/passes/function_to_sdfg_pass.cpp
  src/cmd_args.cpp
  src/docc_paths.cpp
  src/docc.cpp
)

# Adds a shared library for the plugin

add_llvm_library(docc_llvm_pass SHARED
    ${SOURCE_FILES}
    src/plugin.cpp
)
target_include_directories(docc_llvm_pass
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
if(DOCC_BUILD_TARGET_TENSTORRENT)
    target_link_libraries(
        docc_llvm_pass
        PUBLIC
            nlohmann_json::nlohmann_json
            docc::core
            docc::opt
            docc::rpc
            docc::target::tenstorrent
            docc::c-compile
            LLVM
    )
else()
    target_link_libraries(
        docc_llvm_pass
        PUBLIC
            nlohmann_json::nlohmann_json
            docc::core
            docc::opt
            docc::rpc
            docc::c-compile
            LLVM
    )
endif()

target_package_deps(docc_llvm_pass
    TOOLS clang-19
)


target_compile_options (docc_llvm_pass PUBLIC -fexceptions)

add_library(docc::llvm-plugin ALIAS docc_llvm_pass)

install(TARGETS docc_llvm_pass LIBRARY DESTINATION lib/docc-llvm-${DOCC_LLVM_VERSION}/lib COMPONENT llvm-plugin)

# Adds a static library for tests

add_library(docc_llvm_pass_internal STATIC
   ${SOURCE_FILES}
)
target_include_directories(docc_llvm_pass_internal PUBLIC
   ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(
    docc_llvm_pass_internal
    PUBLIC
        ${LLVM_LIBS}
        nlohmann_json::nlohmann_json
        docc::core
        docc::opt
        docc::rpc
        docc::target::tenstorrent
        docc::c-compile
)
target_compile_options(docc_llvm_pass_internal PUBLIC -fexceptions)

if(SDFG_ENABLE_SANITIZER)
  target_compile_options(docc_llvm_pass_internal PUBLIC ${_SAN_COMPILE_OPTS})
  target_link_options   (docc_llvm_pass_internal PUBLIC ${_SAN_LINK_OPTS})
endif()

# Add coverage flags if enabled
if(SDFG_ENABLE_COVERAGE)
  target_compile_options(docc_llvm_pass_internal PRIVATE
    -fprofile-instr-generate
    -fcoverage-mapping
    -O0
    -g
    -fno-inline
  )
  target_link_options(docc_llvm_pass_internal PRIVATE
    -fprofile-instr-generate
  )
endif()

add_subdirectory(driver)

option(LLVM_BUILD_TESTS "Build tests" OFF)
if(LLVM_BUILD_TESTS)
    add_subdirectory(tests)
endif()


# ----------------------------- Packaging configuration -----------------------------
include(GNUInstallDirs)

option(PACKAGE_WITH_TOOL_DEPS "Include manual tool dependencies in the package metadata (e.g. clang-19, lld-19, which are only called but not linked and thus not covered by automatic detection)" OFF)

set(CPACK_GENERATOR "DEB") # default. Recommended to use -G DEB or -G RPM anyway

set(CPACK_PACKAGE_VENDOR "Daisytuner")
set(CPACK_PACKAGE_CONTACT "support@daisytuner.com")
set(CPACK_PACKAGE_VERSION ${DOCC_LLVM_VERSION})
set(CPACK_RELEASE_PACKAGE ${RELEASE_PACKAGE})

set(CPACK_PACKAGE_NAME "docc-llvm")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Daisytuner Optimizing Compiler Collection")

# Debian-specific settings
set(CPACK_DEBIAN_PACKAGE_SECTION "devel")

if(PACKAGE_WITH_TOOL_DEPS)
    get_package_dependencies(DETECTED_DEB_DEPENDS docc::llvm-plugin docc-analysis docc-ld docc-cpp docc)
endif()

set(DEB_PACKAGE_DEPENDS
    ${DETECTED_DEB_DEPENDS}
)
list(REMOVE_DUPLICATES DEB_PACKAGE_DEPENDS)

list(JOIN DEB_PACKAGE_DEPENDS ", " CPACK_DEBIAN_PACKAGE_DEPENDS)
set(CPACK_DEBIAN_COMPRESSION_TYPE "xz")

set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
set(CPACK_DEB_COMPONENT_INSTALL ON)
set(CPACK_COMPONENTS_ALL runtime docc-driver llvm-plugin compile-runtime sdfglib-development)
set(CPACK_COMPONENTS_GROUPING ALL_COMPONENTS_IN_ONE)

set(CPACK_DEBIAN_PACKAGE_NAME "${CPACK_PACKAGE_NAME}")

# RPM-specific settings
set(CPACK_RPM_COMPONENT_INSTALL ON)

if(PACKAGE_WITH_TOOL_DEPS)
    get_package_dependencies(DETECTED_RPM_DEPENDS docc::llvm-plugin docc-analysis docc-ld docc-cpp docc)
endif()

set(RPM_PACKAGE_DEPENDS
    ${DETECTED_RPM_DEPENDS}
)
list(REMOVE_DUPLICATES RPM_PACKAGE_DEPENDS)

set(CPACK_RPM_SPEC_MORE_DEFINE "%define __requires_exclude ^libisl\\\\.so.*$")

list(JOIN RPM_PACKAGE_DEPENDS ", " CPACK_RPM_PACKAGE_REQUIRES)

configure_file(CPackOptions.cmake.in CPackOptions.cmake @ONLY)
set(CPACK_PROJECT_CONFIG_FILE "${CMAKE_CURRENT_BINARY_DIR}/CPackOptions.cmake")

include(CPack)
