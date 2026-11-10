cmake_minimum_required(VERSION 3.16)

# ----------------------------- Version -----------------------------
# MAJOR.MINOR is committed in VERSION_BASE and only edited for a minor or major bump. Release builds get the
# full MAJOR.MINOR.PATCH, allocated by the Daisytuner backend, from the DOCC_RELEASE_VERSION environment
# variable; builds without it are dev builds, identified by the commit hash (see parse_version in
# cmake/Version.cmake). scripts/docc_version.py applies the same rule to the Python packages.
#   DOCC_VERSION       numeric version for project() and library sonames (MAJOR.MINOR in dev builds)
#   DOCC_VERSION_FULL  the release version, or MAJOR.MINOR+g<hash>; names the docc-llvm install prefix,
#                      `docc --version` and the package version (see llvm/CMakeLists.txt)
include(cmake/Version.cmake)
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION_BASE" DOCC_VERSION_BASE LIMIT_COUNT 1)
string(STRIP "${DOCC_VERSION_BASE}" DOCC_VERSION_BASE)
# file(STRINGS) is not tracked by the build system; re-run CMake when VERSION_BASE is edited.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION_BASE")
parse_version(
    BASE "${DOCC_VERSION_BASE}"
    RELEASE_ENV DOCC_RELEASE_VERSION
    GIT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
    OUT_VERSION DOCC_VERSION
    OUT_VERSION_FULL DOCC_VERSION_FULL
)
message(STATUS "docc version: ${DOCC_VERSION_FULL}")
set(SDFGLIB_VERSION "${DOCC_VERSION}")

project(sdfglib VERSION ${SDFGLIB_VERSION} DESCRIPTION "Core SDFG library for generating and manipulating Stateful DataFlow Graphs" LANGUAGES CXX)

include(cmake/CPM.cmake)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(DAISY_STACKTRACE_LIBS Boost::stacktrace Boost::stacktrace_from_exception)
    set(USE_BOOST_EXC_STACKTRACE_INJECTION ON)
    # We link against the GCC runtime (libstdc++), not the libc++ runtime, so the
    # libc++-specific memory leak in from_exception does not apply.
    add_compile_definitions(BOOST_STACKTRACE_LIBCXX_RUNTIME_MAY_CAUSE_MEMORY_LEAK)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(DAISY_STACK_TRACCE_LIBS Boost::stacktrace)
    set(USE_BOOST_EXC_STACKTRACE_INJECTION OFF)
endif()

CPMAddPackage(
  NAME Boost
  VERSION 1.90.0
  URL https://github.com/boostorg/boost/releases/download/boost-1.90.0/boost-1.90.0-cmake.tar.xz
  URL_HASH SHA256=aca59f889f0f32028ad88ba6764582b63c916ce5f77b31289ad19421a96c555f
  OPTIONS "BOOST_ENABLE_CMAKE ON" "BOOST_SKIP_INSTALL_RULES ON" # Set `OFF` for installation
          "BOOST_STACKTRACE_ENABLE_FROM_EXCEPTION ${USE_BOOST_EXC_STACKTRACE_INJECTION}"
          "BUILD_SHARED_LIBS OFF" "BOOST_INCLUDE_LIBRARIES graph\\\;program_options\\\;stacktrace" # Note the escapes!
)

# Boost::boost is the legacy name for the header-only target (now Boost::headers).
# Transitive dependencies may still link against it, so re-export it as an imported target.
if(NOT TARGET Boost::boost)
  add_library(Boost::boost INTERFACE IMPORTED)
  set_target_properties(Boost::boost PROPERTIES INTERFACE_LINK_LIBRARIES Boost::headers)
endif()

# Tests Options
option(SDFG_BUILD_TESTS "Build tests" ON)
if(SDFG_BUILD_TESTS)
    enable_testing()
endif()

# Compiler options
set(SDFG_COMPILE_OPTS
  -fstack-protector-strong
  -D_FORTIFY_SOURCE=3
)

# Sanitizer Options
option(SDFG_ENABLE_SANITIZER "Enable Address/Memory/Undefined/Leak/Thread sanitizers" OFF)
set(SDFG_SANITIZER "address" CACHE STRING
    "List of sanitizers to enable; supported: \
     address, memory, undefined, leak, thread")
if(SDFG_ENABLE_SANITIZER)
  message(STATUS "Enabling sanitizer: ${SDFG_SANITIZER}")
  set(_SAN_COMPILE_OPTS
    -fno-omit-frame-pointer
    -fsanitize=${SDFG_SANITIZER}
  )
  set(_SAN_LINK_OPTS
    -fsanitize=${SDFG_SANITIZER}
  )
endif()

# Coverage option
option(SDFG_ENABLE_COVERAGE "Enable coverage flags" OFF)

if(NOT DEFINED PARENT_PROJECT_INSTALL_PREFIX)
    set(PARENT_PROJECT_INSTALL_PREFIX "" CACHE STRING
            "Path prefix (trailing /!) for installing sdfglib artifacts. For example 'lib/docc-0.1.6/'" )
endif()

add_subdirectory(sdfg)
add_subdirectory(opt)
add_subdirectory(rpc)
add_subdirectory(rtl)
add_subdirectory(c-compile)
add_subdirectory(arg-capture-io)
add_subdirectory(tutorial/printf_target)
add_subdirectory(targets)

option(LLVM_BUILD_FRONTEND "Build LLVM frontend" OFF)
if(LLVM_BUILD_FRONTEND)
    add_subdirectory(llvm)
endif()

option(PYTHON_BUILD_FRONTEND "Build Python frontend" OFF)
if(PYTHON_BUILD_FRONTEND)
    add_subdirectory(python)
endif()

option(MLIR_BUILD_FRONTEND "Build MLIR frontend" OFF)
if(MLIR_BUILD_FRONTEND)
    add_subdirectory(mlir)
endif()
