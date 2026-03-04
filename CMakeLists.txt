cmake_minimum_required(VERSION 3.16)

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" PROJECT_VERSION)
string(STRIP "${PROJECT_VERSION}" PROJECT_VERSION)
set(SDFGLIB_VERSION "${PROJECT_VERSION}")

project(sdfglib VERSION ${SDFGLIB_VERSION} DESCRIPTION "Core SDFG library for generating and manipulating Stateful DataFlow Graphs" LANGUAGES CXX)

include(cmake/CPM.cmake)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

CPMAddPackage(
  NAME Boost
  VERSION 1.86.0 # Versions less than 1.85.0 may need patches for installation targets.
  URL https://github.com/boostorg/boost/releases/download/boost-1.86.0/boost-1.86.0-cmake.tar.xz
  URL_HASH SHA256=2c5ec5edcdff47ff55e27ed9560b0a0b94b07bd07ed9928b476150e16b0efc57
  OPTIONS "BOOST_ENABLE_CMAKE ON" "BOOST_SKIP_INSTALL_RULES ON" # Set `OFF` for installation
          "BUILD_SHARED_LIBS OFF" "BOOST_INCLUDE_LIBRARIES graph\\\;program_options" # Note the escapes!
)

# Boost::boost is the legacy name for the header-only target (now Boost::headers).
# Transitive dependencies may still link against it, so re-export it as an imported target.
if(NOT TARGET Boost::boost)
  add_library(Boost::boost INTERFACE IMPORTED)
  set_target_properties(Boost::boost PROPERTIES INTERFACE_LINK_LIBRARIES Boost::headers)
endif()

# Tests Options
option(SDFG_BUILD_TESTS "Build tests" ON)

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
add_subdirectory(examples/applications)
add_subdirectory(opt)
add_subdirectory(rpc)
add_subdirectory(rtl)
add_subdirectory(arg-capture-io)
add_subdirectory(tutorial/printf_target)
add_subdirectory(targets)

option(PYTHON_BUILD_FRONTEND "Build Python frontend" OFF)
if(PYTHON_BUILD_FRONTEND)
    add_subdirectory(python)
endif()

option(MLIR_BUILD_FRONTEND "Build MLIR frontend" OFF)
if(MLIR_BUILD_FRONTEND)
    add_subdirectory(mlir)
endif()
