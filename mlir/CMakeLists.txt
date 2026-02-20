cmake_minimum_required(VERSION 3.26)

project(mlir-sdfg VERSION 0.0.1 DESCRIPTION "MLIR to SDFG Conversion Library")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# This project is not built by default
# Enable with: cmake -DDOCC_BUILD_MLIR=ON

# Find MLIR and LLVM - canonical way for MLIR projects
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# Add MLIR and LLVM CMake module paths - this is the canonical way
# This gives access to AddMLIR, AddLLVM, and other MLIR CMake functions
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

# Include MLIR/LLVM CMake modules
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# MLIR/LLVM paths
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(lib)
add_subdirectory(tools)

option(MLIR_BUILD_TESTS "Build tests" OFF)
if(MLIR_BUILD_TESTS)
    add_subdirectory(test)
endif()

option(MLIR_BUILD_BINDINGS "Build MLIR bindings" OFF)
if(MLIR_BUILD_BINDINGS)
    add_subdirectory(bindings)
endif()
