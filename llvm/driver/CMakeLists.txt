cmake_minimum_required(VERSION 3.26)

project(docc-llvm-driver VERSION 0.0.1 DESCRIPTION "Daisytuner Optimizing Compiler Collection")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# LLVM dependency
find_package(LLVM REQUIRED CONFIG 19.1)
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

set(driver_definitions
  DOCC_LLVM_VERSION="${DOCC_LLVM_VERSION}"
)

llvm_map_components_to_libnames(LLVM_LIBS support core irreader analysis transformutils scalaropts ipo)


add_executable(docc
	driver.cpp
)
target_compile_definitions(docc PRIVATE ${driver_definitions})
target_package_deps(docc
    TOOLS clang-19
)

install(TARGETS docc DESTINATION bin COMPONENT docc-driver)


add_executable(docc-cpp
	driver.cpp
)
target_compile_definitions(docc-cpp PRIVATE ${driver_definitions})
target_package_deps(docc-cpp
    TOOLS clang-19
)

install(TARGETS docc-cpp DESTINATION bin COMPONENT docc-driver)


add_executable(docc-ld
	linker.cpp
)
target_compile_definitions(docc-ld PRIVATE ${driver_definitions})
target_link_libraries(docc-ld PRIVATE ${LLVM_LIBS})
target_package_deps(docc-ld
    TOOLS lld-19
)

install(TARGETS docc-ld DESTINATION bin COMPONENT docc-driver)


add_executable(docc-analysis
    analysis.cpp
)
target_compile_definitions(docc-analysis PRIVATE ${driver_definitions})
target_compile_options(docc-analysis PRIVATE)
target_link_libraries(docc-analysis PRIVATE docc_llvm_pass)

install(TARGETS docc-analysis DESTINATION bin COMPONENT docc-driver)
set_target_properties(docc-analysis PROPERTIES
    INSTALL_RPATH "$ORIGIN/../lib/docc-llvm-${DOCC_LLVM_VERSION}/lib"
)
