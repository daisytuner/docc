# Find SDFG
if(NOT TARGET sdfg::sdfg)
    find_package(sdfg CONFIG REQUIRED)
endif()

find_package(CURL REQUIRED)

set(SOURCE_FILES
    src/passes/rpc/daisytuner_rpc_context.cpp
    src/passes/rpc/rpc_context.cpp
    src/passes/rpc/rpc_scheduler.cpp
    src/transformations/rpc_node_transform.cpp
    src/util/utils_curl.cpp
)
add_library(sdfgrpc
    ${SOURCE_FILES}
)
add_library(sdfg::rpc ALIAS sdfgrpc)
target_include_directories(sdfgrpc
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Sanitizers
if(SDFG_ENABLE_SANITIZER)
  target_compile_options(sdfgrpc PUBLIC ${_SAN_COMPILE_OPTS})
  target_link_options   (sdfgrpc PUBLIC ${_SAN_LINK_OPTS})
endif()

# Coverage
if(SDFG_ENABLE_COVERAGE)
  target_compile_options(sdfgrpc PRIVATE
    -fprofile-instr-generate
    -fcoverage-mapping
    -O0
    -g
    -fno-inline
  )
  target_link_options(sdfgrpc PRIVATE
    -fprofile-instr-generate
  )
endif()

target_include_directories(sdfgrpc
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(sdfgrpc PUBLIC sdfg::sdfg sdfg::opt CURL::libcurl)
set_target_properties(sdfgrpc PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

## INSTALLATION

install(TARGETS sdfgrpc
    EXPORT sdfgrpcTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(DIRECTORY include/sdfg DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

## CMAKE CONFIGURATION

install(EXPORT sdfgrpcTargets
    NAMESPACE sdfg::
    FILE sdfgrpcTargets.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sdfg
)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/sdfgrpcConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    ${CMAKE_CURRENT_LIST_DIR}/cmake/sdfgrpcConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/sdfgrpcConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sdfg
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/sdfgrpcConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/sdfgrpcConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sdfg
)

if(SDFG_BUILD_TESTS)
    set(INSTALL_GTEST OFF)
    add_subdirectory(tests)
endif()
