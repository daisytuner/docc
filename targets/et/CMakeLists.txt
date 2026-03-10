
set(SOURCE_FILES
    src/et/target.cpp
    src/et/et_lib_node_mapper.cpp
    src/et/blas/gemm.cpp
)
add_library(docc-target-et
    ${SOURCE_FILES}
)
add_library(docc::target::et ALIAS docc-target-et)
target_include_directories(docc-target-et
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${PARENT_PROJECT_INSTALL_PREFIX}include>
)

target_compile_definitions(docc-target-et PUBLIC
    DOCC_HAS_TARGET_ET
)

# Sanitizers
if(SDFG_ENABLE_SANITIZER)
  target_compile_options(docc-target-et PUBLIC ${_SAN_COMPILE_OPTS})
  target_link_options   (docc-target-et PUBLIC ${_SAN_LINK_OPTS})
endif()

# Coverage
if(SDFG_ENABLE_COVERAGE)
  target_compile_options(docc-target-et PRIVATE
    -fprofile-instr-generate
    -fcoverage-mapping
    -O0
    -g
    -fno-inline
  )
  target_link_options(docc-target-et PRIVATE
    -fprofile-instr-generate
  )
endif()

target_link_libraries(docc-target-et PUBLIC docc::core docc::opt)
set_target_properties(docc-target-et PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

## INSTALLATION

install(TARGETS docc-target-et
    EXPORT docc-target-etTargets
    LIBRARY DESTINATION ${PARENT_PROJECT_INSTALL_PREFIX}lib COMPONENT target-et
    ARCHIVE DESTINATION ${PARENT_PROJECT_INSTALL_PREFIX}lib COMPONENT target-et
    PUBLIC_HEADER DESTINATION ${PARENT_PROJECT_INSTALL_PREFIX}include COMPONENT target-et
    INCLUDES DESTINATION ${PARENT_PROJECT_INSTALL_PREFIX}include
)
install(DIRECTORY include/docc DESTINATION ${PARENT_PROJECT_INSTALL_PREFIX}include COMPONENT target-et)

## CMAKE CONFIGURATION

install(EXPORT docc-target-etTargets
    NAMESPACE docc::target::
    FILE docc-target-etTargets.cmake
    DESTINATION ${PARENT_PROJECT_INSTALL_PREFIX}lib/cmake/docc COMPONENT target-et
)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/docc-target-etConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    ${CMAKE_CURRENT_LIST_DIR}/cmake/docc-target-etConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/docc-target-etConfig.cmake"
    INSTALL_DESTINATION ${PARENT_PROJECT_INSTALL_PREFIX}lib/cmake/docc
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/docc-target-etConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/docc-target-etConfigVersion.cmake"
    DESTINATION ${PARENT_PROJECT_INSTALL_PREFIX}lib/cmake/docc COMPONENT target-et
)

if(SDFG_BUILD_TESTS)
    set(INSTALL_GTEST OFF)
    add_subdirectory(tests)
endif()

add_subdirectory(rt)
