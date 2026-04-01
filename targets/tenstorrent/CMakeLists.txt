
set(SOURCE_FILES
    src/tenstorrent/blas/dot.cpp
    src/tenstorrent/blas/gemm.cpp
    src/tenstorrent/codegen.cpp
    src/tenstorrent/library_node_dispatcher.cpp
    src/tenstorrent/library_node_mapping.cpp
    src/tenstorrent/math_node_implementation_override_pass.cpp
    src/tenstorrent/plugin.cpp
    src/tenstorrent/schedule.cpp
    src/tenstorrent/tenstorrent_create_device.cpp
    src/tenstorrent/tenstorrent_offloading_expansion.cpp
    src/tenstorrent/tenstorrent_offloading_node.cpp
    src/tenstorrent/tenstorrent_scheduler.cpp
    src/tenstorrent/tenstorrent_transform.cpp
)

add_library(docc-target-tenstorrent
    ${SOURCE_FILES}
)
add_library(docc::target::tenstorrent ALIAS docc-target-tenstorrent)
target_include_directories(docc-target-tenstorrent
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${PARENT_PROJECT_INSTALL_PREFIX}include>
)

target_compile_definitions(docc-target-tenstorrent PUBLIC
    DOCC_HAS_TARGET_TENSTORRENT
)

# Sanitizers
if(SDFG_ENABLE_SANITIZER)
  target_compile_options(docc-target-tenstorrent PUBLIC ${_SAN_COMPILE_OPTS})
  target_link_options   (docc-target-tenstorrent PUBLIC ${_SAN_LINK_OPTS})
endif()

# Coverage
if(SDFG_ENABLE_COVERAGE)
  target_compile_options(docc-target-tenstorrent PRIVATE
    -fprofile-instr-generate
    -fcoverage-mapping
    -O0
    -g
    -fno-inline
  )
  target_link_options(docc-target-tenstorrent PRIVATE
    -fprofile-instr-generate
  )
endif()

target_link_libraries(docc-target-tenstorrent PUBLIC docc::core docc::opt)
set_target_properties(docc-target-tenstorrent PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

## INSTALLATION

install(TARGETS docc-target-tenstorrent
    EXPORT docc-target-tenstorrentTargets
    LIBRARY DESTINATION ${PARENT_PROJECT_INSTALL_PREFIX}lib COMPONENT target-tenstorrent
    ARCHIVE DESTINATION ${PARENT_PROJECT_INSTALL_PREFIX}lib COMPONENT target-tenstorrent
    PUBLIC_HEADER DESTINATION ${PARENT_PROJECT_INSTALL_PREFIX}include COMPONENT target-tenstorrent
    INCLUDES DESTINATION ${PARENT_PROJECT_INSTALL_PREFIX}include
)
install(DIRECTORY include/docc DESTINATION ${PARENT_PROJECT_INSTALL_PREFIX}include COMPONENT target-tenstorrent)

## CMAKE CONFIGURATION

install(EXPORT docc-target-tenstorrentTargets
    NAMESPACE docc::target::
    FILE docc-target-tenstorrentTargets.cmake
    DESTINATION ${PARENT_PROJECT_INSTALL_PREFIX}lib/cmake/docc COMPONENT target-tenstorrent
)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/docc-target-tenstorrentConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    ${CMAKE_CURRENT_LIST_DIR}/cmake/docc-target-tenstorrentConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/docc-target-tenstorrentConfig.cmake"
    INSTALL_DESTINATION ${PARENT_PROJECT_INSTALL_PREFIX}lib/cmake/docc
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/docc-target-tenstorrentConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/docc-target-tenstorrentConfigVersion.cmake"
    DESTINATION ${PARENT_PROJECT_INSTALL_PREFIX}lib/cmake/docc COMPONENT target-tenstorrent
)

if(SDFG_BUILD_TESTS)
    set(INSTALL_GTEST OFF)
    add_subdirectory(tests)
endif()

#add_subdirectory(rt)
