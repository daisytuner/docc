

set(SOURCE_FILES
    src/util/offloading_instrumentation_plan.cpp
    src/util/offloading_type_collector.cpp
)
add_library(docc-c-compile
    ${SOURCE_FILES}
)
add_library(docc::c-compile ALIAS docc-c-compile)
target_include_directories(docc-c-compile
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Sanitizers
if(SDFG_ENABLE_SANITIZER)
  target_compile_options(docc-c-compile PUBLIC ${_SAN_COMPILE_OPTS})
  target_link_options   (docc-c-compile PUBLIC ${_SAN_LINK_OPTS})
endif()

# Coverage
if(SDFG_ENABLE_COVERAGE)
  target_compile_options(docc-c-compile PRIVATE
    -fprofile-instr-generate
    -fcoverage-mapping
    -O0
    -g
    -fno-inline
  )
  target_link_options(docc-c-compile PRIVATE
    -fprofile-instr-generate
  )
endif()

target_link_libraries(docc-c-compile PUBLIC docc::core docc::opt docc::target::tenstorrent)
set_target_properties(docc-c-compile PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

## INSTALLATION

install(TARGETS docc-c-compile
    EXPORT docc-c-compileTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(DIRECTORY include/sdfg DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

## CMAKE CONFIGURATION

install(EXPORT docc-c-compileTargets
    NAMESPACE docc::
    FILE docc-c-compileTargets.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sdfg
)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/docc-c-compileConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    ${CMAKE_CURRENT_LIST_DIR}/cmake/docc-c-compileConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/docc-c-compileConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sdfg
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/docc-c-compileConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/docc-c-compileConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sdfg
)

if(SDFG_BUILD_TESTS)
    set(INSTALL_GTEST OFF)
    add_subdirectory(tests)
endif()
