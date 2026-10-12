set(SOURCE_FILES
    src/axelera/axelera_scheduler.cpp
    src/axelera/axscript_code_generator.cpp
    src/axelera/axscript_language_extension.cpp
    src/axelera/datapath.cpp
    src/axelera/datapath_dispatcher.cpp
    src/axelera/plugin.cpp
)

add_library(docc-target-axelera
    ${SOURCE_FILES}
)
add_library(docc::target::axelera ALIAS docc-target-axelera)
target_include_directories(docc-target-axelera
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${PARENT_PROJECT_INSTALL_PREFIX}include>
)

target_compile_definitions(docc-target-axelera PUBLIC
    DOCC_HAS_TARGET_AXELERA
)

target_compile_options(docc-target-axelera PRIVATE ${SDFG_COMPILE_OPTS})
if(SDFG_ENABLE_SANITIZER)
  target_compile_options(docc-target-axelera PUBLIC ${_SAN_COMPILE_OPTS})
  target_link_options   (docc-target-axelera PUBLIC ${_SAN_LINK_OPTS})
endif()

target_link_libraries(docc-target-axelera PUBLIC docc::core docc::opt)
set_target_properties(docc-target-axelera PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

## INSTALLATION

install(TARGETS docc-target-axelera
    EXPORT docc-target-axeleraTargets
    LIBRARY DESTINATION ${PARENT_PROJECT_INSTALL_PREFIX}lib COMPONENT target-axelera
    ARCHIVE DESTINATION ${PARENT_PROJECT_INSTALL_PREFIX}lib COMPONENT target-axelera
    PUBLIC_HEADER DESTINATION ${PARENT_PROJECT_INSTALL_PREFIX}include COMPONENT target-axelera
    INCLUDES DESTINATION ${PARENT_PROJECT_INSTALL_PREFIX}include
)
install(DIRECTORY include/docc DESTINATION ${PARENT_PROJECT_INSTALL_PREFIX}include COMPONENT target-axelera)

install(EXPORT docc-target-axeleraTargets
    NAMESPACE docc::target::
    FILE docc-target-axeleraTargets.cmake
    DESTINATION ${PARENT_PROJECT_INSTALL_PREFIX}lib/cmake/docc COMPONENT target-axelera
)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/docc-target-axeleraConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    ${CMAKE_CURRENT_LIST_DIR}/cmake/docc-target-axeleraConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/docc-target-axeleraConfig.cmake"
    INSTALL_DESTINATION ${PARENT_PROJECT_INSTALL_PREFIX}lib/cmake/docc
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/docc-target-axeleraConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/docc-target-axeleraConfigVersion.cmake"
    DESTINATION ${PARENT_PROJECT_INSTALL_PREFIX}lib/cmake/docc COMPONENT target-axelera
)

if(SDFG_BUILD_TESTS)
    add_subdirectory(tests)
endif()
