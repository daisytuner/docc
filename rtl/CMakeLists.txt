include(GNUInstallDirs)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(daisy_rtl SHARED
    src/capture.cpp
    src/instrumentation.cpp
)

target_include_directories(daisy_rtl PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${PARENT_PROJECT_INSTALL_PREFIX}include>
)

target_compile_options(daisy_rtl PRIVATE ${SDFG_COMPILE_OPTS})

# Don't link against PAPI!
# Just link against the dynamic linker
target_link_libraries(daisy_rtl PRIVATE dl sdfg::arg_capture_io)

install(TARGETS daisy_rtl
    EXPORT sdfglibTargets
    LIBRARY
        DESTINATION ${PARENT_PROJECT_INSTALL_PREFIX}lib
        COMPONENT compile-runtime
#        NAMELINK_COMPONENT compile-runtime   # the .so / .so.MAJOR symlinks if the version is baked into python and we link the full version name, we can use this to let scikit only pack the full-named version
    ARCHIVE DESTINATION ${PARENT_PROJECT_INSTALL_PREFIX}lib COMPONENT compile-runtime
    PUBLIC_HEADER DESTINATION ${PARENT_PROJECT_INSTALL_PREFIX}include COMPONENT compile-runtime
    INCLUDES DESTINATION ${PARENT_PROJECT_INSTALL_PREFIX}include
)

install(DIRECTORY include/daisy_rtl DESTINATION ${PARENT_PROJECT_INSTALL_PREFIX}include COMPONENT compile-runtime)
