include(GNUInstallDirs)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Static by default (each artifact embeds its own instrumentation state). The
# python/mlir/torch frontends set this ON so all artifacts in a process share a
# single RTL instance that aggregates into one trace.
option(DAISY_RTL_SHARED "Build the daisy RTL as a shared library" OFF)
if(DAISY_RTL_SHARED)
    set(_DAISY_RTL_LINKAGE SHARED)
else()
    set(_DAISY_RTL_LINKAGE STATIC)
endif()

add_library(daisy_rtl ${_DAISY_RTL_LINKAGE}
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

# Version the static archive only. A versioned shared library would install as a
# soname symlink chain (libdaisy_rtl.so -> .so.MAJOR -> .so.VERSION); wheel
# packaging drops the symlinks, leaving the plain name (needed by both the
# artifacts' DT_NEEDED and the python bindings) missing. Keep the shared RTL a
# single, unversioned libdaisy_rtl.so.
if(NOT DAISY_RTL_SHARED)
    set_target_properties(daisy_rtl PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
    )
endif()

install(TARGETS daisy_rtl
    EXPORT sdfglibTargets
    LIBRARY DESTINATION ${PARENT_PROJECT_INSTALL_PREFIX}lib COMPONENT compile-runtime
    ARCHIVE DESTINATION ${PARENT_PROJECT_INSTALL_PREFIX}lib COMPONENT compile-runtime
    PUBLIC_HEADER DESTINATION ${PARENT_PROJECT_INSTALL_PREFIX}include COMPONENT compile-runtime
    INCLUDES DESTINATION ${PARENT_PROJECT_INSTALL_PREFIX}include
)

install(DIRECTORY include/daisy_rtl DESTINATION ${PARENT_PROJECT_INSTALL_PREFIX}include COMPONENT compile-runtime)
