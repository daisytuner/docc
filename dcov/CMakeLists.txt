find_package(Boost REQUIRED COMPONENTS program_options)
find_package(nlohmann_json REQUIRED)

find_library(ZSTD_LIBRARY NAMES zstd REQUIRED)
find_path(ZSTD_INCLUDE_DIR NAMES zstd.h REQUIRED)

add_library(dcov
    src/region_builder.cpp
    src/region_serializer.cpp
    src/region_differ.cpp
    src/trace_loader.cpp
    src/pack.cpp
)
add_library(docc::dcov ALIAS dcov)

target_include_directories(dcov
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${ZSTD_INCLUDE_DIR}
)
target_link_libraries(dcov PUBLIC
    sdfg::sdfg
    nlohmann_json::nlohmann_json
    ${ZSTD_LIBRARY}
)
target_compile_options(dcov PRIVATE ${SDFG_COMPILE_OPTS})

add_executable(dcov-tool
    src/main.cpp
)
set_target_properties(dcov-tool PROPERTIES OUTPUT_NAME dcov)
target_link_libraries(dcov-tool PRIVATE
    docc::dcov
    sdfg::sdfg
    Boost::program_options
)
