add_subdirectory(sdfg-json-to-c)

configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.cmake.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
    MAIN_CONFIG
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.cmake.cfg.py
)

set(MLIR_SDFG_TEST_DEPENDS
    FileCheck
    count
    not
    mlir-opt
    docc-mlir-opt
    docc-mlir-translate
    sdfg-json-to-c
)

add_lit_testsuite(check-docc-mlir "Running the DOCC MLIR frontend regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${MLIR_SDFG_TEST_DEPENDS}
)
add_lit_testsuites(DOCC-MLIR-OPT ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${MLIR_SDFG_TEST_DEPENDS})