include(GNUInstallDirs)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(DOCC_RT_ET_BUILD_MODE "" CACHE STRING "allows to compile the et-runtime to force device or sysemu use")
set(ET_PATH "/opt/et" CACHE PATH "Path to the et installation, used to find the et sysemu config header")

list(APPEND CMAKE_MODULE_PATH ${ET_PATH}/lib/cmake)

add_library(docc-rt-et STATIC
    src/runtime.cpp
)

if(DOCC_RT_ET_BUILD_MODE STREQUAL "device")
    target_compile_definitions(docc-rt-et PRIVATE DOCC_RT_ET_MODE=DOCC_RT_ET_MODE_DEVICE)
elseif (DOCC_RT_ET_BUILD_MODE STREQUAL "sysemu")
    target_compile_definitions(docc-rt-et PRIVATE DOCC_RT_ET_MODE=DOCC_RT_ET_MODE_SYSEMU)
else()
    target_compile_definitions(docc-rt-et PRIVATE DOCC_RT_ET_MODE=DOCC_RT_ET_MODE_SYSEMU)
endif()

find_package(libcap REQUIRED)
find_package(runtime REQUIRED)
find_package(deviceLayer REQUIRED)
find_package(sw-sysemu REQUIRED)
find_package(EsperantoBootLoader REQUIRED)
find_package(EsperantoDeviceMinionRuntime REQUIRED)
find_package(esperantoTrace REQUIRED)

get_property(BOOTROM_TRAMPOLINE_TO_BL2_ELF TARGET EsperantoBootLoader::BootromTrampolineToBL2.elf PROPERTY LOCATION)
get_property(BL2_ELF_SIM TARGET EsperantoBootLoader::ServiceProcessorBL2_fast-boot.elf PROPERTY LOCATION)
get_property(BL2_ELF_PROD TARGET EsperantoBootLoader::ServiceProcessorBL2_production.elf PROPERTY LOCATION)
get_property(MASTER_MINION_ELF TARGET EsperantoDeviceMinionRuntime::MasterMinion.elf PROPERTY LOCATION)
get_property(MACHINE_MINION_ELF TARGET EsperantoDeviceMinionRuntime::MachineMinion.elf PROPERTY LOCATION)
get_property(WORKER_MINION_ELF TARGET EsperantoDeviceMinionRuntime::WorkerMinion.elf PROPERTY LOCATION)

configure_file(include/docc/rt/et/et-sysemu-config.h.in ${CMAKE_CURRENT_BINARY_DIR}/genInclude/docc/rt/et/et-sysemu-config.h)

target_link_libraries(docc-rt-et PRIVATE
    runtime::etrt
    deviceLayer::deviceLayer
    sw-sysemu::sw-sysemu
    esperantoTrace::et_trace
)

target_include_directories(docc-rt-et PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/genInclude>
    $<INSTALL_INTERFACE:${PARENT_PROJECT_INSTALL_PREFIX}include>
)

set_target_properties(daisy_rtl PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
)

install(TARGETS docc-rt-et
        EXPORT docc-rtTargets
        LIBRARY DESTINATION ${PARENT_PROJECT_INSTALL_PREFIX}lib COMPONENT compile-runtime
        ARCHIVE DESTINATION ${PARENT_PROJECT_INSTALL_PREFIX}lib COMPONENT compile-runtime
        PUBLIC_HEADER DESTINATION ${PARENT_PROJECT_INSTALL_PREFIX}include COMPONENT compile-runtime
        INCLUDES DESTINATION ${PARENT_PROJECT_INSTALL_PREFIX}include
)
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/genInclude/docc DESTINATION ${PARENT_PROJECT_INSTALL_PREFIX}include COMPONENT compile-runtime)
install(DIRECTORY include/docc
        DESTINATION ${PARENT_PROJECT_INSTALL_PREFIX}include
        COMPONENT compile-runtime
        FILES_MATCHING PATTERN "*.h"
)

install(DIRECTORY libexec/docc DESTINATION ${PARENT_PROJECT_INSTALL_PREFIX}libexec COMPONENT compile-runtime)
