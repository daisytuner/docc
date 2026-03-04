
option(DOCC_BUILD_TARGET_ET "Build ETSOC backend" OFF)
if(DOCC_BUILD_TARGET_ET)
    add_subdirectory(et)
endif()
