
option(DOCC_BUILD_TARGET_ET "Build ETSOC backend" OFF)
if(DOCC_BUILD_TARGET_ET)
    add_subdirectory(et)
endif()

option(DOCC_BUILD_TARGET_TENSTORRENT "Build Tenstorrent backend" ON)
if(DOCC_BUILD_TARGET_TENSTORRENT)
    add_subdirectory(tenstorrent)
endif()

option(DOCC_BUILD_TARGET_AXELERA "Build Axelera (Metis/Europa) backend" ON)
if(DOCC_BUILD_TARGET_AXELERA)
    add_subdirectory(axelera)
endif()
