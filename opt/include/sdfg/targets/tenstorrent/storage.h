#pragma once

#include "sdfg/types/type.h"

namespace sdfg {
namespace tenstorrent {

inline types::StorageType StorageType_Tenstorrent_DRAM{"Tenstorrent_DRAM"};
inline types::StorageType StorageType_Tenstorrent_SRAM{"Tenstorrent_SRAM"};
inline types::StorageType StorageType_Tenstorrent_Local{"Tenstorrent_Local"};

} // namespace tenstorrent
} // namespace sdfg
