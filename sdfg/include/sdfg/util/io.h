#pragma once

#include <fstream>
#include <string>

namespace sdfg {
namespace utils {
namespace io {

inline void atomic_write(const std::string& path, const std::string& content) {
    std::ofstream file(path);
    file << content;
}

} // namespace io
} // namespace utils
} // namespace sdfg
