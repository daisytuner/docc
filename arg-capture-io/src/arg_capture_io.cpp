#include "daisy_rtl/arg_capture_io.h"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <utility>

#include "daisy_rtl/base64.h"
#include "daisy_rtl/primitive_types.h"

#ifndef NDEBUG
#define DEBUG_PRINTLN(msg)                           \
    do {                                             \
        std::cout << "[DEBUG] " << msg << std::endl; \
    } while (0)
#define DEBUG_PRINT(msg)                \
    do {                                \
        std::cout << "[DEBUG] " << msg; \
    } while (0)
#else
#define DEBUG_PRINTLN(msg) \
    do {                   \
    } while (0)
#define DEBUG_PRINT(msg) \
    do {                 \
    } while (0)
#endif

namespace arg_capture {

void ArgCapture::serialize_into(nlohmann::json& j) const {
    nlohmann::json entry;

    entry["arg_idx"] = arg_idx;
    entry["after"] = after;
    entry["primitive_type"] = primitive_type;

    auto jDims = nlohmann::json::array();
    for (const auto& dim : dims) {
        jDims.push_back(dim);
    }
    entry["dims"] = jDims;

    if (data) {
        std::string base64_data = base64_encode(data.get()->data(), data.get()->size());
        entry["data"] = base64_data;
    } else if (ext_file) {
        entry["ext_file"] = ext_file->filename();
    }

    j.push_back(entry);
}

void ArgCapture::parse_from(
    const nlohmann::json& entry,
    std::unordered_map<std::pair<int32_t, bool>, ArgCapture, MyHash>& map,
    std::filesystem::path base_path
) {
    ArgCapture capture(
        entry["arg_idx"].get<int>(),
        entry["after"].get<bool>(),
        entry["primitive_type"].get<int>(),
        entry["dims"].get<std::vector<size_t>>()
    );

    if (entry.contains("data")) {
        auto data = base64_decode(entry["data"].get<std::string>());

        capture.data = std::make_shared<std::vector<uint8_t>>(std::move(data));
    } else if (entry.contains("ext_file")) {
        auto filename = entry["ext_file"].get<std::string>();
        capture.ext_file = std::make_shared<std::filesystem::path>(base_path / filename);
    }

    map.emplace(std::make_pair(capture.arg_idx, capture.after), std::move(capture));
}

const std::string& ArgCaptureIO::get_name() const { return name_; }

uint32_t ArgCaptureIO::get_current_invocation(std::string element_id) const { return invokes_.at(element_id); }

void ArgCaptureIO::invocation(std::string element_id) {
    invokes_[element_id] = (invokes_.count(element_id) > 0) ? invokes_[element_id] + 1 : 0;

    DEBUG_PRINTLN("Invoking '" << name_ << "' (" << invokes_[element_id] << ")");
}

void ArgCaptureIO::clear() { current_captures_.clear(); }

const std::unordered_map<std::string, std::unordered_map<std::pair<int32_t, bool>, ArgCapture, MyHash>>& ArgCaptureIO::
    get_captures() const {
    return current_captures_;
}

bool ArgCaptureIO::create_and_capture_inline(
    int arg_idx, bool after, int primitive_type, const std::vector<size_t>& dims, const void* data, std::string element_id
) {
    auto key = std::make_pair(arg_idx, after);

    // Erase the old entry to ensure we capture with FRESH dimensions and types.
    // If current_captures_ has an entry, emplace does nothing, keeping old dimensions.
    current_captures_[element_id].erase(key);
    auto it = current_captures_[element_id].emplace(key, ArgCapture(arg_idx, after, primitive_type, dims));

    return capture_inline(it.first->second, data, element_id);
}

bool ArgCaptureIO::create_and_capture_to_file(
    int arg_idx,
    bool after,
    int primitive_type,
    const std::vector<size_t>& dims,
    std::filesystem::path& file,
    const void* data,
    std::string element_id
) {
    auto key = std::make_pair(arg_idx, after);

    // Erase same as above
    current_captures_[element_id].erase(key);
    auto it = current_captures_[element_id].emplace(key, ArgCapture(arg_idx, after, primitive_type, dims));

    DEBUG_PRINTLN("Writing capture file " + file.string());
    return write_capture_to_file(it.first->second, file, data);
}

bool ArgCaptureIO::capture_inline(ArgCapture& capture, const void* data, std::string element_id) {
    auto size = std::accumulate(capture.dims.begin(), capture.dims.end(), 1, std::multiplies<size_t>());

    DEBUG_PRINT(
        "Capturing " << (capture.dims.size() - 1) << "D arg" << capture.arg_idx << ": type "
                     << primitive_type_names[capture.primitive_type] << "(" << size << " bytes): 0x" << std::hex
    );
    int perGroup = 0;
    // for (int i = 0; i < size; ++i) {
    //     const uint8_t* ptr = static_cast<const uint8_t*>(data) + i;
    //     uint32_t byte = *ptr;
    //     DEBUG_PRINT(byte);

    //     ++perGroup;
    //     if (perGroup == 16) {
    //         perGroup = 0;
    //         DEBUG_PRINT("\n\t");
    //     } else if (perGroup % 4 == 0) {
    //         DEBUG_PRINT(" ");
    //     }
    // }
    DEBUG_PRINTLN(std::dec);

    auto capturedData = std::make_shared<std::vector<uint8_t>>(size);
    std::memcpy(capturedData.get()->data(), data, size);

    current_captures_[element_id][std::make_pair(capture.arg_idx, capture.after)].data = std::move(capturedData);

    return true;
}

bool ArgCaptureIO::write_capture_to_file(ArgCapture& capture, std::filesystem::path file, const void* data) {
    DEBUG_PRINT(
        "Capturing " << (capture.dims.size() - 1) << "D arg" << capture.arg_idx << " as "
                     << (capture.after ? "result" : "input") << ": type "
                     << primitive_type_names[capture.primitive_type] << ": 0x" << std::hex << data << std::dec << ", "
    );
    for (size_t i = 0; i < capture.dims.size(); ++i) {
        auto dim = capture.dims.at(i);
        DEBUG_PRINT(dim);

        if (i < capture.dims.size() - 1) {
            DEBUG_PRINT(" * ");
        }
    }
    DEBUG_PRINTLN(" bytes");

    // Calculate size safely
    auto totalSize = std::accumulate(capture.dims.begin(), capture.dims.end(), size_t{1}, std::multiplies<size_t>());

    auto success = write_data_to_raw_file(file, data, totalSize);

    if (success) {
        capture.ext_file = std::make_shared<std::filesystem::path>(file);
    }

    return success;
}

bool ArgCaptureIO::write_data_to_raw_file(const std::filesystem::path& file, const void* src_data, size_t size) {
    // Ensure directory exists
    std::error_code ec;
    std::filesystem::create_directories(file.parent_path(), ec);
    if (ec) {
        std::cerr << "[ArgCaptureIO] Failed to create directories: " << ec.message() << std::endl;
        return false;
    }

    // Use standard stream without exceptions to capture errno correctly
    std::ofstream ofs(file, std::ofstream::binary | std::ofstream::out);

    if (!ofs.is_open()) {
        // Reset errno might be needed, but usually is_open fail sets it
        std::cerr << "[ArgCaptureIO] Failed to open file: " << file.string() << std::endl;
        std::cerr << "  System Error: " << std::strerror(errno) << std::endl;
        return false;
    }

    DEBUG_PRINTLN(" total size: " << size << " bytes");

    if (size > 0 && src_data == nullptr) {
        std::cerr << "[ArgCaptureIO] Error: Data pointer is NULL for size " << size << std::endl;
        return false;
    }

    // Perform write
    ofs.write(reinterpret_cast<const char*>(src_data), size);

    // Check for write errors using stream state + errno
    if (ofs.bad()) {
        std::cerr << "[ArgCaptureIO] Critical IO Error (badbit) writing " << file.string() << std::endl;
        std::cerr << "  Size attempted: " << size << " bytes" << std::endl;
        std::cerr << "  System Error: " << std::strerror(errno) << std::endl;
        return false;
    }
    if (ofs.fail()) {
        std::cerr << "[ArgCaptureIO] IO Failure (failbit) writing " << file.string() << std::endl;
        std::cerr << "  Size attempted: " << size << " bytes" << std::endl;
        std::cerr << "  System Error: " << std::strerror(errno) << std::endl;
        return false;
    }

    ofs.close();

    if (ofs.fail()) {
        std::cerr << "[ArgCaptureIO] Error closing file " << file.string() << std::endl;
        return false;
    }

    DEBUG_PRINTLN(" closed");

    return true;
}

void ArgCaptureIO::read_data_from_raw_file(const std::filesystem::path& file, void* write_ptr, size_t size) {
    std::ifstream ifs(file, std::ifstream::binary | std::ifstream::in);
    if (!ifs.is_open()) {
        throw std::runtime_error("Failed to open file: " + file.string());
    }

    ifs.read(reinterpret_cast<char*>(write_ptr), size);
    ifs.close();
}

void ArgCaptureIO::write_index(std::filesystem::path base_path) {
    std::filesystem::create_directories(base_path);

    for (const auto& [region_id, captures] : current_captures_) {
        auto file = base_path /
                    (name_ + "_inv" + std::to_string(invokes_.at(region_id)) + "_" + region_id + ".index.json");

        std::filesystem::create_directories(file.parent_path());

        DEBUG_PRINTLN("Writing index file " + file.string());

        std::ofstream ofs(file);
        if (!ofs.is_open()) {
            throw std::runtime_error("Failed to open index file for writing: " + file.string());
        }

        nlohmann::json j;
        j["format"] = INDEX_FORMAT_VERSION;
        j["target"] = name_;
        j["invocation"] = invokes_.at(region_id);
        j["element_id"] = region_id;

        auto arr = nlohmann::json::array();

        for (const auto& [key, capture] : captures) {
            capture.serialize_into(arr);
        }

        j["captures"] = arr;

        ofs << j;

        ofs.close();

        DEBUG_PRINTLN("Wrote capture index to " << file.string());
    }
}


} // namespace arg_capture
