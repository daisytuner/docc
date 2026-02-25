#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <sys/types.h>
#include <unordered_map>
#include <vector>

#include "base64.h"

namespace arg_capture {

struct MyHash {
    std::size_t operator()(const std::pair<int32_t, bool>& p) const {
        return std::hash<int32_t>()(p.first) ^ p.second ? 0x40000000 : 0;
    }
};

struct ArgCapture {
    int32_t arg_idx;
    bool after;
    /**
     * innermost dimension is always the element size
     */
    const std::vector<size_t> dims{0};
    int primitive_type;
    std::shared_ptr<const std::filesystem::path> ext_file;
    std::shared_ptr<const std::vector<uint8_t>> data;


    ArgCapture() = default;

    ArgCapture(int32_t idx, bool after, int primitive_type, const std::vector<size_t> dims)
        : arg_idx(idx), after(after), dims(dims), primitive_type(primitive_type) {}

    ArgCapture(const ArgCapture& other)
        : arg_idx(other.arg_idx), after(other.after), dims(other.dims), primitive_type(other.primitive_type),
          ext_file(other.ext_file), data(other.data) {}

    ArgCapture(const ArgCapture&& other) noexcept
        : arg_idx(other.arg_idx), after(other.after), dims(std::move(other.dims)), primitive_type(other.primitive_type),
          ext_file(std::move(other.ext_file)), data(std::move(other.data)) {}

    void serialize_into(nlohmann::json& j) const;

    static void parse_from(
        const nlohmann::json& entry,
        std::unordered_map<std::pair<int32_t, bool>, ArgCapture, MyHash>& map,
        std::filesystem::path base_path = std::filesystem::path()
    );
};


class ArgCaptureIO {
protected:
    std::string name_;
    std::unordered_map<std::string, uint32_t> invokes_ = std::unordered_map<std::string, uint32_t>{};
    std::unordered_map<std::string, std::unordered_map<std::pair<int32_t, bool>, ArgCapture, MyHash>> current_captures_;

public:
    explicit ArgCaptureIO(const char* name) : name_(name) {}
    const std::string& get_name() const;
    uint32_t get_current_invocation(std::string element_id) const;

    void invocation(std::string element_id);

    void clear();

    bool create_and_capture_inline(
        int arg_idx,
        bool after,
        int primitive_type,
        const std::vector<size_t>& dims,
        const void* data,
        std::string node_id = std::string{}
    );
    bool create_and_capture_to_file(
        int arg_idx,
        bool after,
        int primitive_type,
        const std::vector<size_t>& dims,
        std::filesystem::path& file,
        const void* data,
        std::string node_id = std::string{}
    );

    bool capture_inline(ArgCapture& capture, const void* data, std::string node_id = std::string{});
    bool write_capture_to_file(ArgCapture& capture, std::filesystem::path file, const void* data);

    void write_index(std::filesystem::path base_file);

    const std::unordered_map<std::string, std::unordered_map<std::pair<int32_t, bool>, ArgCapture, MyHash>>& get_captures()
        const;

    template<typename T = ArgCaptureIO>
    static std::shared_ptr<T> from_index(const std::filesystem::path& file);

    static bool write_data_to_raw_file(const std::filesystem::path& file, const void* src_data, size_t size);

    static void read_data_from_raw_file(const std::filesystem::path& file, void* write_ptr, size_t size);
};


static const uint32_t INDEX_FORMAT_VERSION = 0x00000001;

template<typename T>
std::shared_ptr<T> ArgCaptureIO::from_index(const std::filesystem::path& file) {
    if (!std::filesystem::exists(file)) {
        throw std::runtime_error("Index file does not exist: " + file.string());
    }

    std::ifstream ifs(file);
    if (!ifs.is_open()) {
        throw std::runtime_error("Failed to open index file for reading: " + file.string());
    }

    nlohmann::json j;
    ifs >> j;

    if (j["format"].get<uint32_t>() != INDEX_FORMAT_VERSION) {
        throw std::runtime_error("Unsupported index format version");
    }

    auto name = j["target"].get<std::string>();
    auto invokes = j["invocation"].get<uint32_t>();
    auto element_id = j["element_id"].get<std::string>();

    auto captureIO = std::make_shared<T>(name.c_str());
    captureIO->invokes_[element_id] = invokes;

    auto& capture_map = captureIO->current_captures_[element_id];
    for (const auto& entry : j["captures"]) {
        ArgCapture::parse_from(entry, capture_map, file.parent_path());
    }

    return captureIO;
}

} // namespace arg_capture
