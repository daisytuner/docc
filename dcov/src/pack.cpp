#include "sdfg/dcov/pack.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iterator>
#include <map>
#include <regex>
#include <stdexcept>

#include <nlohmann/json.hpp>
#include <zstd.h>

#include "sdfg/dcov/region_serializer.h"

namespace sdfg {
namespace dcov {

namespace {

uint64_t fnv1a64(const uint8_t* data, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) {
        h ^= data[i];
        h *= 1099511628211ull;
    }
    return h;
}

void put_u32(std::ostream& os, uint32_t v) {
    uint8_t b[4] = {
        static_cast<uint8_t>(v),
        static_cast<uint8_t>(v >> 8),
        static_cast<uint8_t>(v >> 16),
        static_cast<uint8_t>(v >> 24)
    };
    os.write(reinterpret_cast<const char*>(b), 4);
}

void put_u64(std::ostream& os, uint64_t v) {
    uint8_t b[8];
    for (int i = 0; i < 8; ++i) b[i] = static_cast<uint8_t>(v >> (8 * i));
    os.write(reinterpret_cast<const char*>(b), 8);
}

uint32_t get_u32(std::istream& is) {
    uint8_t b[4];
    is.read(reinterpret_cast<char*>(b), 4);
    return static_cast<uint32_t>(b[0]) | (static_cast<uint32_t>(b[1]) << 8) | (static_cast<uint32_t>(b[2]) << 16) |
           (static_cast<uint32_t>(b[3]) << 24);
}

uint64_t get_u64(std::istream& is) {
    uint8_t b[8];
    is.read(reinterpret_cast<char*>(b), 8);
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= static_cast<uint64_t>(b[i]) << (8 * i);
    return v;
}

std::vector<uint8_t> zstd_compress(const uint8_t* src, size_t n) {
    size_t bound = ZSTD_compressBound(n);
    std::vector<uint8_t> out(bound);
    size_t w = ZSTD_compress(out.data(), bound, src, n, 9);
    if (ZSTD_isError(w)) throw std::runtime_error(std::string("zstd compress failed: ") + ZSTD_getErrorName(w));
    out.resize(w);
    return out;
}

std::vector<uint8_t> zstd_decompress(const uint8_t* src, size_t n, uint64_t raw_len) {
    std::vector<uint8_t> out(raw_len);
    if (raw_len == 0) return out;
    size_t w = ZSTD_decompress(out.data(), raw_len, src, n);
    if (ZSTD_isError(w) || w != raw_len) throw std::runtime_error("zstd decompress failed or size mismatch");
    return out;
}

/// Write one chunk: tag[4] | u64 comp_len | u64 raw_len | compressed bytes.
void write_chunk(std::ostream& os, const char tag[4], const uint8_t* raw, size_t raw_len) {
    std::vector<uint8_t> comp = raw_len ? zstd_compress(raw, raw_len) : std::vector<uint8_t>{};
    os.write(tag, 4);
    put_u64(os, comp.size());
    put_u64(os, raw_len);
    if (!comp.empty()) os.write(reinterpret_cast<const char*>(comp.data()), static_cast<std::streamsize>(comp.size()));
}

void write_chunk(std::ostream& os, const char tag[4], const std::string& raw) {
    write_chunk(os, tag, reinterpret_cast<const uint8_t*>(raw.data()), raw.size());
}

std::string iso8601_utc_now() {
    std::time_t t = std::time(nullptr);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buf;
}

/// Collect capture records + raw bytes from an arg-capture directory.
/// Builds the CAPS manifest (with blob-store offsets/checksums) and the
/// concatenated blob store.
void collect_captures(
    const std::filesystem::path& dir, std::vector<CaptureRecord>& records, std::vector<uint8_t>& blob_store
) {
    if (dir.empty() || !std::filesystem::is_directory(dir)) return;

    // Deterministic order: sort index files by name.
    std::vector<std::filesystem::path> index_files;
    for (const auto& e : std::filesystem::directory_iterator(dir)) {
        if (e.is_regular_file() && e.path().filename().string().find(".index.json") != std::string::npos)
            index_files.push_back(e.path());
    }
    std::sort(index_files.begin(), index_files.end());

    for (const auto& idx_path : index_files) {
        std::ifstream in(idx_path);
        if (!in.is_open()) continue;
        nlohmann::json j;
        try {
            in >> j;
        } catch (const std::exception&) {
            continue;
        }

        const size_t element_id = std::stoull(j.value("element_id", std::string("0")));
        const int invocation = j.value("invocation", 0);
        const std::string target = j.value("target", std::string());
        const int format = j.value("format", 0);

        if (!j.contains("captures") || !j["captures"].is_array()) continue;

        for (const auto& c : j["captures"]) {
            CaptureRecord rec;
            rec.element_id = element_id;
            rec.invocation = invocation;
            rec.target = target;
            rec.format = format;
            rec.arg_idx = c.value("arg_idx", 0);
            rec.after = c.value("after", false);
            rec.primitive_type = c.value("primitive_type", 0);
            rec.ext_file = c.value("ext_file", std::string());
            if (c.contains("dims") && c["dims"].is_array())
                for (const auto& d : c["dims"]) rec.dims.push_back(d.get<int64_t>());

            std::vector<uint8_t> bytes;
            if (!rec.ext_file.empty()) {
                std::ifstream bin(dir / rec.ext_file, std::ios::binary);
                if (bin.is_open()) {
                    bytes.assign(std::istreambuf_iterator<char>(bin), std::istreambuf_iterator<char>());
                }
            }
            rec.offset = blob_store.size();
            rec.size = bytes.size();
            rec.checksum = fnv1a64(bytes.data(), bytes.size());
            blob_store.insert(blob_store.end(), bytes.begin(), bytes.end());
            records.push_back(std::move(rec));
        }
    }
}

nlohmann::json caps_to_json(const std::vector<CaptureRecord>& records) {
    nlohmann::json arr = nlohmann::json::array();
    for (const auto& r : records) {
        arr.push_back({
            {"element_id", r.element_id},
            {"invocation", r.invocation},
            {"target", r.target},
            {"arg_idx", r.arg_idx},
            {"after", r.after},
            {"dims", r.dims},
            {"primitive_type", r.primitive_type},
            {"format", r.format},
            {"ext_file", r.ext_file},
            {"offset", r.offset},
            {"size", r.size},
            {"checksum", r.checksum},
        });
    }
    return arr;
}

std::vector<CaptureRecord> caps_from_json(const nlohmann::json& arr) {
    std::vector<CaptureRecord> out;
    if (!arr.is_array()) return out;
    for (const auto& c : arr) {
        CaptureRecord r;
        r.element_id = c.value("element_id", static_cast<size_t>(0));
        r.invocation = c.value("invocation", 0);
        r.target = c.value("target", std::string());
        r.arg_idx = c.value("arg_idx", 0);
        r.after = c.value("after", false);
        r.primitive_type = c.value("primitive_type", 0);
        r.format = c.value("format", 0);
        r.ext_file = c.value("ext_file", std::string());
        if (c.contains("dims") && c["dims"].is_array())
            for (const auto& d : c["dims"]) r.dims.push_back(d.get<int64_t>());
        r.offset = c.value("offset", static_cast<uint64_t>(0));
        r.size = c.value("size", static_cast<uint64_t>(0));
        r.checksum = c.value("checksum", static_cast<uint64_t>(0));
        out.push_back(std::move(r));
    }
    return out;
}

nlohmann::json cuts_to_json(const std::vector<CutoutRecord>& records) {
    nlohmann::json arr = nlohmann::json::array();
    for (const auto& r : records) {
        arr.push_back({
            {"region_key", r.region_key},
            {"display_key", r.display_key},
            {"element_type", r.element_type},
            {"element_id", r.element_id},
            {"format", r.format},
            {"offset", r.offset},
            {"size", r.size},
            {"checksum", r.checksum},
        });
    }
    return arr;
}

std::vector<CutoutRecord> cuts_from_json(const nlohmann::json& arr) {
    std::vector<CutoutRecord> out;
    if (!arr.is_array()) return out;
    for (const auto& c : arr) {
        CutoutRecord r;
        r.region_key = c.value("region_key", std::string());
        r.display_key = c.value("display_key", std::string());
        r.element_type = c.value("element_type", std::string());
        r.element_id = c.value("element_id", static_cast<size_t>(0));
        r.format = c.value("format", std::string("json"));
        r.offset = c.value("offset", static_cast<uint64_t>(0));
        r.size = c.value("size", static_cast<uint64_t>(0));
        r.checksum = c.value("checksum", static_cast<uint64_t>(0));
        out.push_back(std::move(r));
    }
    return out;
}

nlohmann::json meta_to_json(const PackMeta& m) {
    return nlohmann::json{
        {"tool_version", m.tool_version},
        {"created_utc", m.created_utc},
        {"kernel", m.kernel},
        {"module_id", m.module_id},
        {"sdfg_file", m.sdfg_file},
        {"trace_file", m.trace_file},
        {"sdfg_format", m.sdfg_format},
        {"capture_count", m.capture_count},
        {"cutout_count", m.cutout_count},
    };
}

PackMeta meta_from_json(const nlohmann::json& j) {
    PackMeta m;
    m.tool_version = j.value("tool_version", std::string());
    m.created_utc = j.value("created_utc", std::string());
    m.kernel = j.value("kernel", std::string());
    m.module_id = j.value("module_id", std::string());
    m.sdfg_file = j.value("sdfg_file", std::string());
    m.trace_file = j.value("trace_file", std::string());
    m.sdfg_format = j.value("sdfg_format", std::string("json"));
    m.capture_count = j.value("capture_count", static_cast<uint64_t>(0));
    m.cutout_count = j.value("cutout_count", static_cast<uint64_t>(0));
    return m;
}

} // namespace

void write_pack(
    const Module& module,
    const std::vector<uint8_t>& sdfg_bytes,
    const std::vector<CutoutInput>& cutouts,
    const std::filesystem::path& arg_capture_dir,
    const PackMeta& meta_in,
    const std::filesystem::path& out_path
) {
    std::vector<CaptureRecord> records;
    std::vector<uint8_t> blob_store;
    collect_captures(arg_capture_dir, records, blob_store);

    // Lay cutouts out into a single store with offsets/checksums.
    std::vector<CutoutRecord> cut_records;
    std::vector<uint8_t> cutout_store;
    for (const auto& c : cutouts) {
        CutoutRecord rec;
        rec.region_key = c.region_key;
        rec.display_key = c.display_key;
        rec.element_type = c.element_type;
        rec.element_id = c.element_id;
        rec.format = c.format;
        rec.offset = cutout_store.size();
        rec.size = c.bytes.size();
        rec.checksum = fnv1a64(c.bytes.data(), c.bytes.size());
        cutout_store.insert(cutout_store.end(), c.bytes.begin(), c.bytes.end());
        cut_records.push_back(std::move(rec));
    }

    PackMeta meta = meta_in;
    meta.capture_count = records.size();
    meta.cutout_count = cut_records.size();
    if (meta.created_utc.empty()) meta.created_utc = iso8601_utc_now();
    if (meta.module_id.empty()) meta.module_id = module.module_id;

    DcovJsonSerializer json_serializer;
    const std::string region_json = json_serializer.serialize(module);
    const std::string meta_json = meta_to_json(meta).dump();
    const std::string caps_json = caps_to_json(records).dump();
    const std::string cuts_json = cuts_to_json(cut_records).dump();

    std::ofstream os(out_path, std::ios::binary);
    if (!os.is_open()) throw std::runtime_error("cannot open output '" + out_path.string() + "'");

    os.write(PACK_MAGIC, 8);
    put_u32(os, PACK_VERSION);
    put_u32(os, PACK_FLAG_ZSTD);

    write_chunk(os, "META", meta_json);
    write_chunk(os, "SDFG", sdfg_bytes.data(), sdfg_bytes.size());
    write_chunk(os, "RGNS", region_json);
    write_chunk(os, "CAPS", caps_json);
    write_chunk(os, "BLBS", blob_store.data(), blob_store.size());
    write_chunk(os, "CUTS", cuts_json);
    write_chunk(os, "CUTB", cutout_store.data(), cutout_store.size());

    const char end_tag[4] = {'E', 'N', 'D', '\0'};
    write_chunk(os, end_tag, nullptr, 0);

    if (!os) throw std::runtime_error("failed while writing '" + out_path.string() + "'");
}

Pack read_pack(const std::filesystem::path& in_path) {
    std::ifstream is(in_path, std::ios::binary);
    if (!is.is_open()) throw std::runtime_error("cannot open pack '" + in_path.string() + "'");

    char magic[8];
    is.read(magic, 8);
    if (is.gcount() != 8 || std::memcmp(magic, PACK_MAGIC, 8) != 0)
        throw std::runtime_error("not a dcovpack file: '" + in_path.string() + "'");

    const uint32_t version = get_u32(is);
    if (version != PACK_VERSION) throw std::runtime_error("unsupported dcovpack version " + std::to_string(version));
    const uint32_t flags = get_u32(is);
    const bool zstd = (flags & PACK_FLAG_ZSTD) != 0;

    Pack pack;
    while (true) {
        char tag[4];
        is.read(tag, 4);
        if (is.gcount() != 4) break;
        const uint64_t comp_len = get_u64(is);
        const uint64_t raw_len = get_u64(is);

        if (std::memcmp(tag, "END\0", 4) == 0) break;

        std::vector<uint8_t> comp(comp_len);
        if (comp_len) is.read(reinterpret_cast<char*>(comp.data()), static_cast<std::streamsize>(comp_len));
        if (!is) throw std::runtime_error("truncated pack while reading chunk");

        std::vector<uint8_t> raw = zstd ? zstd_decompress(comp.data(), comp.size(), raw_len) : std::move(comp);

        if (std::memcmp(tag, "META", 4) == 0)
            pack.meta = meta_from_json(nlohmann::json::parse(raw.begin(), raw.end()));
        else if (std::memcmp(tag, "SDFG", 4) == 0)
            pack.sdfg_bytes = std::move(raw);
        else if (std::memcmp(tag, "RGNS", 4) == 0)
            pack.region_model_json.assign(raw.begin(), raw.end());
        else if (std::memcmp(tag, "CAPS", 4) == 0)
            pack.captures = caps_from_json(nlohmann::json::parse(raw.begin(), raw.end()));
        else if (std::memcmp(tag, "BLBS", 4) == 0)
            pack.blob_store = std::move(raw);
        else if (std::memcmp(tag, "CUTS", 4) == 0)
            pack.cutouts = cuts_from_json(nlohmann::json::parse(raw.begin(), raw.end()));
        else if (std::memcmp(tag, "CUTB", 4) == 0)
            pack.cutout_store = std::move(raw);
        // Unknown chunks are skipped for forward compatibility.
    }
    return pack;
}

} // namespace dcov
} // namespace sdfg
