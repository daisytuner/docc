#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "sdfg/dcov/region.h"

namespace sdfg {
namespace dcov {

/**
 * @file pack.h
 * @brief A self-describing, shareable binary container (`.dcovpack`) bundling a
 *        region model, its runtime profile/metrics, and the raw argument
 *        captures into a single file.
 *
 * Layout (all integers little-endian):
 * ```
 * Header:  magic "DCOVPACK" (8)  | u32 version | u32 flags (bit0 = zstd)
 * Chunks:  tag[4] | u64 comp_len | u64 raw_len | bytes[comp_len]
 *   META   JSON: tool/format version, timestamp, kernel, module_id, sdfg/trace,
 *               serialization formats (sdfg_format)
 *   SDFG   the embedded SDFG, serialized in `meta.sdfg_format` (currently "json")
 *   RGNS   region-model JSON (regions + profiles + metrics)
 *   CAPS   JSON manifest: one record per capture (metadata + offset/size/checksum)
 *   BLBS   concatenated raw capture bytes (CAPS offsets index into this)
 *   CUTS   JSON manifest: one record per per-region cutout SDFG
 *   CUTB   concatenated cutout SDFG bytes (CUTS offsets index into this)
 *   END    terminator (comp_len = raw_len = 0)
 * ```
 *
 * The SDFG and cutout payloads are stored as opaque bytes tagged with a format
 * string, so a future non-JSON SDFG serialization can be swapped in transparently.
 */

constexpr char PACK_MAGIC[8] = {'D', 'C', 'O', 'V', 'P', 'A', 'C', 'K'};
constexpr uint32_t PACK_VERSION = 1;
constexpr uint32_t PACK_FLAG_ZSTD = 0x1;

/// Free-form provenance recorded in the META chunk.
struct PackMeta {
    std::string tool_version;
    std::string created_utc;
    std::string kernel;
    std::string module_id;
    std::string sdfg_file; ///< SDFG stage the model was built from (the runtime matcher)
    std::string trace_file;
    std::string sdfg_format = "json"; ///< Serialization of the SDFG/cutout payloads
    uint64_t capture_count = 0;
    uint64_t cutout_count = 0;
};

/// One captured argument buffer (mirrors an entry of an arg-capture index.json
/// plus its location inside the pack's blob store).
struct CaptureRecord {
    size_t element_id = 0;
    int invocation = 0;
    std::string target;
    int arg_idx = 0;
    bool after = false; ///< false = input ("in"), true = output ("out")
    std::vector<int64_t> dims;
    int primitive_type = 0;
    int format = 0;
    std::string ext_file; ///< Original capture filename (for faithful unpack)
    uint64_t offset = 0; ///< Offset into the decompressed blob store
    uint64_t size = 0; ///< Byte length
    uint64_t checksum = 0; ///< FNV-1a/64 of the bytes (integrity, not security)
};

/// A per-region cutout SDFG and its location inside the pack's cutout store.
struct CutoutRecord {
    std::string region_key;
    std::string display_key;
    std::string element_type;
    size_t element_id = 0;
    std::string format = "json"; ///< Serialization of this cutout's bytes
    uint64_t offset = 0; ///< Offset into the decompressed cutout store
    uint64_t size = 0;
    uint64_t checksum = 0;
};

/// Input form of a cutout supplied to write_pack (bytes not yet placed in a store).
struct CutoutInput {
    std::string region_key;
    std::string display_key;
    std::string element_type;
    size_t element_id = 0;
    std::string format = "json";
    std::vector<uint8_t> bytes;
};

/// In-memory representation of a parsed pack.
struct Pack {
    PackMeta meta;
    std::vector<uint8_t> sdfg_bytes; ///< SDFG chunk (format in meta.sdfg_format)
    std::string region_model_json; ///< RGNS payload (verbatim)
    std::vector<CaptureRecord> captures; ///< CAPS manifest
    std::vector<uint8_t> blob_store; ///< BLBS payload; index via offset/size
    std::vector<CutoutRecord> cutouts; ///< CUTS manifest
    std::vector<uint8_t> cutout_store; ///< CUTB payload; index via offset/size
};

/**
 * @brief Build and write a `.dcovpack` from an (already trace-annotated) module,
 *        the embedded SDFG bytes, and its arg-capture directory.
 *
 * Reads `*.index.json` + referenced `.bin` files from @p arg_capture_dir to
 * populate the CAPS manifest and BLBS blob store. If @p arg_capture_dir is empty
 * or missing, the pack is written with no captures. @p sdfg_bytes is embedded
 * verbatim (tagged with @p meta.sdfg_format) and @p cutouts are stored in CUTB.
 *
 * @throws std::runtime_error on I/O or compression failure.
 */
void write_pack(
    const Module& module,
    const std::vector<uint8_t>& sdfg_bytes,
    const std::vector<CutoutInput>& cutouts,
    const std::filesystem::path& arg_capture_dir,
    const PackMeta& meta,
    const std::filesystem::path& out_path
);

/**
 * @brief Read and decompress a `.dcovpack`.
 * @throws std::runtime_error if the file is missing, malformed, or a bad version.
 */
Pack read_pack(const std::filesystem::path& in_path);

} // namespace dcov
} // namespace sdfg
