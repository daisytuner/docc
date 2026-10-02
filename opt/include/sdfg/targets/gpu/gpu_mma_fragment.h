#pragma once

#include <nlohmann/json_fwd.hpp>
#include <ostream>
#include <string>
#include "sdfg/data_flow/library_node.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/type.h"

namespace sdfg::gpu {

class GpuArch;

enum class MmaFragmentType { A = 1, B = 2, C = 3 };

struct MmaBlockSize {
    int m;
    int n;
    int k;

    std::string toStr() const;

    std::string dim_str(MmaFragmentType for_type) const;
};

std::ostream& operator<<(std::ostream& os, const MmaBlockSize& block_size);

struct GpuMmaTiling {
    MmaBlockSize mma_block_size;
    int wave_tile_blocks_m = 0;
    int wave_tile_blocks_n = 0;
    int threads_per_mma_block_m = 0;
    int macro_blocks_m = 0;
    int macro_blocks_n = 0;
};

enum MmaFragmentLayout {
    MMA_LAYOUT_UNSPECIFIED = 0,
    MMA_LAYOUT_ROW_MAJOR = 1,
    MMA_LAYOUT_COL_MAJOR = 2,
};

constexpr const char* mma_fragment_layout_to_string(MmaFragmentLayout layout) {
    switch (layout) {
        case MmaFragmentLayout::MMA_LAYOUT_UNSPECIFIED:
            return "*";
        case MmaFragmentLayout::MMA_LAYOUT_ROW_MAJOR:
            return "row_major";
        case MmaFragmentLayout::MMA_LAYOUT_COL_MAJOR:
            return "col_major";
        default:
            throw std::invalid_argument("invalid MMA fragment layout");
    }
}

struct GpuMmaSupport {
    MmaBlockSize mma_block_size;
    const uint16_t threads_per_mma_block;

    virtual ~GpuMmaSupport() = default;
    GpuMmaSupport(uint16_t block_m, uint16_t block_n, uint16_t block_k, uint16_t threads_per_mma_block)
        : mma_block_size{block_m, block_n, block_k}, threads_per_mma_block(threads_per_mma_block) {
    }
    virtual bool valid_block_counts(uint16_t block_base, int m_blocks, int n_blocks, int k_blocks) const = 0;

    [[deprecated("only for Tensor Matmul. Use get_mma_impl_type() for everything new")]]
    virtual std::optional<
        data_flow::ImplementationType> get_matmul_impl_type(const GpuArch& arch, const GpuMmaTiling& tiling) const = 0;

    virtual data_flow::ImplementationType get_mma_impl_type() const = 0;

    static int get_integer_block_count(const symbolic::Expression& size, uint16_t block_size);

    virtual bool supported_types(types::PrimitiveType input_type, types::PrimitiveType output_type) const = 0;

    virtual GpuMmaTiling get_mma_tiling(const symbolic::MultiExpression& res_shape) const = 0;

    virtual void set_mma_fragment_storage_type(
        types::StorageType& storage_type, const MmaBlockSize& size, MmaFragmentType type, MmaFragmentLayout layout
    ) const = 0;

protected:
    static int get_storage_type_arg_as_int(const types::StorageType& storage, int idx);
};

/**
 * @brief Describes how a single MMA fragment maps to memory relative to a base pointer.
 *
 * Shared by the fragment load/store nodes. The tile it describes must follow GEMM semantics: one
 * customizable leading-dimension stride, the other dimension contiguous (row- or column-major).
 */
struct GpuMmaFromMemoryLayout {
    symbolic::Expression offset; ///< element offset added to the base pointer
    symbolic::Expression ldstride; ///< leading-dimension stride
    MmaFragmentLayout layout; ///< memory layout of the tile

    std::string toStr() const;
    void collect_symbols(symbolic::SymbolSet& syms) const;
    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression);
    void replace(const symbolic::ExpressionMapping& replacements);
};

std::ostream& operator<<(std::ostream& os, const GpuMmaFromMemoryLayout& layout);

// JSON (de)serialization helpers shared by the MMA library-node serializers.
void serialize_mma_block_size(nlohmann::json& j, const MmaBlockSize& block_size);
MmaBlockSize deserialize_mma_block_size(const nlohmann::json& j);

void serialize_mma_from_memory_layout(nlohmann::json& j, const GpuMmaFromMemoryLayout& layout);
GpuMmaFromMemoryLayout deserialize_mma_from_memory_layout(const nlohmann::json& j);

} // namespace sdfg::gpu
