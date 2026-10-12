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

    symbolic::MultiExpression get_shape(MmaFragmentType frag) const;

    static MmaBlockSize parse_block_size(const std::string& block_size_str);
};

std::ostream& operator<<(std::ostream& os, const MmaBlockSize& block_size);

/**
 * Grouping of MMA blocks.
 */
struct BlockCounts {
    int blocks_m;
    int blocks_n;

    int total_blocks() const {
        return blocks_m * blocks_n;
    }
};

struct GpuMmaTiling {
    /// The MMA block size (m, n, k) in elements. While every underlying storage will only use 2 dimensions of this,
    /// this qualifies all the possible operations on it
    MmaBlockSize mma_block_size;
    /// How many threads will work together on 1 MMA block
    int threads_per_mma_wavefront = 0;
    /// How many MMA blocks will be grouped together in a single wavefront / warp. MMA BlockSize * wave_tiling !=
    /// macro_tiling
    BlockCounts wave_tiling;
    /// How many MMA blocks will be grouped together in a macro block / on a SM or CU
    /// This controls how much data one block needs to handle and will typically need to be stored in block-shared
    /// memory
    BlockCounts macro_tiling;
    /// The type used by the MMA hardware to accumulate the results of the MMA operation. This may be different from the
    /// input and output types, e.g. for FP16 inputs and FP32 outputs. For a MatmulNode this would not be externally
    /// visible other in the precision of the results
    types::PrimitiveType acc_type;

    /// implementation-specific details, e.g. for ROCm MFMA variant selection
    int impl_details = 0;

    /**
     * Check that we can use the given wave tiling for the given macro block tiling, i.e. that the wave tiling divides
     * the macro block tiling and that the number of threads per wave does not exceed 1024.
     * @param wave_tiling
     * @param macro_block
     * @return
     */
    static bool wave_tile_divides(const BlockCounts& wave_tiling, const BlockCounts& macro_block);

    int get_mma_blocks_m_per_wavefront() const;
    int get_mma_blocks_n_per_wavefront() const;
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
    virtual ~GpuMmaSupport() = default;

    virtual types::PrimitiveType get_accumulator_type(
        types::PrimitiveType input_type, types::PrimitiveType output_type, types::PrimitiveType desired_acc_type
    ) const = 0;
    types::PrimitiveType get_accumulator_type(types::PrimitiveType input_type, types::PrimitiveType output_type) const {
        return get_accumulator_type(input_type, output_type, types::PrimitiveType::Void);
    }

    GpuMmaSupport() = default;

    virtual bool is_valid_block_size(
        const MmaBlockSize& block_size, types::PrimitiveType input_type, types::PrimitiveType acc_type
    ) const = 0;

    virtual data_flow::ImplementationType get_mma_impl_type(const GpuMmaTiling& tiling) const = 0;

    static int get_integer_block_count(const symbolic::Expression& size, uint16_t block_size);

    virtual bool supported_types(types::PrimitiveType input_type, types::PrimitiveType output_type) const = 0;

    virtual std::optional<GpuMmaTiling> get_mma_tiling(
        const symbolic::MultiExpression& res_shape,
        types::PrimitiveType input_type,
        types::PrimitiveType acc_type,
        const MmaBlockSize* block_size_hint,
        const BlockCounts* wave_tile_hint
    ) const = 0;

    virtual void set_mma_fragment_storage_type(
        types::StorageType& storage_type, const GpuMmaTiling& tiling, MmaFragmentType type, MmaFragmentLayout layout
    ) const = 0;

    virtual std::vector<MmaBlockSize>
    get_supported_block_sizes(types::PrimitiveType input_type, types::PrimitiveType acc_type) const = 0;

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

    [[nodiscard]] std::optional<math::tensor::TensorLayout>
    to_tensor_layout(const MmaBlockSize& block_size, MmaFragmentType frag) const;

    static std::optional<GpuMmaFromMemoryLayout> from_tensor_layout(const math::tensor::TensorLayout& layout);
};

std::ostream& operator<<(std::ostream& os, const GpuMmaFromMemoryLayout& layout);

// JSON (de)serialization helpers shared by the MMA library-node serializers.
void serialize_mma_block_size(nlohmann::json& j, const MmaBlockSize& block_size);
MmaBlockSize deserialize_mma_block_size(const nlohmann::json& j);

void serialize_mma_from_memory_layout(nlohmann::json& j, const GpuMmaFromMemoryLayout& layout);
GpuMmaFromMemoryLayout deserialize_mma_from_memory_layout(const nlohmann::json& j);

} // namespace sdfg::gpu
