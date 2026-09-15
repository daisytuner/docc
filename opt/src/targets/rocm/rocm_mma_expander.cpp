#include "sdfg/targets/rocm/rocm_mma_expander.h"
#include "sdfg/targets/gpu/gpu_offload_schedule_type.h"
#include "sdfg/targets/rocm/rocm.h"
#include "sdfg/targets/rocm/rocm_mma_dispatcher.h"

namespace sdfg::gpu::rocm {

namespace {

std::optional<data_flow::ImplementationType> resolve_mma_impl_type(const RocmArch& arch) {
    auto& arch_name = arch.name();
    if (arch_name == "gfx1201") {
        return ImplementationType_ROCM_MMA_GFX1201;
    } else if (arch_name == "gfx90a") {
        return ImplementationType_ROCM_MMA_GFX90A;
    }
    return std::nullopt;
}

} // namespace

bool RocmMmaExpander::matches_possible_mma_pattern(const math::tensor::MatMulNode& node) const {
    if (!resolve_mma_impl_type(arch_)) {
        return false;
    }

    auto* mma_arch = arch_.mma_support();

    if (!mma_arch) {
        return false;
    }

    // basic sanity checks
    auto& dims_a = node.layout_a();
    auto& dims_b = node.layout_b();
    if (dims_a.dims() != 2 || dims_b.dims() != 2) {
        return false;
    }

    if (!symbolic::eq(dims_a.get_dim(1), dims_b.get_dim(0))) {
        // K dimension must match
        return false;
    }

    auto layout_a = dims_a.is_2d_col_or_row_major();
    auto layout_b = dims_b.is_2d_col_or_row_major();
    if (layout_a == math::tensor::TensorLayout::LAYOUT_OTHER || layout_b == math::tensor::TensorLayout::LAYOUT_OTHER ||
        node.layout_y().is_2d_col_or_row_major() == math::tensor::TensorLayout::LAYOUT_OTHER) {
        // only support row-major or col-major layout
        return false;
    }

    auto& m = dims_a.get_dim(0);
    auto& n = dims_b.get_dim(1);
    auto& k = dims_a.get_dim(1);

    auto m_blocks = GpuMmaSupport::get_integer_block_count(m, mma_arch->mma_block_m);
    auto n_blocks = GpuMmaSupport::get_integer_block_count(n, mma_arch->mma_block_n);
    auto k_blocks = GpuMmaSupport::get_integer_block_count(k, mma_arch->mma_block_k);

    if (!m_blocks || !n_blocks || !k_blocks) {
        return false;
    }

    if (!mma_arch->valid_block_counts(mma_arch->mma_block_m, m_blocks, n_blocks, k_blocks)) {
        return false;
    }

    auto input_type = node.uniform_quantization(node.get_parent());
    auto output_type = input_type;

    if (!input_type || !output_type || input_type.value() == types::PrimitiveType::Void ||
        output_type.value() == types::PrimitiveType::Void) {
        return false;
    }

    return mma_arch->supported_types(input_type.value(), output_type.value());
}

GpuMmaTiling RocmMmaExpander::get_mma_tiling(const symbolic::MultiExpression& res_shape) const {
    auto* mma_arch = arch_.mma_support();
    if (!mma_arch) {
        throw std::runtime_error("No MMA architecture available for this GPU target.");
    }

    return RocmMmaMatmulDispatcher::get_mma_tiling(mma_arch, res_shape);
}

ScheduleType RocmMmaExpander::get_schedule_type(gpu::TargetLevel dim, const symbolic::Integer& size) const {
    return gpu::ScheduleType_GPU_Offload::create<sdfg::rocm::ScheduleType_ROCM_Offload>(dim, size);
}

void RocmMmaExpander::set_implementation_type_mma(math::tensor::MatMulNode& node, const GpuMmaTiling& mma_tiling) const {
    auto impl_type = resolve_mma_impl_type(arch_);
    if (!impl_type) {
        throw std::runtime_error("Unsupported ROCm architecture: " + arch_.name());
    }
    node.set_implementation_type(*impl_type);
}

} // namespace sdfg::gpu::rocm
