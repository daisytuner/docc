#include "sdfg/targets/gpu/gpu_mma.h"

namespace sdfg::gpu {

int GpuMmaSupport::get_integer_block_count(const symbolic::Expression& size, uint16_t block_size) {
    auto blocks = symbolic::simplify(SymEngine::div(size, symbolic::integer(block_size)));
    if (SymEngine::is_a<SymEngine::Integer>(*blocks)) {
        auto i = SymEngine::rcp_static_cast<const SymEngine::Integer>(blocks);
        if (i->as_int() > 0) {
            return static_cast<int>(i->as_int());
        }
    }
    return 0;
}

} // namespace sdfg::gpu
