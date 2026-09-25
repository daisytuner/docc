#include "sdfg/targets/gpu/gpu_reduce_layout.h"

#include <algorithm>
#include <limits>
#include <sdfg/exceptions.h>
#include <utility>

namespace sdfg {
namespace gpu {

ReductionLayout::ReductionLayout(symbolic::Expression origin, std::vector<Dimension> axes)
    : base(origin), dimensions(std::move(axes)) {
    if (base.is_null()) {
        throw InvalidSDFGException("GPU reduction layout requires an origin");
    }
    std::sort(dimensions.begin(), dimensions.end(), [](const auto& left, const auto& right) {
        return left.stride < right.stride;
    });
    // Each non-unit axis must start beyond the address span of all smaller-stride axes.
    int64_t span = 1;
    for (const auto& dimension : dimensions) {
        if (dimension.count <= 0 || dimension.stride <= 0 ||
            dimension.count > std::numeric_limits<int64_t>::max() / extent ||
            dimension.count - 1 > (std::numeric_limits<int64_t>::max() - span) / dimension.stride) {
            throw InvalidSDFGException("GPU reduction layout has invalid or overflowing dimensions");
        }
        if (dimension.count > 1 && dimension.stride < span) {
            throw InvalidSDFGException("GPU reduction layout has overlapping output dimensions");
        }
        extent *= dimension.count;
        span += dimension.stride * (dimension.count - 1);
    }
    std::erase_if(dimensions, [](const auto& dimension) {
        return dimension.count == 1;
    });
}

// Decode dense mixed-radix slots back into the original strided address space.
symbolic::Expression ReductionLayout::unpack(symbolic::Expression slot) const {
    auto index = base;
    for (const auto& dimension : dimensions) {
        auto quotient = symbolic::div(slot, symbolic::integer(dimension.count));
        auto coordinate = symbolic::sub(slot, symbolic::mul(quotient, symbolic::integer(dimension.count)));
        index = symbolic::add(index, symbolic::mul(coordinate, symbolic::integer(dimension.stride)));
        slot = quotient;
    }
    return symbolic::expand(index);
}

// Extract coordinates from largest source stride to smallest, discarding gaps between axes.
symbolic::Expression ReductionLayout::pack(symbolic::Expression index) const {
    auto remaining = symbolic::expand(symbolic::sub(index, base));
    symbolic::Expression slot = symbolic::zero();
    for (auto dimension = dimensions.rbegin(); dimension != dimensions.rend(); ++dimension) {
        auto coordinate = symbolic::div(remaining, symbolic::integer(dimension->stride));
        slot = symbolic::add(symbolic::mul(slot, symbolic::integer(dimension->count)), coordinate);
        remaining = symbolic::sub(remaining, symbolic::mul(coordinate, symbolic::integer(dimension->stride)));
    }
    return symbolic::expand(slot);
}

} // namespace gpu
} // namespace sdfg
