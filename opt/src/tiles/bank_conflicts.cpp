#include "sdfg/tiles/bank_conflicts.h"

#include <algorithm>
#include <map>
#include <set>
#include <tuple>

namespace sdfg {
namespace tiles {

namespace {

constexpr size_t kMaxAccessBytes = 16;

int64_t element_offset(const std::vector<int64_t>& coord, const std::vector<int64_t>& strides) {
    int64_t off = 0;
    for (size_t d = 0; d < coord.size() && d < strides.size(); ++d) {
        off += coord[d] * strides[d];
    }
    return off;
}

// Cycles of one access in which every lane reads @p width bytes at its byte address.
size_t instruction_cycles(const BankModel& model, const std::vector<int64_t>& addresses, size_t width) {
    const size_t bank_bytes = model.bank_bytes;
    const size_t words = std::max<size_t>(1, width / bank_bytes);
    const size_t phase = std::max<size_t>(1, model.banks * bank_bytes / std::max(width, bank_bytes));
    size_t cycles = 0;
    for (size_t first = 0; first < addresses.size(); first += phase) {
        std::map<int64_t, std::set<int64_t>> bank_words;
        for (size_t l = first; l < std::min(addresses.size(), first + phase); ++l) {
            const int64_t word = addresses[l] / static_cast<int64_t>(bank_bytes);
            for (size_t w = 0; w < words; ++w) {
                const int64_t bank = ((word + w) % model.banks + model.banks) % model.banks;
                bank_words[bank].insert(word + w);
            }
        }
        size_t degree = 1;
        for (const auto& [bank, served] : bank_words) {
            degree = std::max(degree, served.size());
        }
        cycles += degree;
    }
    return cycles;
}

} // namespace

size_t access_cycles(
    const BankModel& model,
    size_t elem_bytes,
    const std::vector<int64_t>& strides,
    const data_flow::SubgroupAccess& access
) {
    std::vector<std::vector<int64_t>> offsets;
    size_t per_lane = 0;
    for (const auto& lane : access.lanes) {
        std::vector<int64_t> lane_offsets;
        for (const auto& coord : lane) {
            lane_offsets.push_back(element_offset(coord, strides));
        }
        per_lane = std::max(per_lane, lane_offsets.size());
        offsets.push_back(std::move(lane_offsets));
    }
    if (per_lane == 0) {
        return 0;
    }

    // One vector access per lane when every lane's elements are adjacent and aligned.
    const size_t width = per_lane * elem_bytes;
    bool vector = width <= kMaxAccessBytes && (width & (width - 1)) == 0;
    for (const auto& lane : offsets) {
        if (!vector) {
            break;
        }
        vector = lane.size() == per_lane && (lane.front() * static_cast<int64_t>(elem_bytes)) % width == 0;
        for (size_t e = 1; vector && e < lane.size(); ++e) {
            vector = lane[e] == lane.front() + static_cast<int64_t>(e);
        }
    }
    if (vector) {
        std::vector<int64_t> addresses;
        for (const auto& lane : offsets) {
            addresses.push_back(lane.front() * static_cast<int64_t>(elem_bytes));
        }
        return instruction_cycles(model, addresses, width);
    }

    size_t cycles = 0;
    for (size_t e = 0; e < per_lane; ++e) {
        std::vector<int64_t> addresses;
        for (const auto& lane : offsets) {
            if (e < lane.size()) {
                addresses.push_back(lane[e] * static_cast<int64_t>(elem_bytes));
            }
        }
        cycles += instruction_cycles(model, addresses, elem_bytes);
    }
    return cycles;
}

size_t slot_pad(const BankModel& model, size_t block, size_t coop_span) {
    const size_t banks = model.banks;
    if (coop_span > 0) {
        const size_t target = coop_span % banks;
        const size_t add = (target + banks - block % banks) % banks;
        const size_t padded = (block + add + 3) & ~size_t{3};
        return padded - block;
    }
    return block % 2 == 0 ? 1 : 0;
}

size_t row_pad(
    const BankModel& model,
    size_t elem_bytes,
    PackedBuffer buffer,
    size_t align_bytes,
    const std::function<std::optional<size_t>(const PackedBuffer&)>& cost
) {
    buffer.conflict = ConflictPolicy::PadRow;
    const size_t candidates = std::max<size_t>(1, model.banks * model.bank_bytes / std::max<size_t>(1, elem_bytes));
    std::optional<std::tuple<size_t, bool, size_t>> best;
    for (size_t pad = 0; pad < candidates; ++pad) {
        buffer.pad = pad;
        auto cycles = cost(buffer);
        if (!cycles) {
            continue;
        }
        bool aligned = false;
        auto axes = buffer.axes();
        if (!axes.empty() && SymEngine::is_a<SymEngine::Integer>(*axes.back())) {
            const auto row = SymEngine::rcp_static_cast<const SymEngine::Integer>(axes.back())->as_int();
            aligned = align_bytes == 0 || (static_cast<size_t>(row) * elem_bytes) % align_bytes == 0;
        }
        std::tuple<size_t, bool, size_t> key{*cycles, !aligned, pad};
        if (!best || key < *best) {
            best = key;
        }
    }
    return best ? std::get<2>(*best) : 0;
}

} // namespace tiles
} // namespace sdfg
