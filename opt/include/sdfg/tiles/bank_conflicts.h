#pragma once

#include <cstddef>
#include <functional>
#include <optional>
#include <vector>

#include "sdfg/data_flow/library_node.h"
#include "sdfg/tiles/buffer_layout.h"

namespace sdfg {
namespace tiles {

/// A banked on-chip memory: @ref banks banks of @ref bank_bytes each, every bank serving one
/// word per cycle.
struct BankModel {
    unsigned banks = 32;
    unsigned bank_bytes = 4;
};

/// Cycles the banked memory spends serving @p access when the operand's elements (of
/// @p elem_bytes) sit at element offset `sum(coord[d] * strides[d])`. A lane whose elements are
/// adjacent and aligned issues one vector access of at most 16 B, otherwise one access per
/// element. Lanes are served in phases that fill every bank once; a phase costs the largest
/// number of distinct words any bank serves (lanes reading the same word share it).
size_t access_cycles(
    const BankModel& model,
    size_t elem_bytes,
    const std::vector<int64_t>& strides,
    const data_flow::SubgroupAccess& access
);

/// @ref ConflictPolicy::PadSlot padding (elements) of a per-slot block of @p block elements
/// whose slots are accessed by @p coop_span consecutive lanes of a subgroup: makes the slot
/// stride congruent to @p coop_span modulo the bank count, rounded up to a multiple of 4
/// elements so blocks stay vector-addressable. Without a span, the next odd stride.
size_t slot_pad(const BankModel& model, size_t block, size_t coop_span);

/// The @ref ConflictPolicy::PadRow padding (elements) of @p buffer minimizing @p cost, the
/// access cycles of a candidate (nullopt rejects it). Ties prefer stored rows that stay
/// @p align_bytes aligned, then less padding. Candidates span one full cycle of the banks.
size_t row_pad(
    const BankModel& model,
    size_t elem_bytes,
    PackedBuffer buffer,
    size_t align_bytes,
    const std::function<std::optional<size_t>(const PackedBuffer&)>& cost
);

} // namespace tiles
} // namespace sdfg
