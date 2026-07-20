#include "sdfg/passes/zero_fill_to_memset_pass.h"

#include <cctype>
#include <memory>
#include <string>
#include <vector>

#include "sdfg/analysis/memory_layout_analysis.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_nodes/stdlib/memset.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/utils.h"

namespace sdfg::passes {

// Returns true if the literal string represents a numeric zero (e.g. "0", "0.0", "-0.0f", "false").
static bool is_zero_literal(const std::string& value) {
    std::string t;
    for (char c : value) {
        if (!std::isspace(static_cast<unsigned char>(c))) {
            t += c;
        }
    }
    if (t.empty()) {
        return false;
    }
    if (t == "false") {
        return true;
    }

    size_t begin = 0;
    if (t[begin] == '+' || t[begin] == '-') {
        begin++;
    }
    size_t end = t.size();
    while (end > begin && (t[end - 1] == 'f' || t[end - 1] == 'F' || t[end - 1] == 'l' || t[end - 1] == 'L')) {
        end--;
    }

    bool saw_digit = false;
    for (size_t k = begin; k < end; k++) {
        char c = t[k];
        if (c == '0') {
            saw_digit = true;
        } else if (c == '.' || c == 'x' || c == 'X') {
            continue;
        } else {
            return false;
        }
    }
    return saw_digit;
}

// Decides whether the tile of accessed elements covers the entire container contiguously
// starting at offset 0, so that the loop nest is equivalent to a single memset.
//
// `bounds` are the exclusive upper bounds of the loop nest (outermost first); their product
// is the number of written elements. The check succeeds when
//   - every dimension starts at index 0,
//   - every dimension with a known (bounded) extent is fully covered (extent == declared
//     shape), which rejects partial fills of statically sized arrays, and
//   - the accessed elements form a dense, contiguous block [0, count) where count equals
//     the number of loop iterations, which rejects strided / gapped accesses.
static bool covers_full_container(const analysis::MemoryTile& tile, const std::vector<symbolic::Expression>& bounds) {
    size_t ndims = tile.min_subset.size();
    if (ndims == 0) {
        return false;
    }
    const auto& shape = tile.layout.shape();
    if (shape.size() != ndims) {
        return false;
    }

    auto extents = tile.extents();
    if (extents.size() != ndims) {
        return false;
    }

    for (size_t d = 0; d < ndims; d++) {
        if (!symbolic::eq(tile.min_subset[d], symbolic::zero())) {
            return false;
        }
        if (extents[d].is_null()) {
            return false;
        }
        // Only the leading dimension can be unbounded (raw pointer base); its extent is
        // not encoded in the type and is therefore trusted to match the loop. Every other
        // dimension must fully cover its declared extent.
        bool bounded = (d != 0) || tile.first_dim_bounded;
        if (bounded && !symbolic::eq(extents[d], shape[d])) {
            return false;
        }
    }

    // The written elements must form a contiguous block [0, trip) with no gaps.
    auto [first, last] = tile.contiguous_range();
    if (first.is_null() || last.is_null()) {
        return false;
    }
    if (!symbolic::eq(first, symbolic::zero())) {
        return false;
    }

    symbolic::Expression trip = symbolic::one();
    for (const auto& bound : bounds) {
        trip = symbolic::mul(trip, bound);
    }
    return symbolic::eq(symbolic::add(last, symbolic::one()), trip);
}

ZeroFillToMemsetVisitor::ZeroFillToMemsetVisitor(
    builder::StructuredSDFGBuilder& builder,
    ZeroFillToMemsetPass::State& state,
    analysis::MemoryLayoutAnalysis& memory_layout_analysis
)
    : builder_(builder), state_(state), memory_layout_analysis_(memory_layout_analysis) {}

bool ZeroFillToMemsetVisitor::match(structured_control_flow::Map& node, Candidate& candidate) {
    // Walk the perfect nest of Maps, collecting the exclusive upper bounds.
    std::vector<symbolic::Expression> bounds;

    structured_control_flow::StructuredLoop* loop = &node;
    structured_control_flow::Block* terminal = nullptr;
    while (true) {
        // Each loop must be a normalized [0, bound) unit-stride loop.
        if (!symbolic::eq(loop->init(), symbolic::zero()) || !loop->is_contiguous()) {
            return false;
        }
        auto bound = loop->canonical_bound_upper();
        if (bound.is_null()) {
            return false;
        }
        bounds.push_back(bound);

        auto& body = loop->root();
        // A perfect nest: the body contains exactly the next nesting level.
        if (body.size() != 1) {
            return false;
        }
        auto entry = body.at(0);
        if (!entry.second.assignments().empty()) {
            return false;
        }

        auto& child = entry.first;
        if (auto* inner_map = dynamic_cast<structured_control_flow::Map*>(&child)) {
            loop = inner_map;
            continue;
        }
        if (auto* block = dynamic_cast<structured_control_flow::Block*>(&child)) {
            terminal = block;
            break;
        }
        return false;
    }

    // The innermost body must be a single "A[...] = 0" assignment and nothing else.
    auto& dfg = terminal->dataflow();
    if (dfg.tasklets().size() != 1 || !dfg.library_nodes().empty() || dfg.data_nodes().size() != 2) {
        return false;
    }

    auto* tasklet = *dfg.tasklets().begin();
    if (tasklet->code() != data_flow::TaskletCode::assign || tasklet->inputs().size() != 1) {
        return false;
    }

    const data_flow::Memlet* in_edge = nullptr;
    for (auto& edge : dfg.in_edges(*tasklet)) {
        if (in_edge != nullptr) {
            return false;
        }
        in_edge = &edge;
    }
    const data_flow::Memlet* out_edge = nullptr;
    for (auto& edge : dfg.out_edges(*tasklet)) {
        if (out_edge != nullptr) {
            return false;
        }
        out_edge = &edge;
    }
    if (in_edge == nullptr || out_edge == nullptr) {
        return false;
    }

    // Input: the constant literal 0.
    auto* constant = dynamic_cast<const data_flow::ConstantNode*>(&in_edge->src());
    if (constant == nullptr || !is_zero_literal(constant->data())) {
        return false;
    }

    // Output: a plain array element write.
    auto* access = dynamic_cast<const data_flow::AccessNode*>(&out_edge->dst());
    if (access == nullptr || dynamic_cast<const data_flow::ConstantNode*>(access) != nullptr) {
        return false;
    }
    const std::string& container = access->data();

    // Coverage is decided by the memory-layout analysis: the tile for this container at the
    // outermost Map must cover the whole array contiguously. This handles native arrays,
    // pointer bases and linearized accesses (e.g. `A[i*M + j]`) uniformly.
    const analysis::MemoryTile* tile = memory_layout_analysis_.tile(node, container);
    if (tile == nullptr || !covers_full_container(*tile, bounds)) {
        return false;
    }

    // The container's element must be a scalar so a byte-wise memset is well-defined.
    const auto& container_type = builder_.subject().type(container);
    const auto& element_type = types::peel_to_innermost_element(container_type);
    const auto* scalar = dynamic_cast<const types::Scalar*>(&element_type);
    if (scalar == nullptr) {
        return false;
    }
    auto element_size = types::get_contiguous_element_size(container_type);
    if (element_size.is_null()) {
        return false;
    }

    // Total size in bytes = element size times the number of written elements.
    symbolic::Expression num = element_size;
    for (const auto& bound : bounds) {
        num = symbolic::mul(num, bound);
    }

    candidate.map = &node;
    candidate.array = container;
    candidate.num = num;
    candidate.ptr_type = std::make_unique<types::Pointer>(*scalar);
    return true;
}

bool ZeroFillToMemsetVisitor::visit(structured_control_flow::Map& node) {
    Candidate candidate;
    if (match(node, candidate)) {
        candidates_.push_back(std::move(candidate));
        // Do not descend into a matched nest.
        return false;
    }
    return ActualStructuredSDFGVisitor::visit(node);
}

void ZeroFillToMemsetVisitor::apply() {
    for (auto& candidate : candidates_) {
        auto* parent = dynamic_cast<structured_control_flow::Sequence*>(candidate.map->get_parent());
        if (parent == nullptr) {
            continue;
        }
        int index = parent->index(*candidate.map);
        if (index < 0) {
            continue;
        }

        // Preserve any symbol assignments attached to the replaced nest.
        auto assignments = parent->at(index).second.assignments();

        auto& block = builder_.add_block_before(*parent, *candidate.map, assignments);
        stdlib::add_memset_node(builder_, block, candidate.array, symbolic::integer(0), candidate.num, *candidate.ptr_type);

        int map_index = parent->index(*candidate.map);
        if (map_index < 0) {
            continue;
        }
        builder_.remove_child(*parent, map_index);
        state_.applied++;
    }
}

bool ZeroFillToMemsetPass::run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    State state;

    auto& memory_layout_analysis = analysis_manager.get<analysis::MemoryLayoutAnalysis>();

    ZeroFillToMemsetVisitor visitor(builder, state, memory_layout_analysis);
    visitor.dispatch(builder.subject().root());
    visitor.apply();

    return state.applied > 0;
}

} // namespace sdfg::passes
