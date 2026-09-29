#include "sdfg/tiles/transformations/tile_vectorizer.h"

#include <functional>
#include <unordered_map>
#include <vector>

#include "sdfg/data_flow/tasklet.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/tiles/library_nodes/pipeline_node.h"
#include "sdfg/tiles/library_nodes/tile_copy_node.h"
#include "sdfg/tiles/tile.h"
#include "sdfg/tiles/tile_target_registry.h"

namespace sdfg {
namespace transformations {

namespace {

using structured_control_flow::Block;
using structured_control_flow::ControlFlowNode;

// The whole-copy TileCopyNode in @p block, or nullptr.
tiles::TileCopyNode* tile_copy_node_in(Block& block) {
    for (auto& node : block.dataflow().nodes()) {
        if (auto* n = dynamic_cast<tiles::TileCopyNode*>(&node)) {
            return n;
        }
    }
    return nullptr;
}

// Widen a scalar/async TileCopyNode to the widest legal vector transfer, deriving
// contiguity/alignment from the plan layouts: the inner (fastest) tile dim must be
// unit-stride on both sides, the inner extent and total must be a multiple of the
// coalescing factor, and the outer source strides must keep each inner run
// factor-aligned. A ScalarSync copy becomes VectorSync; a (pipelined) CpAsync copy
// keeps its atom and only widens the transfer. Mutates atom/bytes; true if widened.
bool widen_tile_copy_node(tiles::TileCopyNode& node) {
    if (node.atom() != tiles::CopyAtom::ScalarSync && node.atom() != tiles::CopyAtom::CpAsync) {
        return false;
    }
    const auto& plan = node.plan();
    if (plan.src.dims() == 0) {
        return false;
    }
    // A swizzled buffer XORs the inner offset, so consecutive tile coords are not
    // consecutive in the buffer: a widened vector/async move would scatter.
    if (!plan.dst_swizzle.is_identity()) {
        return false;
    }
    if (!symbolic::eq(plan.src.strides().back(), symbolic::integer(1)) ||
        !symbolic::eq(plan.dst.strides().back(), symbolic::integer(1))) {
        return false;
    }
    auto* inner_i = dynamic_cast<const SymEngine::Integer*>(plan.src.shape().back().get());
    auto total = plan.src.total_elements();
    auto* total_i = dynamic_cast<const SymEngine::Integer*>(total.get());
    if (inner_i == nullptr || total_i == nullptr) {
        return false;
    }
    const long long inner_n = inner_i->as_int();
    const long long total_n = total_i->as_int();
    const size_t elem_bytes = node.bytes();
    if (elem_bytes == 0) {
        return false;
    }
    for (size_t width : {size_t{16}, size_t{8}}) {
        if (width <= elem_bytes || width % elem_bytes != 0) {
            continue;
        }
        const long long factor = static_cast<long long>(width / elem_bytes);
        if (inner_n % factor != 0 || total_n % factor != 0) {
            continue;
        }
        bool outer_ok = true;
        for (size_t d = 0; d + 1 < plan.src.strides().size(); ++d) {
            auto* s = dynamic_cast<const SymEngine::Integer*>(plan.src.strides()[d].get());
            if (s == nullptr || s->as_int() % factor != 0) {
                outer_ok = false;
                break;
            }
        }
        if (!outer_ok) {
            continue;
        }
        node.set_bytes(width);
        if (node.atom() == tiles::CopyAtom::ScalarSync) {
            node.set_atom(tiles::CopyAtom::VectorSync);
        }
        return true;
    }
    return false;
}

// Does @p block hold a single scalar `assign` copy (one in, one out, both access
// nodes)? That is the shape LocalStorage emits before any widening.
// The nearest enclosing sequential (non-Map) loop — the pipeline panel loop.
structured_control_flow::StructuredLoop* enclosing_panel_loop(ControlFlowNode& from) {
    ControlFlowNode* n = from.get_parent();
    while (n != nullptr) {
        if (auto* loop = dynamic_cast<structured_control_flow::StructuredLoop*>(n)) {
            if (dynamic_cast<structured_control_flow::Map*>(loop) == nullptr) {
                return loop;
            }
        }
        n = n->get_parent();
    }
    return nullptr;
}

// Sum the cp.async transfer words (bytes / 4) over @p scope's TileCopyNodes.
size_t sum_cp_async_words(ControlFlowNode& scope) {
    size_t words = 0;
    std::function<void(ControlFlowNode&)> walk = [&](ControlFlowNode& n) {
        if (auto* block = dynamic_cast<Block*>(&n)) {
            for (auto& node : block->dataflow().nodes()) {
                if (auto* tc = dynamic_cast<tiles::TileCopyNode*>(&node)) {
                    if (tc->atom() == tiles::CopyAtom::CpAsync) {
                        words += tc->bytes() / 4;
                    }
                }
            }
        } else if (auto* seq = dynamic_cast<structured_control_flow::Sequence*>(&n)) {
            for (size_t i = 0; i < seq->size(); i++) {
                walk(seq->at(i));
            }
        } else if (auto* map = dynamic_cast<structured_control_flow::Map*>(&n)) {
            walk(map->root());
        } else if (auto* loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&n)) {
            walk(loop->root());
        } else if (auto* ie = dynamic_cast<structured_control_flow::IfElse*>(&n)) {
            for (size_t i = 0; i < ie->size(); i++) {
                walk(ie->at(i).first);
            }
        }
    };
    walk(scope);
    return words;
}

} // namespace

TileVectorizer::TileVectorizer(structured_control_flow::StructuredLoop& loop) : loop_(loop) {
}

std::string TileVectorizer::name() const {
    return "TileVectorizer";
}

bool TileVectorizer::can_be_applied(builder::StructuredSDFGBuilder&, analysis::AnalysisManager&) {
    // Applicable when the subtree holds at least one widenable cooperative copy: a
    // TileCopyNode whose atom is still ScalarSync or (pipelined) CpAsync.
    bool found = false;
    std::function<void(ControlFlowNode&)> scan = [&](ControlFlowNode& n) {
        if (found) {
            return;
        }
        if (auto* block = dynamic_cast<Block*>(&n)) {
            if (auto* tc = tile_copy_node_in(*block)) {
                if (tc->atom() == tiles::CopyAtom::ScalarSync || tc->atom() == tiles::CopyAtom::CpAsync) {
                    found = true;
                    return;
                }
            }
        }
        if (auto* map = dynamic_cast<structured_control_flow::Map*>(&n)) {
            scan(map->root());
        } else if (auto* seq = dynamic_cast<structured_control_flow::Sequence*>(&n)) {
            for (size_t i = 0; i < seq->size(); i++) {
                scan(seq->at(i));
            }
        } else if (auto* loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&n)) {
            scan(loop->root());
        } else if (auto* ie = dynamic_cast<structured_control_flow::IfElse*>(&n)) {
            for (size_t i = 0; i < ie->size(); i++) {
                scan(ie->at(i).first);
            }
        }
    };
    scan(loop_.root());
    return found;
}

void TileVectorizer::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    // Collect the copy nodes first (widening mutates the graph), then widen each to
    // the widest legal vector/async transfer.
    std::vector<tiles::TileCopyNode*> tile_copies;
    std::function<void(ControlFlowNode&)> collect = [&](ControlFlowNode& n) {
        if (auto* block = dynamic_cast<Block*>(&n)) {
            if (auto* tc = tile_copy_node_in(*block)) {
                tile_copies.push_back(tc);
            }
        }
        if (auto* map = dynamic_cast<structured_control_flow::Map*>(&n)) {
            collect(map->root());
        } else if (auto* seq = dynamic_cast<structured_control_flow::Sequence*>(&n)) {
            for (size_t i = 0; i < seq->size(); i++) {
                collect(seq->at(i));
            }
        } else if (auto* loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&n)) {
            collect(loop->root());
        } else if (auto* ie = dynamic_cast<structured_control_flow::IfElse*>(&n)) {
            for (size_t i = 0; i < ie->size(); i++) {
                collect(ie->at(i).first);
            }
        }
    };
    collect(loop_.root());

    for (auto* tc : tile_copies) {
        widen_tile_copy_node(*tc);
    }

    // Recompute each pipeline wait's loads_per_group from the (now widened) cp.async
    // widths in its panel loop, so the vmcnt fence still keeps the right depth.
    std::unordered_map<structured_control_flow::StructuredLoop*, std::vector<tiles::PipelineWaitNode*>> waits;
    std::function<void(ControlFlowNode&)> gather = [&](ControlFlowNode& n) {
        if (auto* block = dynamic_cast<Block*>(&n)) {
            for (auto& node : block->dataflow().nodes()) {
                if (auto* w = dynamic_cast<tiles::PipelineWaitNode*>(&node)) {
                    if (auto* panel = enclosing_panel_loop(*block)) {
                        waits[panel].push_back(w);
                    }
                }
            }
        } else if (auto* seq = dynamic_cast<structured_control_flow::Sequence*>(&n)) {
            for (size_t i = 0; i < seq->size(); i++) {
                gather(seq->at(i));
            }
        } else if (auto* map = dynamic_cast<structured_control_flow::Map*>(&n)) {
            gather(map->root());
        } else if (auto* loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&n)) {
            gather(loop->root());
        } else if (auto* ie = dynamic_cast<structured_control_flow::IfElse*>(&n)) {
            for (size_t i = 0; i < ie->size(); i++) {
                gather(ie->at(i).first);
            }
        }
    };
    gather(loop_.root());
    for (auto& [panel, ws] : waits) {
        size_t words = sum_cp_async_words(panel->root());
        if (words > 0) {
            for (auto* w : ws) {
                w->set_loads_per_group(words);
            }
        }
    }

    analysis_manager.invalidate_all();
}

void TileVectorizer::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["parameters"] = nlohmann::json::object();

    serializer::JSONSerializer ser_flat(false);
    j["subgraph"] = nlohmann::json::object();
    j["subgraph"]["0"] = nlohmann::json::object();
    ser_flat.serialize_node(j["subgraph"]["0"], loop_);
}

TileVectorizer TileVectorizer::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j) {
    auto loop_id = j["subgraph"]["0"]["element_id"].get<size_t>();
    auto* element = builder.find_element_by_id(loop_id);
    if (element == nullptr) {
        throw InvalidTransformationDescriptionException("Element with ID " + std::to_string(loop_id) + " not found.");
    }
    auto* loop = dynamic_cast<structured_control_flow::StructuredLoop*>(element);
    if (loop == nullptr) {
        throw InvalidTransformationDescriptionException(
            "Element with ID " + std::to_string(loop_id) + " is not a structured loop."
        );
    }
    return TileVectorizer(*loop);
}

} // namespace transformations
} // namespace sdfg
