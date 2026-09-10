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
#include "sdfg/tiles/library_nodes/async_copy_node.h"
#include "sdfg/tiles/tile.h"
#include "sdfg/tiles/tile_target_registry.h"
#include "sdfg/tiles/vectorize/copy_widen.h"

namespace sdfg {
namespace transformations {

namespace {

using structured_control_flow::Block;
using structured_control_flow::ControlFlowNode;

// A GPU thread-scheduled map — the coverage map that drives a cooperative copy.
bool is_gpu_coverage_map(structured_control_flow::Map& map) {
    return tiles::AxisSchedule::classify_level(map.schedule_type()).has_value();
}

// The copy library node in @p block (CpAsync or VectorCopy), or nullptr.
data_flow::LibraryNode* copy_node_in(Block& block) {
    for (auto& node : block.dataflow().nodes()) {
        if (dynamic_cast<tiles::CpAsyncCopyNode*>(&node) || dynamic_cast<tiles::VectorCopyNode*>(&node)) {
            return dynamic_cast<data_flow::LibraryNode*>(&node);
        }
    }
    return nullptr;
}

// Does @p block hold a single scalar `assign` copy (one in, one out, both access
// nodes)? That is the shape LocalStorage emits before any widening.
bool is_scalar_copy(Block& block) {
    auto& df = block.dataflow();
    data_flow::Tasklet* tk = nullptr;
    for (auto& node : df.nodes()) {
        if (auto* t = dynamic_cast<data_flow::Tasklet*>(&node)) {
            if (tk != nullptr) {
                return false; // more than one tasklet
            }
            tk = t;
        }
    }
    if (tk == nullptr || tk->code() != data_flow::TaskletCode::assign) {
        return false;
    }
    size_t ins = 0, outs = 0;
    for (auto& m : df.in_edges(*tk)) {
        ins++;
        if (dynamic_cast<data_flow::AccessNode*>(&m.src()) == nullptr) return false;
    }
    for (auto& m : df.out_edges(*tk)) {
        outs++;
        if (dynamic_cast<data_flow::AccessNode*>(&m.dst()) == nullptr) return false;
    }
    return ins == 1 && outs == 1;
}

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

// Sum the cp.async transfer words (bytes / 4) over @p scope's blocks.
size_t sum_cp_async_words(ControlFlowNode& scope) {
    size_t words = 0;
    std::function<void(ControlFlowNode&)> walk = [&](ControlFlowNode& n) {
        if (auto* block = dynamic_cast<Block*>(&n)) {
            for (auto& node : block->dataflow().nodes()) {
                if (auto* cp = dynamic_cast<tiles::CpAsyncCopyNode*>(&node)) {
                    words += cp->bytes() / 4;
                }
            }
        } else if (auto* seq = dynamic_cast<structured_control_flow::Sequence*>(&n)) {
            for (size_t i = 0; i < seq->size(); i++) walk(seq->at(i));
        } else if (auto* map = dynamic_cast<structured_control_flow::Map*>(&n)) {
            walk(map->root());
        } else if (auto* loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&n)) {
            walk(loop->root());
        } else if (auto* ie = dynamic_cast<structured_control_flow::IfElse*>(&n)) {
            for (size_t i = 0; i < ie->size(); i++) walk(ie->at(i).first);
        }
    };
    walk(scope);
    return words;
}

} // namespace

TileVectorizer::TileVectorizer(structured_control_flow::StructuredLoop& loop) : loop_(loop) {}

std::string TileVectorizer::name() const { return "TileVectorizer"; }

bool TileVectorizer::can_be_applied(builder::StructuredSDFGBuilder&, analysis::AnalysisManager&) {
    // Applicable when the subtree holds at least one cooperative copy to widen: a
    // scalar copy under a GPU coverage map, or an already-lowered copy node.
    bool found = false;
    std::function<void(ControlFlowNode&)> scan = [&](ControlFlowNode& n) {
        if (found) {
            return;
        }
        if (auto* map = dynamic_cast<structured_control_flow::Map*>(&n)) {
            if (is_gpu_coverage_map(*map)) {
                for (size_t j = 0; j < map->root().size(); j++) {
                    if (auto* b = dynamic_cast<Block*>(&map->root().at(j))) {
                        if (copy_node_in(*b) != nullptr || is_scalar_copy(*b)) {
                            found = true;
                            return;
                        }
                    }
                }
            }
            scan(map->root());
        } else if (auto* seq = dynamic_cast<structured_control_flow::Sequence*>(&n)) {
            for (size_t i = 0; i < seq->size(); i++) scan(seq->at(i));
        } else if (auto* loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&n)) {
            scan(loop->root());
        } else if (auto* ie = dynamic_cast<structured_control_flow::IfElse*>(&n)) {
            for (size_t i = 0; i < ie->size(); i++) scan(ie->at(i).first);
        }
    };
    scan(loop_.root());
    return found;
}

void TileVectorizer::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    // Collect targets first (widening mutates the graph): scalar copy blocks (with
    // the resolved backend impl for the node they become) and existing copy nodes.
    std::vector<std::pair<Block*, data_flow::ImplementationType>> scalar_copies;
    std::vector<std::pair<Block*, data_flow::LibraryNode*>> node_copies;
    std::function<void(ControlFlowNode&)> collect = [&](ControlFlowNode& n) {
        if (auto* map = dynamic_cast<structured_control_flow::Map*>(&n)) {
            if (is_gpu_coverage_map(*map)) {
                auto impl = tiles::TileTargetRegistry::instance().implementation_type(map->schedule_type().value());
                for (size_t j = 0; j < map->root().size(); j++) {
                    if (auto* b = dynamic_cast<Block*>(&map->root().at(j))) {
                        if (auto* cn = copy_node_in(*b)) {
                            node_copies.emplace_back(b, cn);
                        } else if (is_scalar_copy(*b)) {
                            scalar_copies.emplace_back(b, impl);
                        }
                    }
                }
            }
            collect(map->root());
        } else if (auto* seq = dynamic_cast<structured_control_flow::Sequence*>(&n)) {
            for (size_t i = 0; i < seq->size(); i++) collect(seq->at(i));
        } else if (auto* loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&n)) {
            collect(loop->root());
        } else if (auto* ie = dynamic_cast<structured_control_flow::IfElse*>(&n)) {
            for (size_t i = 0; i < ie->size(); i++) collect(ie->at(i).first);
        }
    };
    collect(loop_.root());

    for (auto& [block, node] : node_copies) {
        auto* cp = dynamic_cast<tiles::CpAsyncCopyNode*>(node);
        size_t cur = cp ? cp->bytes() : static_cast<tiles::VectorCopyNode*>(node)->bytes();
        tiles::widen_existing_copy(builder, *block, *node, cur);
    }
    for (auto& [block, impl] : scalar_copies) {
        tiles::rewrite_cooperative_copy(builder, *block, /*allow_vectorize=*/true, tiles::CopyTransfer::VectorSync, impl);
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
            for (size_t i = 0; i < seq->size(); i++) gather(seq->at(i));
        } else if (auto* map = dynamic_cast<structured_control_flow::Map*>(&n)) {
            gather(map->root());
        } else if (auto* loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&n)) {
            gather(loop->root());
        } else if (auto* ie = dynamic_cast<structured_control_flow::IfElse*>(&n)) {
            for (size_t i = 0; i < ie->size(); i++) gather(ie->at(i).first);
        }
    };
    gather(loop_.root());
    for (auto& [panel, ws] : waits) {
        size_t words = sum_cp_async_words(panel->root());
        if (words > 0) {
            for (auto* w : ws) w->set_loads_per_group(words);
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
