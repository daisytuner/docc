#include "sdfg/targets/gpu/gpu_mma_interleave.h"

#include <algorithm>
#include <functional>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_nodes/barrier_local_node.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/targets/gpu/gpu_map_utils.h"
#include "sdfg/targets/gpu/gpu_mma_fragment_load_node.h"
#include "sdfg/targets/gpu/gpu_mma_matmul_node.h"
#include "sdfg/targets/gpu/gpu_offload_schedule_type.h"
#include "sdfg/tiles/library_nodes/tile_copy_node.h"
#include "sdfg/tiles/tile_target_registry.h"
#include "sdfg/types/utils.h"
#include "sdfg/visitor/for_each.h"

namespace sdfg::gpu {

namespace {

using structured_control_flow::Block;
using structured_control_flow::ControlFlowNode;
using structured_control_flow::Sequence;
using structured_control_flow::StructuredLoop;

std::optional<long long> as_constant(const symbolic::Expression& e) {
    if (e.is_null() || !SymEngine::is_a<SymEngine::Integer>(*e)) {
        return std::nullopt;
    }
    return SymEngine::rcp_static_cast<const SymEngine::Integer>(e)->as_int();
}

Sequence& loop_body(StructuredLoop& loop) {
    Sequence* body = &loop.root();
    while (body->size() == 1) {
        auto* inner = dynamic_cast<Sequence*>(&body->at(0));
        if (inner == nullptr) {
            break;
        }
        body = inner;
    }
    return *body;
}

const tiles::TileTarget* enclosing_tile_target(ControlFlowNode& node) {
    for (auto* parent : ControlFlowNode::parent_chain(node)) {
        if (auto* map = dynamic_cast<structured_control_flow::Map*>(parent)) {
            if (auto* target = tiles::TileTargetRegistry::instance().get(map->schedule_type().value())) {
                return target;
            }
        }
    }
    return nullptr;
}

// Lanes per wave of the innermost enclosing GPU map that maps waves (lanes > 1).
std::optional<long long> wave_lanes(ControlFlowNode& node) {
    for (auto* parent : ControlFlowNode::parent_chain(node)) {
        auto* map = dynamic_cast<structured_control_flow::Map*>(parent);
        if (map == nullptr || !is_gpu_schedule(map->schedule_type()) ||
            !map->schedule_type().properties().contains("target_level")) {
            continue;
        }
        const long long lanes = ScheduleType_GPU_Offload::lanes(map->schedule_type())->as_int();
        if (lanes > 1) {
            return lanes;
        }
    }
    return std::nullopt;
}

bool is_barrier_block(ControlFlowNode& node) {
    auto* block = dynamic_cast<Block*>(&node);
    if (block == nullptr) {
        return false;
    }
    for (auto* lib : block->dataflow().library_nodes()) {
        if (dynamic_cast<data_flow::BarrierLocalNode*>(lib) != nullptr) {
            return true;
        }
    }
    return false;
}

const types::IType* connector_type(const data_flow::LibraryNode& node, const std::string& conn) {
    for (auto& m : node.get_parent().in_edges(node)) {
        if (m.dst_conn() == conn) {
            return &m.base_type();
        }
    }
    return nullptr;
}

const std::string* connector_container(const data_flow::LibraryNode& node, const std::string& conn) {
    for (auto& m : node.get_parent().in_edges(node)) {
        if (m.dst_conn() == conn) {
            if (auto* access = dynamic_cast<const data_flow::AccessNode*>(&m.src())) {
                return &access->data();
            }
        }
    }
    return nullptr;
}

// Global read instructions per thread of a register-staged load.
std::optional<long long> global_reads(const tiles::TileCopyNode& copy) {
    auto total = as_constant(copy.plan().src.total_elements());
    auto threads = as_constant(copy.coop_threads());
    const auto* src_type = connector_type(copy, "_src");
    if (!total || !threads || *threads <= 0 || src_type == nullptr) {
        return std::nullopt;
    }
    const long long elem_bytes = types::bit_width(src_type->primitive_type()) / 8;
    if (elem_bytes <= 0) {
        return std::nullopt;
    }
    auto ceil_div = [](long long a, long long b) {
        return (a + b - 1) / b;
    };
    if (copy.atom() == tiles::CopyAtom::CoalesceSync) {
        return 4 * ceil_div(*total / 16, *threads);
    }
    const bool wide = copy.atom() == tiles::CopyAtom::VectorSync || copy.atom() == tiles::CopyAtom::CpAsync;
    const long long factor = wide ? static_cast<long long>(copy.bytes()) / elem_bytes : 1;
    if (factor <= 0) {
        return std::nullopt;
    }
    return ceil_div(*total, *threads * factor);
}

struct PanelShape {
    std::vector<tiles::TileCopyNode*> loads;
    StructuredLoop* k_loop = nullptr;
    long long global_reads = 0;
    long long mma_per_step = 0;
    long long steps = 0;
    long long lds_bytes_per_step = -1;
};

std::optional<PanelShape> analyze(builder::StructuredSDFGBuilder& builder, StructuredLoop& loop) {
    if (dynamic_cast<structured_control_flow::Map*>(&loop) != nullptr) {
        return std::nullopt;
    }
    auto& sdfg = builder.subject();
    auto& body = loop_body(loop);
    PanelShape shape;
    bool barrier_after = false;
    for (size_t i = 0; i < body.size(); ++i) {
        auto& child = body.at(i);
        if (dynamic_cast<structured_control_flow::IfElse*>(&child) != nullptr) {
            return std::nullopt;
        }
        if (auto* block = dynamic_cast<Block*>(&child)) {
            for (auto* lib : block->dataflow().library_nodes()) {
                if (dynamic_cast<tiles::PipelineScheduleNode*>(lib) != nullptr) {
                    return std::nullopt;
                }
                auto* copy = dynamic_cast<tiles::TileCopyNode*>(lib);
                if (copy != nullptr && copy->phase() == tiles::CopyPhase::LoadRegs) {
                    if (shape.k_loop != nullptr) {
                        return std::nullopt;
                    }
                    shape.loads.push_back(copy);
                }
            }
            barrier_after |= shape.k_loop != nullptr && is_barrier_block(child);
            continue;
        }
        auto* inner = dynamic_cast<StructuredLoop*>(&child);
        if (inner == nullptr) {
            continue;
        }
        long long mmas = 0;
        bool nested = false;
        std::vector<GpuMmaFragmentLoadNode*> fragments;
        std::function<void(ControlFlowNode&)> scan = [&](ControlFlowNode& n) {
            if (auto* b = dynamic_cast<Block*>(&n)) {
                for (auto* lib : b->dataflow().library_nodes()) {
                    mmas += dynamic_cast<GpuMmaMatmulNode*>(lib) != nullptr;
                    if (auto* f = dynamic_cast<GpuMmaFragmentLoadNode*>(lib)) {
                        fragments.push_back(f);
                    }
                }
            } else if (auto* s = dynamic_cast<Sequence*>(&n)) {
                for (size_t j = 0; j < s->size(); ++j) {
                    scan(s->at(j));
                }
            } else {
                nested = true;
            }
        };
        scan(inner->root());
        if (mmas == 0) {
            continue;
        }
        auto steps = as_constant(inner->num_iterations());
        if (shape.k_loop != nullptr || nested || !steps || *steps <= 0) {
            return std::nullopt;
        }
        shape.k_loop = inner;
        shape.mma_per_step = mmas;
        shape.steps = *steps;
        if (auto lanes = wave_lanes(*inner)) {
            long long bytes = 0;
            for (auto* f : fragments) {
                const auto* container = connector_container(*f, f->input(GpuMmaFragmentLoadNode::PTR_INPUT_IDX));
                if (container == nullptr || !sdfg.type(*container).storage_type().is_nv_shared()) {
                    continue;
                }
                long long elems = 1;
                for (const auto& d : f->block_size().get_shape(f->fragment_type())) {
                    elems *= as_constant(d).value_or(0);
                }
                bytes += elems * static_cast<long long>(types::bit_width(f->element_type()) / 8) / *lanes;
            }
            shape.lds_bytes_per_step = bytes;
        }
    }
    if (shape.loads.empty() || shape.k_loop == nullptr || !barrier_after) {
        return std::nullopt;
    }
    for (auto* copy : shape.loads) {
        auto reads = global_reads(*copy);
        if (!reads) {
            return std::nullopt;
        }
        shape.global_reads += *reads;
    }
    return shape;
}

} // namespace

GpuMmaInterleave::GpuMmaInterleave(
    StructuredLoop& loop, size_t global_read_spacing, size_t lds_read_position, size_t lds_reads_per_step
)
    : loop_(loop), global_read_spacing_(std::max<size_t>(1, global_read_spacing)),
      lds_read_position_(lds_read_position), lds_reads_per_step_(lds_reads_per_step) {
}

std::optional<std::vector<tiles::ScheduleGroup>> GpuMmaInterleave::recipe(builder::StructuredSDFGBuilder& builder) const {
    auto shape = analyze(builder, loop_);
    if (!shape) {
        return std::nullopt;
    }
    long long lds_reads = static_cast<long long>(lds_reads_per_step_);
    if (lds_reads == 0) {
        if (shape->lds_bytes_per_step < 0) {
            return std::nullopt;
        }
        lds_reads = (shape->lds_bytes_per_step + 15) / 16;
    }
    const long long total_mma = shape->mma_per_step * shape->steps;
    long long spacing = static_cast<long long>(global_read_spacing_);
    if (shape->global_reads > 0) {
        spacing = std::max<long long>(1, std::min(spacing, total_mma / shape->global_reads));
    }
    const long long read_at = std::min<long long>(static_cast<long long>(lds_read_position_), shape->mma_per_step - 1);

    std::vector<tiles::ScheduleGroup> groups;
    auto push = [&](tiles::InstructionClass cls, long long count) {
        if (count > 0) {
            groups.push_back({cls, static_cast<size_t>(count)});
        }
    };
    push(tiles::InstructionClass::LdsRead, lds_reads);
    long long left = shape->global_reads;
    for (long long k = 0; k < shape->steps; ++k) {
        for (long long m = 0; m < shape->mma_per_step; ++m) {
            push(tiles::InstructionClass::Mfma, 1);
            if (left > 0 && (k * shape->mma_per_step + m) % spacing == 0) {
                push(tiles::InstructionClass::GlobalRead, 1);
                --left;
            }
            if (k + 1 < shape->steps && m == read_at) {
                push(tiles::InstructionClass::LdsRead, lds_reads);
            }
        }
    }
    push(tiles::InstructionClass::GlobalRead, left);
    return groups;
}

bool GpuMmaInterleave::can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager&) {
    return enclosing_tile_target(loop_) != nullptr && recipe(builder).has_value();
}

void GpuMmaInterleave::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto groups = recipe(builder);
    auto shape = analyze(builder, loop_);
    const auto* target = enclosing_tile_target(loop_);
    if (!groups || !shape || target == nullptr) {
        throw transformations::InvalidTransformationException("GpuMmaInterleave: not applicable");
    }
    for (auto* copy : shape->loads) {
        copy->set_scheduled(true);
    }
    auto& body = loop_body(loop_);
    auto& block = builder.add_block_after(body, *shape->k_loop, loop_.debug_info());
    builder.add_library_node<
        tiles::PipelineScheduleNode>(block, loop_.debug_info(), target->implementation_type(), *groups);
    analysis_manager.invalidate_all();
}

void GpuMmaInterleave::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["parameters"] = {
        {"global_read_spacing", global_read_spacing_},
        {"lds_read_position", lds_read_position_},
        {"lds_reads_per_step", lds_reads_per_step_},
    };
    serializer::JSONSerializer ser_flat(false);
    j["subgraph"] = nlohmann::json::object();
    j["subgraph"]["0"] = nlohmann::json::object();
    ser_flat.serialize_node(j["subgraph"]["0"], loop_);
}

GpuMmaInterleave GpuMmaInterleave::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j) {
    auto loop_id = j.at("subgraph").at("0").at("element_id").get<size_t>();
    auto* loop = dynamic_cast<StructuredLoop*>(builder.find_element_by_id(loop_id));
    if (loop == nullptr) {
        throw transformations::InvalidTransformationDescriptionException(
            "Element with ID " + std::to_string(loop_id) + " is not a structured loop."
        );
    }
    const auto& params = j.value("parameters", nlohmann::json::object());
    return GpuMmaInterleave(
        *loop,
        params.value("global_read_spacing", size_t(3)),
        params.value("lds_read_position", size_t(1)),
        params.value("lds_reads_per_step", size_t(0))
    );
}

} // namespace sdfg::gpu
