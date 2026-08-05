#include "sdfg/passes/dataflow/constant_propagation.h"

#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/structured_control_flow/while.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace passes {

namespace {

// A floating-point constant known to be held by a scalar container.
struct ConstVal {
    std::string data;
    types::PrimitiveType prim;
};

using Env = std::unordered_map<std::string, ConstVal>;
using ContainerSet = std::unordered_set<std::string>;

bool is_fp_scalar(const types::IType& type) {
    if (type.type_id() != types::TypeID::Scalar) {
        return false;
    }
    return types::is_floating_point(static_cast<const types::Scalar&>(type).primitive_type());
}

// Collects containers used through a non-computational (reference/dereference) memlet. Such
// containers are address-taken and must never be replaced by a literal.
void collect_address_taken(structured_control_flow::ControlFlowNode& node, ContainerSet& blacklist) {
    if (auto* block = dynamic_cast<structured_control_flow::Block*>(&node)) {
        for (auto& edge : block->dataflow().edges()) {
            if (edge.type() == data_flow::MemletType::Computational) {
                continue;
            }
            if (auto* src = dynamic_cast<const data_flow::AccessNode*>(&edge.src())) {
                blacklist.insert(src->data());
            }
            if (auto* dst = dynamic_cast<const data_flow::AccessNode*>(&edge.dst())) {
                blacklist.insert(dst->data());
            }
        }
    } else if (auto* seq = dynamic_cast<structured_control_flow::Sequence*>(&node)) {
        for (size_t i = 0; i < seq->size(); i++) {
            collect_address_taken(seq->at(i), blacklist);
        }
    } else if (auto* if_else = dynamic_cast<structured_control_flow::IfElse*>(&node)) {
        for (size_t i = 0; i < if_else->size(); i++) {
            collect_address_taken(if_else->at(i).first, blacklist);
        }
    } else if (auto* while_loop = dynamic_cast<structured_control_flow::While*>(&node)) {
        collect_address_taken(while_loop->root(), blacklist);
    } else if (auto* loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&node)) {
        collect_address_taken(loop->root(), blacklist);
    }
}

// Processes a single block: repoints reads of known constants and records/kills definitions.
// Returns the set of containers written in this block.
ContainerSet process_block(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Block& block,
    Env& env,
    const ContainerSet& blacklist,
    bool& applied
) {
    auto& dataflow = block.dataflow();

    // Count writes per container (a computational edge into an access node is a write).
    std::unordered_map<std::string, int> write_count;
    for (auto& edge : dataflow.edges()) {
        if (edge.type() != data_flow::MemletType::Computational || edge.dst_conn() != "void") {
            continue;
        }
        if (auto* dst = dynamic_cast<const data_flow::AccessNode*>(&edge.dst())) {
            write_count[dst->data()]++;
        }
    }

    // Phase 1: collect scalar reads of known constants that are not (re)written in this block.
    std::vector<std::pair<data_flow::Memlet*, ConstVal>> repoints;
    for (auto& edge : dataflow.edges()) {
        if (edge.type() != data_flow::MemletType::Computational || edge.dst_conn() == "void") {
            continue;
        }
        auto* src = dynamic_cast<data_flow::AccessNode*>(&edge.src());
        if (src == nullptr || dynamic_cast<data_flow::ConstantNode*>(src) != nullptr) {
            continue;
        }
        if (!edge.subset().empty()) {
            continue; // only whole-scalar reads
        }
        const std::string& name = src->data();
        if (blacklist.count(name) != 0 || write_count.count(name) != 0) {
            continue;
        }
        auto it = env.find(name);
        if (it == env.end()) {
            continue;
        }
        repoints.emplace_back(&edge, it->second);
    }
    for (auto& [edge, val] : repoints) {
        types::Scalar type(val.prim);
        builder.replace_memlet_src_with_constant(block, *edge, val.data, type);
        applied = true;
    }

    // Phase 2: record new constant definitions (before killing overwrites).
    std::vector<std::pair<std::string, ConstVal>> defs;
    for (auto* tasklet : dataflow.tasklets()) {
        if (tasklet->code() != data_flow::TaskletCode::assign) {
            continue;
        }
        // Single constant input
        const data_flow::Memlet* in_edge = nullptr;
        int in_count = 0;
        for (auto& ie : dataflow.in_edges(*tasklet)) {
            in_edge = &ie;
            in_count++;
        }
        if (in_count != 1 || in_edge == nullptr) {
            continue;
        }
        auto* constant = dynamic_cast<const data_flow::ConstantNode*>(&in_edge->src());
        if (constant == nullptr || !in_edge->subset().empty()) {
            continue;
        }
        // Single whole-scalar output
        const data_flow::Memlet* out_edge = nullptr;
        int out_count = 0;
        for (auto& oe : dataflow.out_edges(*tasklet)) {
            out_edge = &oe;
            out_count++;
        }
        if (out_count != 1 || out_edge == nullptr || out_edge->dst_conn() != "void" || !out_edge->subset().empty()) {
            continue;
        }
        auto* dst = dynamic_cast<const data_flow::AccessNode*>(&out_edge->dst());
        if (dst == nullptr || dynamic_cast<const data_flow::ConstantNode*>(dst) != nullptr) {
            continue;
        }
        const std::string& name = dst->data();
        if (blacklist.count(name) != 0 || write_count[name] != 1) {
            continue; // only a single, unambiguous definition qualifies
        }
        auto& type = builder.subject().type(name);
        if (!is_fp_scalar(type)) {
            continue;
        }
        defs.emplace_back(name, ConstVal{constant->data(), static_cast<const types::Scalar&>(type).primitive_type()});
    }

    // Any write invalidates a previously known constant; the recorded definitions re-establish it.
    ContainerSet written;
    for (auto& [name, count] : write_count) {
        env.erase(name);
        written.insert(name);
    }
    for (auto& [name, val] : defs) {
        env[name] = val;
    }
    return written;
}

ContainerSet process(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::ControlFlowNode& node,
    Env& env,
    const ContainerSet& blacklist,
    bool& applied
);

// Threads the environment through a sequence of children in program order.
ContainerSet process_sequence(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& sequence,
    Env& env,
    const ContainerSet& blacklist,
    bool& applied
) {
    ContainerSet written;
    for (size_t i = 0; i < sequence.size(); i++) {
        auto child_written = process(builder, sequence.at(i), env, blacklist, applied);
        written.insert(child_written.begin(), child_written.end());
    }
    return written;
}

ContainerSet process(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::ControlFlowNode& node,
    Env& env,
    const ContainerSet& blacklist,
    bool& applied
) {
    if (auto* block = dynamic_cast<structured_control_flow::Block*>(&node)) {
        return process_block(builder, *block, env, blacklist, applied);
    } else if (auto* assignment = dynamic_cast<structured_control_flow::AssignmentBlock*>(&node)) {
        ContainerSet written;
        for (auto& entry : assignment->assignments()) {
            const std::string& name = entry.first->get_name();
            env.erase(name);
            written.insert(name);
        }
        return written;
    } else if (auto* sequence = dynamic_cast<structured_control_flow::Sequence*>(&node)) {
        return process_sequence(builder, *sequence, env, blacklist, applied);
    } else if (auto* if_else = dynamic_cast<structured_control_flow::IfElse*>(&node)) {
        ContainerSet written;
        for (size_t i = 0; i < if_else->size(); i++) {
            Env branch_env = env; // each branch sees the entry state
            auto branch_written = process_sequence(builder, if_else->at(i).first, branch_env, blacklist, applied);
            written.insert(branch_written.begin(), branch_written.end());
        }
        // A branch may or may not run, so any definition it makes cannot be assumed afterwards.
        for (auto& name : written) {
            env.erase(name);
        }
        return written;
    } else if (auto* while_loop = dynamic_cast<structured_control_flow::While*>(&node)) {
        Env body_env = env; // entry constants still hold at the top of the body
        auto written = process_sequence(builder, while_loop->root(), body_env, blacklist, applied);
        for (auto& name : written) {
            env.erase(name);
        }
        return written;
    } else if (auto* loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&node)) {
        Env body_env = env;
        auto written = process_sequence(builder, loop->root(), body_env, blacklist, applied);
        for (auto& name : written) {
            env.erase(name);
        }
        return written;
    } else if (dynamic_cast<structured_control_flow::Return*>(&node) != nullptr ||
               dynamic_cast<structured_control_flow::Break*>(&node) != nullptr ||
               dynamic_cast<structured_control_flow::Continue*>(&node) != nullptr) {
        return {}; // Return / Break / Continue: no scalar writes to track
    } else {
        throw InvalidSDFGException("ConstantPropagation: unrecognized structured control flow node type");
    }
}

} // namespace

ConstantPropagation::ConstantPropagation() : Pass() {};

std::string ConstantPropagation::name() { return "ConstantPropagation"; };

bool ConstantPropagation::run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& sdfg = builder.subject();

    ContainerSet blacklist;
    collect_address_taken(sdfg.root(), blacklist);

    bool applied = false;
    Env env;
    process(builder, sdfg.root(), env, blacklist, applied);
    return applied;
};

} // namespace passes
} // namespace sdfg
