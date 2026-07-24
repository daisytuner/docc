#include "sdfg/passes/offloading/device_buffer_reuse_pass.h"

#include <algorithm>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "sdfg/analysis/dominance_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/code_node.h"
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/data_flow/data_flow_node.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/offloading/data_offloading_node.h"

namespace sdfg {
namespace passes {

// The device container an offloading node operates on: the destination of an ALLOC node's `_dev`
// edge, or the source of a FREE node's `_dev` edge.
const data_flow::AccessNode* device_access(structured_control_flow::Block& block, offloading::DataOffloadingNode& node) {
    auto& dataflow = block.dataflow();
    if (node.is_alloc()) {
        for (auto& memlet : dataflow.out_edges(node)) {
            if (auto* access = dynamic_cast<const data_flow::AccessNode*>(&memlet.dst())) {
                return access;
            }
        }
    } else if (node.is_free()) {
        for (auto& memlet : dataflow.in_edges(node)) {
            if (auto* access = dynamic_cast<const data_flow::AccessNode*>(&memlet.src())) {
                return access;
            }
        }
    }
    return nullptr;
}

// Find the liveness User attached to a specific access-node element of a container.
analysis::User* user_for_element(analysis::Users& users, const std::string& container, const data_flow::AccessNode* el) {
    for (auto* user : users.uses(container)) {
        if (user->element() == el) {
            return user;
        }
    }
    return nullptr;
}

BufferReuseMarker::BufferReuseMarker(
    builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager, analysis::Users& users
)
    : visitor::StructuredSDFGVisitor(builder, analysis_manager), users_(users), sdfg_(builder.subject()) {}

bool BufferReuseMarker::visit_internal(structured_control_flow::Sequence& parent) {
    for (size_t i = 0; i < parent.size(); i++) {
        auto& current = parent.at(i);
        if (auto* block = dyn_cast<structured_control_flow::Block*>(&current)) {
            process_block(*block);
        } else if (auto* seq = dyn_cast<structured_control_flow::Sequence*>(&current)) {
            visit_internal(*seq);
        } else if (auto* if_else = dyn_cast<structured_control_flow::IfElse*>(&current)) {
            // Buffers created in different branches are not totally ordered, so they interfere.
            std::vector<std::pair<size_t, size_t>> ranges;
            for (size_t c = 0; c < if_else->size(); c++) {
                size_t start = candidates.size();
                visit_internal(if_else->at(c).first);
                ranges.emplace_back(start, candidates.size());
            }
            for (size_t a = 0; a < ranges.size(); a++) {
                for (size_t b = a + 1; b < ranges.size(); b++) {
                    for (size_t x = ranges[a].first; x < ranges[a].second; x++) {
                        for (size_t y = ranges[b].first; y < ranges[b].second; y++) {
                            add_edge(x, y);
                        }
                    }
                }
            }
        } else if (auto* loop = dyn_cast<structured_control_flow::StructuredLoop*>(&current)) {
            visit_internal(loop->root());
        }
    }
    return false;
}

void BufferReuseMarker::add_edge(size_t a, size_t b) {
    if (a == b) {
        return;
    }
    adjacency[a].insert(b);
    adjacency[b].insert(a);
}

// Promote a pending allocation to a live candidate once a using kernel has been found.
void BufferReuseMarker::assign(const std::string& container) {
    auto it = unassigned_.find(container);
    if (it == unassigned_.end()) {
        return;
    }
    AllocRec rec = it->second;
    unassigned_.erase(it);

    Candidate cand;
    cand.container = container;
    cand.dtype = sdfg_.type(container).primitive_type();
    cand.size = rec.node->size();
    cand.alloc_block = rec.block;
    cand.alloc_node = rec.node;
    cand.alloc_user = user_for_element(users_, container, rec.access);

    size_t idx = candidates.size();
    candidates.push_back(std::move(cand));
    adjacency.emplace_back();

    // Overlapping live ranges interfere.
    for (size_t l : live_) {
        add_edge(idx, l);
    }
    live_.push_back(idx);
    live_index_[container] = idx;
}

// End a buffer's live range at its free.
void BufferReuseMarker::close(
    const std::string& container,
    structured_control_flow::Block& block,
    offloading::DataOffloadingNode& node,
    const data_flow::AccessNode* access
) {
    if (live_index_.find(container) == live_index_.end()) {
        // Allocated but never used by a kernel: realise it now so the alloc/free still pair up.
        assign(container);
    }
    auto li = live_index_.find(container);
    if (li == live_index_.end()) {
        return;
    }
    size_t idx = li->second;
    candidates[idx].free_block = &block;
    candidates[idx].free_node = &node;
    candidates[idx].free_user = user_for_element(users_, container, access);
    live_.erase(std::remove(live_.begin(), live_.end(), idx), live_.end());
    live_index_.erase(li);
}

void BufferReuseMarker::process_block(structured_control_flow::Block& block) {
    auto& dataflow = block.dataflow();

    // A kernel use of a container is an access node wired to a (non-offloading) code node.
    std::unordered_set<std::string> kernel_uses;
    for (auto& edge : dataflow.edges()) {
        const data_flow::AccessNode* access = nullptr;
        const data_flow::DataFlowNode* other = nullptr;
        if (auto* a = dynamic_cast<const data_flow::AccessNode*>(&edge.src())) {
            access = a;
            other = &edge.dst();
        } else if (auto* a = dynamic_cast<const data_flow::AccessNode*>(&edge.dst())) {
            access = a;
            other = &edge.src();
        }
        if (access == nullptr || dynamic_cast<const offloading::DataOffloadingNode*>(other) != nullptr) {
            continue;
        }
        if (dynamic_cast<const data_flow::CodeNode*>(other) != nullptr) {
            kernel_uses.insert(access->data());
        }
    }

    // Record pure allocs as pending, disqualify transfer buffers, and gather frees.
    std::vector<std::tuple<std::string, offloading::DataOffloadingNode*, const data_flow::AccessNode*>> frees;
    for (auto& n : dataflow.nodes()) {
        auto* off = dynamic_cast<offloading::DataOffloadingNode*>(&n);
        if (off == nullptr) {
            continue;
        }
        const auto* access = device_access(block, *off);
        if (access == nullptr) {
            continue;
        }
        const std::string& container = access->data();
        if (off->has_transfer()) {
            disqualified_.insert(container);
            unassigned_.erase(container);
        } else if (off->is_alloc()) {
            if (!disqualified_.count(container)) {
                unassigned_[container] = {&block, off, access};
            }
        } else if (off->is_free()) {
            frees.emplace_back(container, off, access);
        }
    }

    // Assign allocations that this block's kernels consume, then apply frees.
    for (const auto& container : kernel_uses) {
        assign(container);
    }
    for (auto& [container, node, access] : frees) {
        close(container, block, *node, access);
    }
}

DeviceBufferReusePass::DeviceBufferReusePass(bool consider_dataflow_branching)
    : consider_dataflow_branching_(consider_dataflow_branching) {}

std::string DeviceBufferReusePass::name() { return "DeviceBufferReuse"; }

void add_edge(std::vector<std::unordered_set<size_t>>& adjacency, size_t a, size_t b) {
    if (a == b) {
        return;
    }
    adjacency[a].insert(b);
    adjacency[b].insert(a);
}

// Control-flow ordering of two buffers: 1 if `a` is entirely before `b` (a's free dominates b's
// alloc), 2 if `b` is entirely before `a`, 0 if they are not totally ordered (overlap/divergent).
int ordering(const Candidate& a, const Candidate& b, analysis::DominanceAnalysis& dominance) {
    if (a.alloc_user == nullptr || a.free_user == nullptr || b.alloc_user == nullptr || b.free_user == nullptr) {
        return 0;
    }
    if (dominance.dominates(*a.free_user, *b.alloc_user)) {
        return 1;
    }
    if (dominance.dominates(*b.free_user, *a.alloc_user)) {
        return 2;
    }
    return 0;
}

// Whether a user lies within a candidate's live range [alloc, free].
bool in_range(const Candidate& c, analysis::User* user, analysis::DominanceAnalysis& dominance) {
    return dominance.dominates(*c.alloc_user, *user) && dominance.post_dominates(*c.free_user, *user);
};

// For a control-flow-ordered pair, decide whether a data dependency forces the later buffer's
// use to happen-after the earlier's. If so, the two are not independent dataflow branches.
bool data_serialized(
    const Candidate& earlier, const Candidate& later, analysis::Users& users, analysis::DominanceAnalysis& dominance
) {
    std::unordered_set<std::string> produced;
    for (auto* w : users.writes()) {
        const std::string& c = w->container();
        if (c == earlier.container || c == later.container) {
            continue;
        }
        if (in_range(earlier, w, dominance)) {
            produced.insert(c);
        }
    }
    if (produced.empty()) {
        return false;
    }
    for (auto* u : users.uses()) {
        const std::string& c = u->container();
        if (c == earlier.container || c == later.container) {
            continue;
        }
        if (produced.count(c) && in_range(later, u, dominance)) {
            return true;
        }
    }
    return false;
};

bool DeviceBufferReusePass::run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& users = analysis_manager.get<analysis::Users>();
    auto& dominance = analysis_manager.get<analysis::DominanceAnalysis>();

    // =======================================================================================
    // MARK: build candidates and the interference graph in a single visitor traversal.
    //
    // The marker walks the SDFG in program order with a set of unassigned allocations. An
    // allocation becomes a candidate the moment a kernel using it is found, at which point it is
    // wired against every buffer currently live (overlapping live ranges interfere). Frees end a
    // live range directly. Buffers created in sibling if/else branches interfere too, since they
    // are not totally ordered.
    // =======================================================================================
    BufferReuseMarker marker(builder, analysis_manager, users);
    marker.visit();

    auto& candidates = marker.candidates;
    auto& adjacency = marker.adjacency;

    if (candidates.size() < 2) {
        return false;
    }

    // A buffer whose liveness boundaries could not be resolved is never safe to merge.
    for (size_t i = 0; i < candidates.size(); i++) {
        if (candidates[i].alloc_user == nullptr || candidates[i].free_user == nullptr) {
            for (size_t j = 0; j < candidates.size(); j++) {
                add_edge(adjacency, i, j);
            }
        }
    }

    // Dataflow branching: asynchronous kernels that are control-flow-ordered but exchange no data
    // may still run concurrently, so independent sequential buffers must not share storage. This is
    // the only interference that depends on data dependencies, so it augments the traversal graph
    // here, during marking, using the cached user analysis.
    if (consider_dataflow_branching_) {
        for (size_t i = 0; i < candidates.size(); i++) {
            for (size_t j = i + 1; j < candidates.size(); j++) {
                if (adjacency[i].count(j) || candidates[i].dtype != candidates[j].dtype) {
                    continue;
                }
                int ord = ordering(candidates[i], candidates[j], dominance);
                if (ord == 0) {
                    add_edge(adjacency, i, j);
                    continue;
                }
                const Candidate& earlier = (ord == 1) ? candidates[i] : candidates[j];
                const Candidate& later = (ord == 1) ? candidates[j] : candidates[i];
                if (!data_serialized(earlier, later, users, dominance)) {
                    add_edge(adjacency, i, j);
                }
            }
        }
    }

    // =======================================================================================
    // COLOR: greedy graph colouring, preferring larger allocations first.
    // =======================================================================================
    std::vector<size_t> order(candidates.size());
    for (size_t i = 0; i < order.size(); i++) {
        order[i] = i;
    }
    std::stable_sort(order.begin(), order.end(), [&](size_t i, size_t j) {
        return symbolic::is_true(symbolic::Gt(candidates[i].size, candidates[j].size));
    });

    std::vector<std::vector<size_t>> colors; // each colour is a list of candidate indices
    for (size_t idx : order) {
        bool placed = false;
        for (auto& members : colors) {
            if (candidates[members.front()].dtype != candidates[idx].dtype) {
                continue;
            }
            bool ok = true;
            for (size_t m : members) {
                if (adjacency[idx].count(m)) {
                    ok = false;
                    break;
                }
            }
            if (ok) {
                members.push_back(idx);
                placed = true;
                break;
            }
        }
        if (!placed) {
            colors.push_back({idx});
        }
    }

    // =======================================================================================
    // SWEEP: apply every merge in a single pass.
    // =======================================================================================
    bool changed = false;
    for (auto& members : colors) {
        if (members.size() < 2) {
            continue;
        }

        // The colouring order is only a heuristic: with symbolic sizes `Gt` is not a total order
        // (e.g. S vs. T compares false both ways), so members.front() is not guaranteed to be the
        // largest. It still serves as the shared allocation's name, but the surviving allocation
        // must be sized to the symbolic maximum over all members.
        const std::string representative = candidates[members.front()].container;
        symbolic::Expression max_size = candidates[members.front()].size;
        for (size_t m : members) {
            max_size = symbolic::max(max_size, candidates[m].size);
        }

        // Identify the earliest allocation and latest free in program order; these survive.
        size_t earliest = members.front();
        size_t latest = members.front();
        for (size_t m : members) {
            bool is_earliest = true;
            bool is_latest = true;
            for (size_t o : members) {
                if (o == m) {
                    continue;
                }
                if (ordering(candidates[m], candidates[o], dominance) != 1) {
                    is_earliest = false;
                }
                if (ordering(candidates[o], candidates[m], dominance) != 1) {
                    is_latest = false;
                }
            }
            if (is_earliest) {
                earliest = m;
            }
            if (is_latest) {
                latest = m;
            }
        }

        // Alias every member onto the representative container.
        for (size_t m : members) {
            if (candidates[m].container != representative) {
                builder.rename_container(candidates[m].container, representative);
            }
        }

        // Resize the surviving allocation to cover the largest member.
        candidates[earliest].alloc_node->set_size(max_size);

        // Drop the now-redundant allocations and frees, keeping the earliest alloc / latest free.
        for (size_t m : members) {
            if (m != earliest) {
                builder.remove_from_parent(*candidates[m].alloc_block);
            }
            if (m != latest) {
                builder.remove_from_parent(*candidates[m].free_block);
            }
        }

        changed = true;
    }

    return changed;
}

} // namespace passes
} // namespace sdfg
