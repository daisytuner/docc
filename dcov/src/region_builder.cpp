#include "sdfg/dcov/region_builder.h"

#include <algorithm>
#include <cstdio>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/code_node.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/reduce.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/structured_control_flow/while.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace dcov {

namespace {

std::string short_hash(const std::string& s) {
    size_t h = std::hash<std::string>{}(s);
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%08x", static_cast<unsigned>(h & 0xffffffffu));
    return std::string(buf);
}

std::string join(const std::vector<std::string>& xs, char sep) {
    std::string out;
    for (size_t i = 0; i < xs.size(); ++i) {
        if (i) out += sep;
        out += xs[i];
    }
    return out;
}

std::string tasklet_op_name(data_flow::TaskletCode code) { return "t" + std::to_string(static_cast<int>(code)); }

std::string access_name(const data_flow::DataFlowNode& node) {
    if (auto an = dynamic_cast<const data_flow::AccessNode*>(&node)) return an->data();
    if (auto cn = dynamic_cast<const data_flow::ConstantNode*>(&node)) return cn->data();
    return "";
}

std::string container_dtype(const StructuredSDFG& sdfg, const std::string& name) {
    if (name.empty() || !sdfg.exists(name)) return "";
    return sdfg.type(name).print();
}

/// Rename-robust operand token used in statement_key. Transient (intermediate)
/// containers are renamed freely by normalization, so their literal names are
/// not stable across compiler stages. We replace them with a canonical token
/// keyed by datatype + role, while keeping external (non-transient) names
/// literal so distinct arguments stay distinguishable.
std::string operand_token(const StructuredSDFG& sdfg, const std::string& name, const char* role) {
    if (name.empty()) return "";
    if (sdfg.exists(name) && sdfg.is_transient(name)) return "$t:" + container_dtype(sdfg, name) + ":" + role;
    return name;
}

Statement make_statement(
    const StructuredSDFG& sdfg,
    const data_flow::DataFlowGraph& dataflow,
    const data_flow::CodeNode& node,
    const std::string& op
) {
    Statement stmt;
    stmt.op = op;

    std::vector<std::string> key_inputs;
    for (const auto& conn : node.inputs()) {
        const data_flow::Memlet* m = dataflow.in_edge_for_connector(node, conn);
        if (m != nullptr) {
            std::string name = access_name(m->src());
            stmt.inputs.push_back(name);
            key_inputs.push_back(operand_token(sdfg, name, "in"));
        }
    }

    for (const auto& conn : node.outputs()) {
        auto edges = dataflow.out_edges_for_connector(node, conn);
        if (!edges.empty()) {
            stmt.output = access_name(edges.front()->dst());
            break;
        }
    }

    stmt.dtype = container_dtype(sdfg, stmt.output);
    if (stmt.dtype.empty()) {
        for (const auto& in : stmt.inputs) {
            stmt.dtype = container_dtype(sdfg, in);
            if (!stmt.dtype.empty()) break;
        }
    }

    std::string key_output = operand_token(sdfg, stmt.output, "out");
    stmt.statement_key =
        short_hash("stmt|" + op + "|in:" + join(key_inputs, ',') + "|out:" + key_output + "|dt:" + stmt.dtype);
    return stmt;
}

/// Builder carrying the growing module plus per-region parent links so region
/// keys can be computed in a second pass once all statements are collected.
struct Walk {
    StructuredSDFG& sdfg;
    analysis::LoopAnalysis& loop_analysis;
    Module& module;
    std::vector<int> parent_index;

    int add_region(int parent, Region&& region) {
        parent_index.push_back(parent);
        module.regions.push_back(std::move(region));
        return static_cast<int>(module.regions.size()) - 1;
    }

    void process_block(
        structured_control_flow::Block& block, int cur, const std::string& path, std::map<std::string, int>& counters
    ) {
        auto& dataflow = block.dataflow();

        // Topological order gives a deterministic, program-order traversal so that
        // library-node ordinals and statement ordering are stable across runs for git diffs.
        for (const auto* node : dataflow.topological_sort()) {
            const auto* code = dynamic_cast<const data_flow::CodeNode*>(node);
            if (code == nullptr) continue;
            if (auto tasklet = dynamic_cast<const data_flow::Tasklet*>(code)) {
                module.regions[cur]
                    .statements.push_back(make_statement(sdfg, dataflow, *tasklet, tasklet_op_name(tasklet->code())));
            } else if (auto lib = dynamic_cast<const data_flow::LibraryNode*>(code)) {
                std::string op(lib->code().value());
                int ord = counters["lib"]++;
                Region region;
                region.element_type = "library";
                region.op_class = op;
                region.structural_path = path + "/lib:" + op + "[" + std::to_string(ord) + "]";
                region.debug_info = lib->debug_info();
                region.element_id = lib->element_id();
                region.instrumentable = false;
                int idx = add_region(cur, std::move(region));
                module.regions[idx].statements.push_back(make_statement(sdfg, dataflow, *lib, op));
            }
        }
    }

    void visit_sequence(structured_control_flow::Sequence& seq, int cur, const std::string& path) {
        std::map<std::string, int> counters;
        for (size_t i = 0; i < seq.size(); ++i) {
            visit_node(seq.at(i).first, cur, path, counters);
        }
    }

    void visit_node(
        structured_control_flow::ControlFlowNode& node,
        int cur,
        const std::string& path,
        std::map<std::string, int>& counters
    ) {
        using namespace structured_control_flow;

        if (auto block = dynamic_cast<Block*>(&node)) {
            process_block(*block, cur, path, counters);
        } else if (auto loop = dynamic_cast<StructuredLoop*>(&node)) {
            std::string kind = "for";
            if (dynamic_cast<Map*>(loop))
                kind = "map";
            else if (dynamic_cast<Reduce*>(loop))
                kind = "reduce";

            int ord = counters[kind]++;
            std::string child_path = path + "/" + kind + "[" + std::to_string(ord) + "]";

            Region region;
            region.element_type = kind;
            region.structural_path = child_path;
            region.schedule_type = loop->schedule_type().value();
            region.debug_info = loop->debug_info();
            region.element_id = loop->element_id();
            region.loop_info = loop_analysis.loop_info(loop);
            region.instrumentable = loop_analysis.is_outermost_loop(loop);

            int idx = add_region(cur, std::move(region));
            visit_sequence(loop->root(), idx, child_path);
        } else if (auto while_node = dynamic_cast<While*>(&node)) {
            int ord = counters["while"]++;
            std::string child_path = path + "/while[" + std::to_string(ord) + "]";

            Region region;
            region.element_type = "while";
            region.structural_path = child_path;
            region.debug_info = while_node->debug_info();
            region.element_id = while_node->element_id();
            region.loop_info = loop_analysis.loop_info(while_node);
            region.instrumentable = loop_analysis.is_outermost_loop(while_node);

            int idx = add_region(cur, std::move(region));
            visit_sequence(while_node->root(), idx, child_path);
        } else if (auto if_else = dynamic_cast<IfElse*>(&node)) {
            int ord = counters["if"]++;
            std::string branch_base = path + "/if[" + std::to_string(ord) + "]";
            for (size_t b = 0; b < if_else->size(); ++b) {
                visit_sequence(if_else->at(b).first, cur, branch_base + "/b" + std::to_string(b));
            }
        } else if (auto seq = dynamic_cast<Sequence*>(&node)) {
            // A bare nested sequence shares its parent's structural path, so it must
            // also share the sibling ordinal counters; allocating fresh ones here would
            // restart numbering at [0] and collide with the parent's siblings.
            for (size_t i = 0; i < seq->size(); ++i) visit_node(seq->at(i).first, cur, path, counters);
        }
        // break/continue/return: not region-worthy
    }
};

std::string compute_region_key(const Module& module, const Region& region) {
    std::vector<std::string> stmt_sig;
    stmt_sig.reserve(region.statements.size());
    for (const auto& s : region.statements) stmt_sig.push_back(s.op + "/" + std::to_string(s.inputs.size()));
    std::sort(stmt_sig.begin(), stmt_sig.end());

    std::string canonical = "rgn|" + module.module_id + "|" + region.structural_path + "|" + region.element_type + "|" +
                            region.op_class + "|" + join(stmt_sig, ',');
    return short_hash(canonical);
}

} // namespace

Module RegionBuilder::build(StructuredSDFG& sdfg, const std::vector<std::pair<std::string, std::string>>& build_config) {
    Module module;
    module.name = sdfg.name();

    const auto& metadata = sdfg.metadata();
    auto it = metadata.find("source_file");
    if (it != metadata.end())
        module.source_file = it->second;
    else if (sdfg.root().debug_info().has())
        module.source_file = sdfg.root().debug_info().filename();

    module.module_id = short_hash("mod|" + module.source_file + "|" + module.name);
    module.build_config = build_config;

    analysis::AnalysisManager analysis_manager(sdfg);
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();

    // Synthetic function-root region: parent of all top-level statements.
    Region root;
    root.element_type = "function";
    root.display_key = module.name;
    root.structural_path = module.name;
    root.instrumentable = false;

    Walk walk{sdfg, loop_analysis, module, {}};
    walk.add_region(-1, std::move(root));
    walk.visit_sequence(sdfg.root(), 0, module.name);

    // Second pass: compute region keys (statements now fully collected) and parent links.
    for (size_t i = 0; i < module.regions.size(); ++i) {
        Region& region = module.regions[i];
        region.region_key = compute_region_key(module, region);
        if (region.display_key.empty()) region.display_key = region.structural_path;
        int parent = walk.parent_index[i];
        region.parent_key = parent >= 0 ? module.regions[parent].region_key : "";
    }

    return module;
}

} // namespace dcov
} // namespace sdfg
