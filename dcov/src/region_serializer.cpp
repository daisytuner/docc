#include "sdfg/dcov/region_serializer.h"

#include <cmath>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <sstream>

#include "sdfg/analysis/loop_analysis.h"

namespace sdfg {
namespace dcov {

std::string RegionSerializer::serialize(const Module& module) {
    std::ostringstream os;
    this->serialize(module, os);
    return os.str();
}

namespace {

std::string join(const std::vector<std::string>& xs, char sep) {
    std::string out;
    for (size_t i = 0; i < xs.size(); ++i) {
        if (i) out += sep;
        out += xs[i];
    }
    return out;
}

std::string dash(const std::string& s) { return s.empty() ? "-" : s; }

/// Render a metric value compactly and deterministically: integral values as
/// integers, fractional values with up to 3 decimals (trailing zeros trimmed).
std::string fmt_num(double v) {
    if (std::isfinite(v) && v == std::floor(v) && std::fabs(v) < 1e15) return std::to_string(static_cast<long long>(v));
    std::ostringstream o;
    o << std::fixed << std::setprecision(3) << v;
    std::string s = o.str();
    s.erase(s.find_last_not_of('0') + 1, std::string::npos);
    if (!s.empty() && s.back() == '.') s.pop_back();
    return s;
}

nlohmann::json debug_info_to_json(const DebugInfo& di) {
    if (!di.has()) return nullptr;
    return nlohmann::json{
        {"file", di.filename()},
        {"function", di.function()},
        {"start_line", di.start_line()},
        {"start_column", di.start_column()},
        {"end_line", di.end_line()},
        {"end_column", di.end_column()},
    };
}

} // namespace

void DcovRecordSerializer::serialize(const Module& module, std::ostream& os) {
    os << "MOD " << module.name << "  src=" << dash(module.source_file) << "  module_id=" << module.module_id << "\n";

    if (!module.build_config.empty()) {
        os << "CFG";
        for (const auto& kv : module.build_config) os << " " << kv.first << "=" << kv.second;
        os << "\n";
    }

    for (const auto& region : module.regions) {
        os << "RGN " << region.region_key << "  " << region.display_key << "  parent=" << dash(region.parent_key)
           << "  instr=" << (region.instrumentable ? 1 : 0) << "\n";

        os << "  TYP " << region.element_type << "  op=" << dash(region.op_class);
        if (region.element_id != 0) os << "  eid=" << region.element_id;
        if (region.debug_info.has()) {
            os << "  src=" << region.debug_info.filename() << ":" << region.debug_info.start_line() << "-"
               << region.debug_info.end_line();
        }
        os << "\n";

        if (region.loop_info) {
            const auto& l = *region.loop_info;
            os << "  LOOP level=" << l.loop_level << "  depth=" << l.max_depth << "  loops=" << l.num_loops
               << "  maps=" << l.num_maps << "  fors=" << l.num_fors << "  whiles=" << l.num_whiles
               << "  pn=" << (l.is_perfectly_nested ? 1 : 0) << "  pp=" << (l.is_perfectly_parallel ? 1 : 0)
               << "  sched=" << dash(region.schedule_type) << "\n";
        }

        for (const auto& stmt : region.statements) {
            os << "  STMT " << stmt.statement_key << "  " << stmt.op << "  " << dash(stmt.dtype)
               << "  in=" << dash(join(stmt.inputs, ',')) << "  out=" << dash(stmt.output) << "\n";
        }

        if (region.profile) {
            const auto& p = *region.profile;
            os << "  PROF runtime_us=" << fmt_num(p.runtime_us) << "  invocations=" << p.invocations
               << "  target=" << dash(p.target_type) << "\n";
            for (const auto& m : p.metrics) {
                os << "  METRIC " << m.name << "  mean=" << fmt_num(m.mean) << "  min=" << fmt_num(m.min)
                   << "  max=" << fmt_num(m.max) << "  count=" << m.count << "\n";
            }
        }

        if (region.has_arg_capture) os << "  CAPTURE " << join(region.arg_captures, ',') << "\n";

        os << "END\n";
    }
}

void DcovJsonSerializer::serialize(const Module& module, std::ostream& os) {
    nlohmann::json j;
    j["name"] = module.name;
    j["source_file"] = module.source_file;
    j["module_id"] = module.module_id;

    j["build_config"] = nlohmann::json::object();
    for (const auto& kv : module.build_config) j["build_config"][kv.first] = kv.second;

    j["regions"] = nlohmann::json::array();
    for (const auto& region : module.regions) {
        nlohmann::json rj;
        rj["region_key"] = region.region_key;
        rj["display_key"] = region.display_key;
        rj["parent_key"] = region.parent_key;
        rj["element_type"] = region.element_type;
        rj["op_class"] = region.op_class;
        rj["schedule_type"] = region.schedule_type;
        rj["instrumentable"] = region.instrumentable;
        rj["structural_path"] = region.structural_path;
        rj["debug_info"] = debug_info_to_json(region.debug_info);
        rj["loop_info"] = region.loop_info ? analysis::loop_info_to_json(*region.loop_info) : nlohmann::json(nullptr);

        rj["element_id"] = region.element_id;

        if (region.profile) {
            const auto& p = *region.profile;
            nlohmann::json pj;
            pj["runtime_us"] = p.runtime_us;
            pj["invocations"] = p.invocations;
            pj["target_type"] = p.target_type;
            pj["metrics"] = nlohmann::json::array();
            for (const auto& m : p.metrics) {
                pj["metrics"].push_back({
                    {"name", m.name},
                    {"mean", m.mean},
                    {"min", m.min},
                    {"max", m.max},
                    {"variance", m.variance},
                    {"count", m.count},
                });
            }
            rj["profile"] = std::move(pj);
        } else {
            rj["profile"] = nullptr;
        }

        rj["has_arg_capture"] = region.has_arg_capture;
        rj["arg_captures"] = region.arg_captures;

        rj["statements"] = nlohmann::json::array();
        for (const auto& stmt : region.statements) {
            rj["statements"].push_back({
                {"statement_key", stmt.statement_key},
                {"op", stmt.op},
                {"dtype", stmt.dtype},
                {"inputs", stmt.inputs},
                {"output", stmt.output},
            });
        }

        j["regions"].push_back(rj);
    }

    os << j.dump(2) << "\n";
}

} // namespace dcov
} // namespace sdfg
