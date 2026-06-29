#include "sdfg/dcov/region_differ.h"

#include <algorithm>
#include <cctype>
#include <map>
#include <set>
#include <sstream>

#include <nlohmann/json.hpp>

namespace sdfg {
namespace dcov {

const char* to_string(RegionStatus status) {
    switch (status) {
        case RegionStatus::Unchanged:
            return "unchanged";
        case RegionStatus::Reshaped:
            return "reshaped";
        case RegionStatus::Collapse:
            return "collapse";
        case RegionStatus::Fission:
            return "fission";
        case RegionStatus::Fusion:
            return "fusion";
        case RegionStatus::Inserted:
            return "inserted";
        case RegionStatus::Removed:
            return "removed";
    }
    return "?";
}

namespace {

struct AtomInfo {
    std::string op;
    std::string dtype;
    int count = 0;
};

/// Multiset of conserved atoms (statement_key -> op/dtype/multiplicity).
std::map<std::string, AtomInfo> atom_multiset(const Module& m) {
    std::map<std::string, AtomInfo> out;
    for (const auto& region : m.regions) {
        for (const auto& stmt : region.statements) {
            auto& a = out[stmt.statement_key];
            a.op = stmt.op;
            a.dtype = stmt.dtype;
            a.count++;
        }
    }
    return out;
}

/// A region that directly holds atoms; matching operates on these carriers so
/// that loop-nest depth (which has no atoms of its own) never confuses identity.
struct Carrier {
    std::string path;
    std::set<std::string> atoms;
};

std::vector<Carrier> carriers(const Module& m) {
    std::vector<Carrier> out;
    for (const auto& region : m.regions) {
        if (region.statements.empty()) continue;
        Carrier c;
        c.path = region.structural_path;
        for (const auto& stmt : region.statements) c.atoms.insert(stmt.statement_key);
        out.push_back(std::move(c));
    }
    return out;
}

bool subset(const std::set<std::string>& sub, const std::set<std::string>& sup) {
    return std::includes(sup.begin(), sup.end(), sub.begin(), sub.end());
}

std::string dash(const std::string& s) { return s.empty() ? "-" : s; }

std::string join(const std::vector<std::string>& xs, const char* sep) {
    std::string out;
    for (size_t i = 0; i < xs.size(); ++i) {
        if (i) out += sep;
        out += xs[i];
    }
    return out;
}

bool is_loop_type(const std::string& et) { return et == "map" || et == "for" || et == "while" || et == "reduce"; }

/// Index a module's regions by their (now unique) structural path.
std::map<std::string, const Region*> index_by_path(const Module& m) {
    std::map<std::string, const Region*> out;
    for (const auto& r : m.regions) out[r.structural_path] = &r;
    return out;
}

/// Render a structural path with inline {schedule} tags on each loop segment, and
/// report how many enclosing loop levels it has (used to detect loop collapse).
struct PathInfo {
    std::string annotated;
    int loop_depth = 0;
    std::set<std::string> schedules;
};

PathInfo annotate_path(const std::string& path, const std::map<std::string, const Region*>& by_path) {
    PathInfo info;
    std::string prefix;
    size_t start = 0;
    while (start <= path.size()) {
        size_t slash = path.find('/', start);
        std::string seg = path.substr(start, slash == std::string::npos ? std::string::npos : slash - start);
        if (!prefix.empty()) prefix += '/';
        prefix += seg;
        if (!info.annotated.empty()) info.annotated += '/';
        info.annotated += seg;
        auto it = by_path.find(prefix);
        if (it != by_path.end() && is_loop_type(it->second->element_type)) {
            info.loop_depth++;
            if (!it->second->schedule_type.empty()) {
                info.annotated += "{" + it->second->schedule_type + "}";
                info.schedules.insert(it->second->schedule_type);
            }
        }
        if (slash == std::string::npos) break;
        start = slash + 1;
    }
    return info;
}

} // namespace

ModuleDiff RegionDiffer::diff(const Module& a, const Module& b) {
    ModuleDiff d;
    d.name_a = a.name;
    d.name_b = b.name;
    d.module_id_a = a.module_id;
    d.module_id_b = b.module_id;
    d.same_source = a.module_id == b.module_id;

    // --- Atom-level diff (the conserved-computation invariant) ---
    auto ma = atom_multiset(a);
    auto mb = atom_multiset(b);

    std::set<std::string> keys;
    for (const auto& kv : ma) keys.insert(kv.first);
    for (const auto& kv : mb) keys.insert(kv.first);

    bool identical = true;
    for (const auto& k : keys) {
        auto ia = ma.find(k);
        auto ib = mb.find(k);
        int ca = ia != ma.end() ? ia->second.count : 0;
        int cb = ib != mb.end() ? ib->second.count : 0;

        AtomChange ac;
        ac.statement_key = k;
        const AtomInfo& info = ia != ma.end() ? ia->second : ib->second;
        ac.op = info.op;
        ac.dtype = info.dtype;
        ac.count_a = ca;
        ac.count_b = cb;

        if (ca > 0 && cb > 0) {
            d.atoms_unchanged.push_back(ac);
            if (ca != cb) identical = false;
        } else if (cb > 0) {
            d.atoms_added.push_back(ac);
            identical = false;
        } else {
            d.atoms_removed.push_back(ac);
            identical = false;
        }
    }
    d.computation_identical = identical;

    // --- Region-level correspondence via atom-set algebra on carriers ---
    auto ra = carriers(a);
    auto rb = carriers(b);
    std::vector<bool> used_a(ra.size(), false);
    std::vector<bool> used_b(rb.size(), false);

    // 1. Exact atom-set match -> unchanged (same path) or reshaped (moved).
    for (size_t i = 0; i < ra.size(); ++i) {
        if (used_a[i]) continue;
        for (size_t j = 0; j < rb.size(); ++j) {
            if (used_b[j]) continue;
            if (ra[i].atoms == rb[j].atoms) {
                used_a[i] = used_b[j] = true;
                RegionMatch m;
                m.status = ra[i].path == rb[j].path ? RegionStatus::Unchanged : RegionStatus::Reshaped;
                m.a_paths = {ra[i].path};
                m.b_paths = {rb[j].path};
                d.region_matches.push_back(std::move(m));
                break;
            }
        }
    }

    // 2. Fission: one A carrier == disjoint union of >=2 unmatched B carriers.
    for (size_t i = 0; i < ra.size(); ++i) {
        if (used_a[i]) continue;
        std::vector<size_t> parts;
        std::set<std::string> uni;
        bool disjoint = true;
        for (size_t j = 0; j < rb.size(); ++j) {
            if (used_b[j] || rb[j].atoms.empty()) continue;
            if (subset(rb[j].atoms, ra[i].atoms)) {
                for (const auto& k : rb[j].atoms)
                    if (!uni.insert(k).second) disjoint = false;
                parts.push_back(j);
            }
        }
        if (parts.size() >= 2 && disjoint && uni == ra[i].atoms) {
            RegionMatch m;
            m.status = RegionStatus::Fission;
            m.a_paths = {ra[i].path};
            for (auto j : parts) {
                used_b[j] = true;
                m.b_paths.push_back(rb[j].path);
            }
            used_a[i] = true;
            d.region_matches.push_back(std::move(m));
        }
    }

    // 3. Fusion: one B carrier == disjoint union of >=2 unmatched A carriers.
    for (size_t j = 0; j < rb.size(); ++j) {
        if (used_b[j]) continue;
        std::vector<size_t> parts;
        std::set<std::string> uni;
        bool disjoint = true;
        for (size_t i = 0; i < ra.size(); ++i) {
            if (used_a[i] || ra[i].atoms.empty()) continue;
            if (subset(ra[i].atoms, rb[j].atoms)) {
                for (const auto& k : ra[i].atoms)
                    if (!uni.insert(k).second) disjoint = false;
                parts.push_back(i);
            }
        }
        if (parts.size() >= 2 && disjoint && uni == rb[j].atoms) {
            RegionMatch m;
            m.status = RegionStatus::Fusion;
            m.b_paths = {rb[j].path};
            for (auto i : parts) {
                used_a[i] = true;
                m.a_paths.push_back(ra[i].path);
            }
            used_b[j] = true;
            d.region_matches.push_back(std::move(m));
        }
    }

    // 4. Leftovers whose atoms are entirely absent on the other side.
    for (size_t i = 0; i < ra.size(); ++i) {
        if (used_a[i]) continue;
        bool any_in_b = false;
        for (const auto& k : ra[i].atoms)
            if (mb.count(k)) {
                any_in_b = true;
                break;
            }
        if (!any_in_b) {
            RegionMatch m;
            m.status = RegionStatus::Removed;
            m.a_paths = {ra[i].path};
            d.region_matches.push_back(std::move(m));
            used_a[i] = true;
        }
    }
    for (size_t j = 0; j < rb.size(); ++j) {
        if (used_b[j]) continue;
        bool any_in_a = false;
        for (const auto& k : rb[j].atoms)
            if (ma.count(k)) {
                any_in_a = true;
                break;
            }
        if (!any_in_a) {
            RegionMatch m;
            m.status = RegionStatus::Inserted;
            m.b_paths = {rb[j].path};
            d.region_matches.push_back(std::move(m));
            used_b[j] = true;
        }
    }

    // Annotate paths with schedule types and reclassify depth-reducing reshapes
    // as loop collapses (clearer signal for a performance engineer).
    auto by_path_a = index_by_path(a);
    auto by_path_b = index_by_path(b);
    for (auto& m : d.region_matches) {
        int depth_a = 0;
        int depth_b = 0;
        std::set<std::string> sched_a;
        std::set<std::string> sched_b;
        for (const auto& p : m.a_paths) {
            auto info = annotate_path(p, by_path_a);
            m.a_annotated.push_back(info.annotated);
            depth_a = std::max(depth_a, info.loop_depth);
            sched_a.insert(info.schedules.begin(), info.schedules.end());
        }
        for (const auto& p : m.b_paths) {
            auto info = annotate_path(p, by_path_b);
            m.b_annotated.push_back(info.annotated);
            depth_b = std::max(depth_b, info.loop_depth);
            sched_b.insert(info.schedules.begin(), info.schedules.end());
        }
        if (m.status == RegionStatus::Reshaped && depth_b < depth_a) m.status = RegionStatus::Collapse;
        // Schedule change is only well-defined for 1:1 carrier correspondences.
        if (m.a_paths.size() == 1 && m.b_paths.size() == 1) m.schedule_changed = sched_a != sched_b;
    }

    // Deterministic ordering for git-friendly output.
    std::sort(d.region_matches.begin(), d.region_matches.end(), [](const RegionMatch& x, const RegionMatch& y) {
        if (x.status != y.status) return static_cast<int>(x.status) < static_cast<int>(y.status);
        if (x.schedule_changed != y.schedule_changed)
            return static_cast<int>(x.schedule_changed) < static_cast<int>(y.schedule_changed);
        if (x.a_paths != y.a_paths) return x.a_paths < y.a_paths;
        return x.b_paths < y.b_paths;
    });

    return d;
}

// ---------------------------------------------------------------------------
// Text report
// ---------------------------------------------------------------------------

namespace {

std::string to_upper(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return s;
}

/// Compose the change label, treating schedule change as an orthogonal tag:
/// e.g. "unchanged", "scheduled", "collapse", "collapse|scheduled".
std::string composite_label(RegionStatus status, bool schedule_changed) {
    std::string primary = (status == RegionStatus::Unchanged && schedule_changed) ? "scheduled" : to_string(status);
    if (schedule_changed && (status == RegionStatus::Reshaped || status == RegionStatus::Collapse))
        primary += "|scheduled";
    return primary;
}

std::string pad(const std::string& s, size_t w) { return s.size() >= w ? s : s + std::string(w - s.size(), ' '); }

} // namespace

std::string DiffReportSerializer::serialize(const ModuleDiff& diff) {
    std::ostringstream os;
    this->serialize(diff, os);
    return os.str();
}

void DiffReportSerializer::serialize(const ModuleDiff& diff, std::ostream& os) {
    os << "DIFF " << diff.name_a << " -> " << diff.name_b << "\n";
    os << "  source: " << (diff.same_source ? "same" : "different") << " (" << diff.module_id_a << " -> "
       << diff.module_id_b << ")\n";
    os << "  computation: " << (diff.computation_identical ? "IDENTICAL" : "CHANGED") << "\n";
    os << "  atoms: unchanged=" << diff.atoms_unchanged.size() << "  added=" << diff.atoms_added.size()
       << "  removed=" << diff.atoms_removed.size() << "\n";

    for (const auto& a : diff.atoms_removed)
        os << "  [-] " << a.op << "  " << dash(a.dtype) << "  " << a.statement_key << "  (x" << a.count_a << ")\n";
    for (const auto& a : diff.atoms_added)
        os << "  [+] " << a.op << "  " << dash(a.dtype) << "  " << a.statement_key << "  (x" << a.count_b << ")\n";
    for (const auto& a : diff.atoms_unchanged)
        if (a.count_a != a.count_b)
            os << "  [~] " << a.op << "  " << dash(a.dtype) << "  " << a.statement_key << "  (" << a.count_a << " -> "
               << a.count_b << ")\n";

    os << "REGIONS\n";
    size_t omitted = 0;
    for (const auto& m : diff.region_matches) {
        // A region that is structurally and schedule-wise unchanged is noise in a diff.
        if (m.status == RegionStatus::Unchanged && !m.schedule_changed) {
            ++omitted;
            continue;
        }
        const auto& a = m.a_annotated.empty() ? m.a_paths : m.a_annotated;
        const auto& b = m.b_annotated.empty() ? m.b_paths : m.b_annotated;
        std::string label = pad(to_upper(composite_label(m.status, m.schedule_changed)), 20);
        switch (m.status) {
            case RegionStatus::Unchanged:
                os << "  " << label << a.front() << "  ->  " << b.front() << "\n";
                break;
            case RegionStatus::Reshaped:
            case RegionStatus::Collapse:
                os << "  " << label << a.front() << "  ->  " << b.front() << "\n";
                break;
            case RegionStatus::Fission:
                os << "  " << label << a.front() << "  ->  " << join(b, ", ") << "\n";
                break;
            case RegionStatus::Fusion:
                os << "  " << label << join(a, ", ") << "  ->  " << b.front() << "\n";
                break;
            case RegionStatus::Inserted:
                os << "  " << label << b.front() << "\n";
                break;
            case RegionStatus::Removed:
                os << "  " << label << a.front() << "\n";
                break;
        }
    }
    if (omitted) os << "  (" << omitted << " unchanged regions omitted)\n";
}

// ---------------------------------------------------------------------------
// JSON report
// ---------------------------------------------------------------------------

std::string DiffJsonSerializer::serialize(const ModuleDiff& diff) {
    std::ostringstream os;
    this->serialize(diff, os);
    return os.str();
}

void DiffJsonSerializer::serialize(const ModuleDiff& diff, std::ostream& os) {
    auto atom_json = [](const AtomChange& a) {
        return nlohmann::json{
            {"statement_key", a.statement_key},
            {"op", a.op},
            {"dtype", a.dtype},
            {"count_a", a.count_a},
            {"count_b", a.count_b},
        };
    };

    nlohmann::json j;
    j["name_a"] = diff.name_a;
    j["name_b"] = diff.name_b;
    j["module_id_a"] = diff.module_id_a;
    j["module_id_b"] = diff.module_id_b;
    j["same_source"] = diff.same_source;
    j["computation_identical"] = diff.computation_identical;

    j["atoms_unchanged"] = nlohmann::json::array();
    for (const auto& a : diff.atoms_unchanged) j["atoms_unchanged"].push_back(atom_json(a));
    j["atoms_added"] = nlohmann::json::array();
    for (const auto& a : diff.atoms_added) j["atoms_added"].push_back(atom_json(a));
    j["atoms_removed"] = nlohmann::json::array();
    for (const auto& a : diff.atoms_removed) j["atoms_removed"].push_back(atom_json(a));

    j["region_matches"] = nlohmann::json::array();
    for (const auto& m : diff.region_matches) {
        j["region_matches"].push_back({
            {"status", to_string(m.status)},
            {"label", composite_label(m.status, m.schedule_changed)},
            {"schedule_changed", m.schedule_changed},
            {"a_paths", m.a_paths},
            {"b_paths", m.b_paths},
            {"a_annotated", m.a_annotated},
            {"b_annotated", m.b_annotated},
        });
    }

    os << j.dump(2) << "\n";
}

} // namespace dcov
} // namespace sdfg
