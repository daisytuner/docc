#pragma once

#include <ostream>
#include <string>
#include <vector>

#include "sdfg/dcov/region.h"

namespace sdfg {
namespace dcov {

/**
 * @brief Multiplicity change of a single conserved statement (atom) between two
 *        module models.
 *
 * Atoms are matched by @ref Statement::statement_key, which is invariant under
 * horizontal transformations (fission/fusion/scheduling/tiling). Whether an atom
 * was conserved is therefore independent of the loop nest it lives in.
 */
struct AtomChange {
    std::string statement_key;
    std::string op;
    std::string dtype;
    int count_a = 0; ///< Multiplicity in module A
    int count_b = 0; ///< Multiplicity in module B
};

/// How a region's atom set evolved from module A to module B.
enum class RegionStatus {
    Unchanged, ///< Same atom set, same structural path
    Reshaped, ///< Same atom set, different structural path (rescheduled/moved)
    Collapse, ///< Same atom set, fewer enclosing loop levels (loop collapse/flatten)
    Fission, ///< One A region's atoms split across several B regions
    Fusion, ///< Several A regions' atoms merged into one B region
    Inserted, ///< B region whose atoms have no counterpart in A
    Removed, ///< A region whose atoms have no counterpart in B
};

const char* to_string(RegionStatus status);

/**
 * @brief A correspondence between region(s) in A and region(s) in B, classified
 *        by the relationship between their atom sets.
 */
struct RegionMatch {
    RegionStatus status;
    bool schedule_changed = false; ///< Loop schedule type(s) differ between A and B (1:1 matches only)
    std::vector<std::string> a_paths; ///< structural_path of involved A regions
    std::vector<std::string> b_paths; ///< structural_path of involved B regions
    std::vector<std::string> a_annotated; ///< A paths with inline {schedule} annotations
    std::vector<std::string> b_annotated; ///< B paths with inline {schedule} annotations
};

/**
 * @brief The full diff between two module models.
 */
struct ModuleDiff {
    std::string name_a;
    std::string name_b;
    std::string module_id_a;
    std::string module_id_b;

    bool same_source = false; ///< module_id_a == module_id_b
    bool computation_identical = false; ///< atom multisets are equal

    std::vector<AtomChange> atoms_unchanged; ///< present in both (count_a>0 && count_b>0)
    std::vector<AtomChange> atoms_added; ///< present only in B
    std::vector<AtomChange> atoms_removed; ///< present only in A

    std::vector<RegionMatch> region_matches;
};

/**
 * @brief Computes the semantic diff between two static region models.
 */
class RegionDiffer {
public:
    ModuleDiff diff(const Module& a, const Module& b);
};

/**
 * @brief Renders a @ref ModuleDiff as a human-readable, git-friendly report.
 */
class DiffReportSerializer {
public:
    void serialize(const ModuleDiff& diff, std::ostream& os);
    std::string serialize(const ModuleDiff& diff);
};

/**
 * @brief Renders a @ref ModuleDiff as canonical JSON.
 */
class DiffJsonSerializer {
public:
    void serialize(const ModuleDiff& diff, std::ostream& os);
    std::string serialize(const ModuleDiff& diff);
};

} // namespace dcov
} // namespace sdfg
