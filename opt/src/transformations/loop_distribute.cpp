#include "sdfg/transformations/loop_distribute.h"

#include "sdfg/analysis/data_dependency_analysis.h"
#include "sdfg/analysis/loop_carried_dependency_analysis.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/deepcopy/structured_sdfg_deep_copy.h"

namespace sdfg {
namespace transformations {

LoopDistribute::
    LoopDistribute(structured_control_flow::StructuredLoop& loop, structured_control_flow::ControlFlowNode& child)
    : loop_(loop), child_(child) {};

LoopDistribute::LoopDistribute(structured_control_flow::StructuredLoop& loop)
    : LoopDistribute(loop, loop.root().at(0).first) {};

std::string LoopDistribute::name() const { return "LoopDistribute"; };

namespace {

// Walk parent_scope chain from `user`'s scope up; return true if `subtree` is
// reached. Mirrors LoopCarriedDependencyAnalysis::pairs_between's containment.
bool user_in_subtree(
    analysis::User* user,
    const structured_control_flow::ControlFlowNode& subtree,
    analysis::ScopeAnalysis& scope_analysis
) {
    auto* scope = analysis::Users::scope(user);
    while (scope != nullptr) {
        if (scope == &subtree) return true;
        scope = scope_analysis.parent_scope(scope);
    }
    return false;
}

} // namespace

bool LoopDistribute::can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    // Criterion: Loop must have at least 2 children to distribute.
    auto& body = this->loop_.root();
    if (body.size() < 2) {
        return false;
    }

    // Criterion: child_ must be in loop body; record its index.
    size_t child_idx = body.size();
    for (size_t i = 0; i < body.size(); i++) {
        if (&body.at(i).first == &this->child_) {
            child_idx = i;
            break;
        }
    }
    if (child_idx == body.size()) {
        return false;
    }

    // Criterion: Transitions must not have assignments.
    for (size_t i = 0; i < body.size(); i++) {
        auto& transition = body.at(i).second;
        if (!transition.assignments().empty()) {
            return false;
        }
    }

    // Criterion: subset-aware loop-carried dependence check.
    //
    // apply() performs an in-place 3-way split of `loop_` preserving program
    // order:
    //     prefix loop  : children [0 .. child_idx)
    //     center loop  : { child_ }   (the original loop_)
    //     suffix loop  : children (child_idx .. body.size())
    // Empty pieces are not materialized.
    //
    // For correctness we examine each loop-carried (writer, reader) pair from
    // LoopCarriedDependencyAnalysis. After the split, the body is partitioned
    // into THREE destination groups (prefix / center / suffix) — note that
    // siblings inside the same destination group remain inside ONE shared
    // loop body, so a (writer, reader) pair whose users land in the same
    // destination group is NOT cross-loop after distribution and survives
    // unchanged. Only pairs that straddle two distinct destination groups
    // get split apart.
    //
    // For pairs that straddle destination groups:
    //   - WAW pairs are always safe: program-order between groups is
    //     preserved, so the surviving last-writer to any cell is unchanged.
    //   - RAW pairs are safe iff writer and reader stay in the same group.
    //     A cross-group RAW would force the reader to see a different write
    //     after the groups are serialized.
    auto& lcd = analysis_manager.get<analysis::LoopCarriedDependencyAnalysis>();
    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    if (!lcd.available(this->loop_)) {
        return false;
    }

    auto indvar_name = this->loop_.indvar()->get_name();

    // Classify a user into its piece (direct child index in body) and from
    // there into its destination group: 0 = prefix, 1 = center, 2 = suffix.
    auto piece_of = [&](analysis::User* u) -> size_t {
        for (size_t i = 0; i < body.size(); i++) {
            if (user_in_subtree(u, body.at(i).first, scope_analysis)) {
                return i;
            }
        }
        return body.size();
    };
    auto group_of = [&](size_t piece) -> int {
        if (piece == body.size()) return -1;
        if (piece < child_idx) return 0; // prefix
        if (piece == child_idx) return 1; // center
        return 2; // suffix
    };

    // (1) Cross-iteration loop-carried pairs (from LCDA).
    std::unordered_set<std::string> lc_containers;
    for (auto& pair : lcd.pairs(this->loop_)) {
        lc_containers.insert(pair.writer->container());
    }
    for (auto& pair : lcd.pairs(this->loop_)) {
        const std::string& container = pair.writer->container();
        if (container == indvar_name) continue;
        if (pair.deltas.empty) continue; // defensive; pairs() should not store empties

        size_t w_piece = piece_of(pair.writer);
        size_t r_piece = piece_of(pair.reader);
        if (w_piece == body.size() || r_piece == body.size()) {
            // User not localizable to a direct child piece -- be conservative.
            return false;
        }
        int w_group = group_of(w_piece);
        int r_group = group_of(r_piece);
        if (w_group == r_group) continue; // intra-group, distribution preserves it

        // Cross-group pair.
        if (pair.type == analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE) {
            continue; // WAW always safe under program-order-preserving split
        }

        // Cross-group RAW: unsafe. After distribution the writer's piece runs
        // to completion across all iterations before the reader's piece, so
        // the reader sees a different (later) value than in the fused order.
        return false;
    }

    // (2) Intra-iteration cross-piece RAW on scalars (from DDA).
    //
    // LCDA captures only loop-carried (cross-iteration) flow. A reader that
    // is satisfied within the same iteration by a writer in a different piece
    // becomes loop-carried after distribution: the writer's piece runs all
    // iterations before the reader's piece.
    //
    // For array containers this is generally safe: if no LC pair exists on
    // the container the writer's per-iter footprint is disjoint, so the
    // intra-iter cell flow is preserved. If LC pairs DO exist on the array,
    // LCDA still distinguishes which specific (writer, reader) pairs flow
    // across iterations via delta-set computation, so the LC pass above
    // catches the unsafe ones.
    //
    // For SCALAR containers (empty subset) every access is to the same single
    // cell. If the scalar has any LC dependence in this loop (i.e. it is
    // overwritten / re-read across iters rather than being a loop-invariant
    // constant), an intra-iter cross-piece RAW becomes unsafe under
    // distribution: the reader's piece would see the writer's last-iteration
    // value instead of the matching iteration's value. DDA's reaching-defs
    // also drop these cross-piece reads from `ue_reads` (they are killed
    // intra-iter by the same-iter writer in the middle piece), so LCDA
    // never sees them as loop-carried -- we must catch them directly.
    auto& dda = analysis_manager.get<analysis::DataDependencyAnalysis>();
    auto& users = analysis_manager.get<analysis::Users>();
    analysis::UsersView body_view(users, this->loop_.root());
    for (auto* writer : body_view.writes()) {
        if (writer->container() == indvar_name) continue;
        if (lc_containers.find(writer->container()) == lc_containers.end()) {
            continue; // no LC dep on this container
        }
        if (!writer->subsets().empty()) {
            // Array-typed access: per-pair LC analysis above is precise enough.
            bool any_nonempty = false;
            for (auto& s : writer->subsets()) {
                if (!s.empty()) {
                    any_nonempty = true;
                    break;
                }
            }
            if (any_nonempty) continue;
        }
        size_t w_piece = piece_of(writer);
        if (w_piece == body.size()) continue;
        for (auto* reader : dda.defines(*writer)) {
            size_t r_piece = piece_of(reader);
            if (r_piece == body.size()) continue;
            if (w_piece == r_piece) continue;
            int w_group = group_of(w_piece);
            int r_group = group_of(r_piece);
            if (w_group == r_group) continue;
            // Cross-group intra-iter scalar RAW: unsafe under distribution.
            return false;
        }
    }

    return true;
};

void LoopDistribute::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& sdfg = builder.subject();

    auto indvar = this->loop_.indvar();
    auto condition = this->loop_.condition();
    auto update = this->loop_.update();
    auto init = this->loop_.init();

    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto parent = static_cast<structured_control_flow::Sequence*>(scope_analysis.parent_scope(&this->loop_));

    structured_control_flow::ScheduleType schedule_type = structured_control_flow::ScheduleType_Sequential::create();
    if (auto map_stmt = dynamic_cast<structured_control_flow::Map*>(&this->loop_)) {
        schedule_type = map_stmt->schedule_type();
    }

    auto& body = this->loop_.root();

    // Locate child_'s current index (guaranteed present by can_be_applied).
    size_t child_idx = body.size();
    for (size_t i = 0; i < body.size(); i++) {
        if (&body.at(i).first == &this->child_) {
            child_idx = i;
            break;
        }
    }

    // Create suffix loop FIRST (added after `loop_`) so prefix creation does
    // not shift body indices we still need.
    if (child_idx + 1 < body.size()) {
        auto& suffix_loop = builder.add_map_after(
            *parent, this->loop_, indvar, condition, init, update, schedule_type, {}, this->loop_.debug_info()
        );
        // Move all children after child_ into the suffix loop, preserving
        // their relative order. Each move shifts subsequent indices down by 1.
        size_t to_move = body.size() - (child_idx + 1);
        for (size_t i = 0; i < to_move; i++) {
            builder.move_child(body, child_idx + 1, suffix_loop.root());
        }
        std::string suffix_indvar = builder.find_new_name(indvar->get_name());
        builder.add_container(suffix_indvar, sdfg.type(indvar->get_name()));
        suffix_loop.replace(indvar, symbolic::symbol(suffix_indvar));
    }

    // Create prefix loop (added before `loop_`).
    if (child_idx > 0) {
        auto& prefix_loop = builder.add_map_before(
            *parent, this->loop_, indvar, condition, init, update, schedule_type, {}, this->loop_.debug_info()
        );
        // Move all children before child_ into the prefix loop. Always move
        // child at index 0 since each move shifts subsequent indices down.
        for (size_t i = 0; i < child_idx; i++) {
            builder.move_child(body, 0, prefix_loop.root());
        }
        std::string prefix_indvar = builder.find_new_name(indvar->get_name());
        builder.add_container(prefix_indvar, sdfg.type(indvar->get_name()));
        prefix_loop.replace(indvar, symbolic::symbol(prefix_indvar));
    }

    analysis_manager.invalidate_all();
};

void LoopDistribute::to_json(nlohmann::json& j) const {
    std::string loop_type;
    if (dynamic_cast<structured_control_flow::For*>(&loop_)) {
        loop_type = "for";
    } else if (dynamic_cast<structured_control_flow::Map*>(&loop_)) {
        loop_type = "map";
    } else {
        throw std::runtime_error("Unsupported loop type for serialization of loop: " + loop_.indvar()->get_name());
    }

    j["transformation_type"] = this->name();
    j["subgraph"] = {{"0", {{"element_id", this->loop_.element_id()}, {"type", loop_type}}}};
    j["transformation_type"] = this->name();
};

LoopDistribute LoopDistribute::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc) {
    auto loop_id = desc["subgraph"]["0"]["element_id"].get<size_t>();
    auto element = builder.find_element_by_id(loop_id);
    if (element == nullptr) {
        throw InvalidTransformationDescriptionException("Element with ID " + std::to_string(loop_id) + " not found.");
    }
    auto loop = dynamic_cast<structured_control_flow::StructuredLoop*>(element);
    if (loop == nullptr) {
        throw InvalidTransformationDescriptionException(
            "Element with ID " + std::to_string(loop_id) + " is not a StructuredLoop."
        );
    }

    return LoopDistribute(*loop);
};

} // namespace transformations
} // namespace sdfg
