#include "sdfg/passes/normalization/stride_minimization.h"

#include <sdfg/analysis/loop_analysis.h>
#include <sdfg/transformations/loop_interchange.h>

namespace sdfg {
namespace passes {
namespace normalization {

StrideMinimization::StrideMinimization() : Pass() {};

bool StrideMinimization::is_admissible(
    std::vector<std::string>& current, std::vector<std::string>& target, std::unordered_set<std::string>& allowed_swaps
) {
    std::vector<size_t> permutation_indices;
    for (auto& p : current) {
        auto it = std::find(target.begin(), target.end(), p);
        permutation_indices.push_back(std::distance(target.begin(), it));
    }
    for (size_t i = 0; i < permutation_indices.size(); i++) {
        for (size_t j = 0; j < permutation_indices.size() - i - 1; j++) {
            if (permutation_indices[j] > permutation_indices[j + 1]) {
                std::string swap = target[permutation_indices[j]] + "_" + target[permutation_indices[j + 1]];
                if (allowed_swaps.find(swap) == allowed_swaps.end()) {
                    return false;
                }
                size_t tmp = permutation_indices[j];
                permutation_indices[j] = permutation_indices[j + 1];
                permutation_indices[j + 1] = tmp;
            }
        }
    }

    return true;
};

std::unordered_set<std::string> StrideMinimization::allowed_swaps(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    std::vector<structured_control_flow::ControlFlowNode*>& nested_loops
) {
    std::unordered_set<std::string> allowed_swaps;
    for (size_t i = 0; i < nested_loops.size() - 1; i++) {
        if (dynamic_cast<structured_control_flow::While*>(nested_loops.at(i)) ||
            dynamic_cast<structured_control_flow::While*>(nested_loops.at(i + 1))) {
            continue;
        } else {
            auto first_loop = dynamic_cast<structured_control_flow::StructuredLoop*>(nested_loops.at(i));
            auto second_loop = dynamic_cast<structured_control_flow::StructuredLoop*>(nested_loops.at(i + 1));
            transformations::LoopInterchange loop_interchange(*first_loop, *second_loop);
            if (loop_interchange.can_be_applied(builder, analysis_manager)) {
                allowed_swaps.insert(first_loop->indvar()->get_name() + "_" + second_loop->indvar()->get_name());
            }
        }
    }
    return allowed_swaps;
};

std::pair<bool, std::vector<std::string>> StrideMinimization::can_be_applied(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    std::vector<structured_control_flow::ControlFlowNode*>& nested_loops
) {
    if (nested_loops.size() < 2) {
        return {false, {}};
    }

    size_t while_loops = 0;
    std::vector<std::string> permutation;
    for (auto& loop : nested_loops) {
        if (!dynamic_cast<structured_control_flow::StructuredLoop*>(loop)) {
            permutation.push_back("WHILE_" + std::to_string(while_loops++));
        } else {
            auto for_loop = dynamic_cast<structured_control_flow::StructuredLoop*>(loop);
            permutation.push_back(for_loop->indvar()->get_name());
        }
    }

    auto admissible_swaps = allowed_swaps(builder, analysis_manager, nested_loops);

    // Collect all memory accesses of body
    structured_control_flow::ControlFlowNode* nested_root = nullptr;
    if (auto for_loop = dynamic_cast<structured_control_flow::StructuredLoop*>(nested_loops.at(0))) {
        nested_root = &for_loop->root();
    } else if (auto while_loop = dynamic_cast<structured_control_flow::While*>(nested_loops.at(0))) {
        nested_root = &while_loop->root();
    } else {
        assert(false);
    }
    auto& all_users = analysis_manager.get<analysis::Users>();
    analysis::UsersView users(all_users, *nested_root);

    // Collect all memory accesses
    std::vector<data_flow::Subset> subsets;
    for (auto& use : users.reads()) {
        auto element = use->element();
        if (!dynamic_cast<data_flow::AccessNode*>(element)) {
            continue;
        }
        auto access = dynamic_cast<data_flow::AccessNode*>(element);
        auto& graph = access->get_parent();
        for (auto& oedge : graph.out_edges(*access)) {
            auto& subset = oedge.subset();

            // Filter the subset for only dimensions using the permutation
            data_flow::Subset filtered_subset;
            for (auto& dim : subset) {
                for (auto& perm : permutation) {
                    if (symbolic::uses(dim, perm)) {
                        filtered_subset.push_back(dim);
                        break;
                    }
                }
            }
            if (filtered_subset.empty()) {
                continue;
            }

            subsets.push_back(filtered_subset);
        }
    }
    for (auto& use : users.writes()) {
        auto element = use->element();
        if (!dynamic_cast<data_flow::AccessNode*>(element)) {
            continue;
        }
        auto access = dynamic_cast<data_flow::AccessNode*>(element);
        auto& graph = access->get_parent();
        for (auto& iedge : graph.in_edges(*access)) {
            auto& subset = iedge.subset();

            // Filter the subset for only dimensions using the permutation
            data_flow::Subset filtered_subset;
            for (auto& dim : subset) {
                for (auto& perm : permutation) {
                    if (symbolic::uses(dim, perm)) {
                        filtered_subset.push_back(dim);
                        break;
                    }
                }
            }
            if (filtered_subset.empty()) {
                continue;
            }

            subsets.push_back(filtered_subset);
        }
    }

    // No memory accesses to optimize
    if (subsets.empty()) {
        return {false, {}};
    }

    // Test all permutations (must start from a sorted sequence)
    std::vector<std::string> current = permutation;
    std::sort(current.begin(), current.end());

    std::vector<std::string> best_permutation;
    std::vector<std::map<size_t, size_t>> best_scores;
    do {
        if (!is_admissible(permutation, current, admissible_swaps)) {
            continue;
        }

        // Init scores per dimension
        std::vector<std::map<size_t, size_t>> scores;
        for (size_t i = 0; i < current.size(); i++) {
            scores.push_back({});
        }

        // Compute skipped dimensions and sum up
        for (auto& subset : subsets) {
            std::vector<size_t> strides;
            for (auto& var : current) {
                bool found = false;
                for (size_t j = 0; j < subset.size(); j++) {
                    auto& dim = subset.at(j);
                    if (symbolic::uses(dim, var)) {
                        strides.push_back(subset.size() - j);
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    strides.push_back(0);
                }
            }

            for (size_t i = 0; i < current.size(); i++) {
                auto& score_dim = scores[i];
                if (score_dim.find(strides[i]) == score_dim.end()) {
                    score_dim[strides[i]] = 1;
                } else {
                    score_dim[strides[i]]++;
                }
            }
        }

        // Compare scores
        if (best_permutation.empty()) {
            best_permutation = current;
            best_scores = scores;
        } else {
            bool better = false;
            bool equal = true;
            for (int i = scores.size() - 1; i >= 0; i--) {
                auto& best_score_dim = best_scores[i];
                auto& score_dim = scores[i];

                // Decide by maximal stride
                auto best_max_stride = best_score_dim.rbegin()->first;
                auto max_stride = score_dim.rbegin()->first;
                if (max_stride < best_max_stride) {
                    better = true;
                    equal = false;
                    break;
                } else if (max_stride == best_max_stride) {
                    // Decide by number of occurences
                    auto best_max_stride_count = best_score_dim.rbegin()->second;
                    auto max_stride_count = score_dim.rbegin()->second;
                    if (max_stride_count < best_max_stride_count) {
                        better = true;
                        equal = false;
                        break;
                    } else if (max_stride_count == best_max_stride_count) {
                        continue;
                    } else {
                        equal = false;
                        break;
                    }
                } else {
                    equal = false;
                    break;
                }
            }
            if (better) {
                best_permutation = current;
                best_scores = scores;
            } else if (equal) {
                // If equal and original permutation
                if (std::equal(current.begin(), current.end(), permutation.begin())) {
                    best_permutation = current;
                    best_scores = scores;
                }
            }
        }
    } while (std::next_permutation(current.begin(), current.end()));

    // Check if permutation is better than original
    return {best_permutation != permutation, best_permutation};
};

void StrideMinimization::apply(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    std::vector<structured_control_flow::ControlFlowNode*>& nested_loops,
    std::vector<std::string> target_permutation
) {
    size_t while_loops = 0;
    std::vector<std::string> permutation;
    for (auto& loop : nested_loops) {
        if (!dynamic_cast<structured_control_flow::StructuredLoop*>(loop)) {
            permutation.push_back("WHILE_" + std::to_string(while_loops++));
        } else {
            auto for_loop = dynamic_cast<structured_control_flow::StructuredLoop*>(loop);
            permutation.push_back(for_loop->indvar()->get_name());
        }
    }
    std::vector<size_t> permutation_indices;
    for (auto& p : permutation) {
        auto it = std::find(target_permutation.begin(), target_permutation.end(), p);
        permutation_indices.push_back(std::distance(target_permutation.begin(), it));
    }

    // Bubble sort permutation indices
    for (size_t i = 0; i < permutation_indices.size(); i++) {
        for (size_t j = 0; j < permutation_indices.size() - i - 1; j++) {
            if (permutation_indices[j] > permutation_indices[j + 1]) {
                auto first_loop = static_cast<structured_control_flow::StructuredLoop*>(nested_loops.at(j));
                auto second_loop = static_cast<structured_control_flow::StructuredLoop*>(nested_loops.at(j + 1));
                transformations::LoopInterchange loop_interchange(*first_loop, *second_loop);
                if (!loop_interchange.can_be_applied(builder, analysis_manager)) {
                    throw std::runtime_error("Loop interchange cannot be applied");
                }
                loop_interchange.apply(builder, analysis_manager);
                std::swap(permutation_indices[j], permutation_indices[j + 1]);
            }
        }
    }
};

std::string StrideMinimization::name() { return "StrideMinimization"; };

bool StrideMinimization::run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    bool applied = false;

    auto& loop_tree_analysis = analysis_manager.get<analysis::LoopAnalysis>();

    // Collect outermost loops
    std::vector<structured_control_flow::ControlFlowNode*> outer_loops = loop_tree_analysis.outermost_loops();

    // Apply stride minimization
    for (auto& outer_loop : outer_loops) {
        auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
        auto paths = loop_analysis.loop_tree_paths(outer_loop);
        for (auto& path : paths) {
            auto [applicable, permutation] = this->can_be_applied(builder, analysis_manager, path);
            if (applicable) {
                this->apply(builder, analysis_manager, path, permutation);
                applied = true;
                break;
            }
        }
    }

    return applied;
};

} // namespace normalization
} // namespace passes
} // namespace sdfg
