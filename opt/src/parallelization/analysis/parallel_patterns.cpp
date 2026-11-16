#include "sdfg/parallelization/analysis/parallel_patterns.h"

#include <algorithm>
#include <cassert>
#include <limits>

#include "sdfg/analysis/memory_layout_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/structured_control_flow/block.h"

namespace sdfg::parallelization {

ParallelPatterns::ParallelPatterns(StructuredSDFG& sdfg) : Analysis(sdfg) {
}

void ParallelPatterns::run(analysis::AnalysisManager& analysis_manager) {
    analysis_manager_ = &analysis_manager;
    neighborhoods_.clear();
}

const std::optional<StencilNeighborhood>& ParallelPatterns::stencil(structured_control_flow::Map& map) {
    assert(analysis_manager_ != nullptr && "ParallelPatterns: run() must be called before queries");
    auto it = neighborhoods_.find(&map);
    if (it == neighborhoods_.end()) {
        it = neighborhoods_.emplace(&map, detect(map)).first;
    }
    return it->second;
}

std::optional<StencilNeighborhood> ParallelPatterns::detect(structured_control_flow::Map& map) {
    StencilNeighborhood result;
    auto* current = &map;
    while (true) {
        if (!current->is_monotonic() || !symbolic::eq(current->stride(), symbolic::one())) {
            return std::nullopt;
        }
        for (auto* outer : result.axes) {
            if (symbolic::uses(current->init(), outer->indvar()) ||
                symbolic::uses(current->condition(), outer->indvar())) {
                return std::nullopt;
            }
        }
        result.axes.push_back(current);
        auto& body = current->root();
        if (body.size() == 1) {
            if (auto* inner = dynamic_cast<structured_control_flow::Map*>(&body.at(0))) {
                current = inner;
                continue;
            }
        }
        break;
    }

    auto& users = analysis_manager_->get<analysis::Users>();
    auto locals = users.locals(map.root());
    analysis::UsersView body_users(users, map.root());
    auto& layout = analysis_manager_->get<analysis::MemoryLayoutAnalysis>();
    struct Access {
        std::string container;
        data_flow::Subset index;
        bool write;
    };
    std::vector<Access> accesses;
    auto& body = current->root();
    for (size_t i = 0; i < body.size(); ++i) {
        auto* block = dynamic_cast<structured_control_flow::Block*>(&body.at(i));
        if (!block) {
            return std::nullopt;
        }
        for (auto& node : block->dataflow().nodes()) {
            if (!dynamic_cast<data_flow::AccessNode*>(&node) && !dynamic_cast<data_flow::Tasklet*>(&node)) {
                return std::nullopt;
            }
        }
        for (auto& edge : block->dataflow().edges()) {
            if (dynamic_cast<data_flow::ConstantNode*>(&edge.src())) {
                continue;
            }
            auto* source = dynamic_cast<data_flow::AccessNode*>(&edge.src());
            auto* destination = dynamic_cast<data_flow::AccessNode*>(&edge.dst());
            if ((source && destination) || edge.type() != data_flow::Computational) {
                return std::nullopt;
            }
            if (!source && !destination) {
                continue;
            }
            auto& container = source ? source->data() : destination->data();
            bool write = destination != nullptr;
            if (sdfg_.type(container).type_id() == types::TypeID::Scalar) {
                if (write && !locals.count(container)) {
                    return std::nullopt;
                }
                continue;
            }
            auto* access = layout.access(edge);
            if (access && !access->is_point()) {
                return std::nullopt;
            }
            auto matches_axes = [&](const data_flow::Subset& index) {
                if (index.size() != result.axes.size()) {
                    return false;
                }
                for (size_t d = 0; d < index.size(); ++d) {
                    auto remainder = symbolic::expand(symbolic::sub(index[d], result.axes[d]->indvar()));
                    for (auto* axis : result.axes) {
                        if (symbolic::uses(remainder, axis->indvar())) {
                            return false;
                        }
                    }
                    for (const auto& symbol : SymEngine::free_symbols(*remainder)) {
                        if (!body_users.writes(symbol->__str__()).empty()) {
                            return false;
                        }
                    }
                }
                return true;
            };
            auto index = edge.subset();
            if (!matches_axes(index)) {
                if (!access || !matches_axes(access->min_subset)) {
                    return std::nullopt;
                }
                index = access->min_subset;
            }
            accesses.push_back({container, std::move(index), write});
        }
    }
    for (auto& access : accesses) {
        if (!access.write) {
            continue;
        }
        if (result.output_index.empty()) {
            result.output_index = access.index;
        }
        for (size_t d = 0; d < access.index.size(); ++d) {
            if (!symbolic::eq(access.index[d], result.output_index[d])) {
                return std::nullopt;
            }
        }
        result.outputs.insert(access.container);
    }
    if (result.outputs.empty()) {
        return std::nullopt;
    }
    for (auto& access : accesses) {
        if (access.write) {
            continue;
        }
        NeighborhoodOffset offset;
        for (size_t d = 0; d < access.index.size(); ++d) {
            auto difference = symbolic::expand(symbolic::sub(access.index[d], result.output_index[d]));
            if (!SymEngine::is_a<SymEngine::Integer>(*difference)) {
                return std::nullopt;
            }
            auto value = SymEngine::down_cast<const SymEngine::Integer&>(*difference).as_int();
            if (value == std::numeric_limits<int64_t>::min()) {
                return std::nullopt;
            }
            offset.push_back(value);
        }
        bool shifted = std::ranges::any_of(offset, [](int64_t value) {
            return value != 0;
        });
        if (result.outputs.count(access.container) && shifted) {
            return std::nullopt;
        }
        auto& reads = result.inputs[access.container];
        if (reads.offsets.empty()) {
            reads.lower_halo.assign(offset.size(), 0);
            reads.upper_halo.assign(offset.size(), 0);
        }
        reads.offsets.insert(offset);
        for (size_t d = 0; d < offset.size(); ++d) {
            reads.lower_halo[d] = std::max(reads.lower_halo[d], -offset[d]);
            reads.upper_halo[d] = std::max(reads.upper_halo[d], offset[d]);
        }
    }
    bool neighborhood = false;
    for (auto& [container, reads] : result.inputs) {
        neighborhood |= reads.offsets.size() > 1 &&
                        std::ranges::any_of(reads.offsets, [](const NeighborhoodOffset& offset) {
                            return std::ranges::any_of(offset, [](int64_t value) {
                                return value != 0;
                            });
                        });
    }
    return neighborhood ? std::optional(std::move(result)) : std::nullopt;
}

} // namespace sdfg::parallelization
